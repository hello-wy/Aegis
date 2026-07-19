import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from utils.vis_utils import latent_to_joints,visualize_Scene_wo_color
from model.base_cross_model import PerceiveEncoder
from model.pointnet_plus2 import PointNet2SemSegSSGShape, MyFPModule
from model.base_cross_model import SelfAttentionLayer
from model.partial_scene import HumanGazeSceneUpSample
import math
from utils.rotation import axis_angle_to_matrix, matrix_to_axis_angle


def load_compatible_state_dict(module, state_dict, label="checkpoint"):
    model_state = module.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = len(state_dict) - len(compatible_state)
    if skipped:
        print(f"Skipped {skipped} incompatible tensors from {label}.")
    return module.load_state_dict(compatible_state, strict=False)


def human_centered_scene(xyz, ori, trans):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        ori: input global orient of the human, [B, len, 3]
        trans: input translation of the human, [B, len, 3]
    Return:
        new_xyz: normalized points position data, [B, len, N, 3]
    """
    bs, T, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    xyz = xyz[:, None, :, :].repeat(1, T, 1, 1) - trans[:, :, None, :]        # [B, len, N, 3]

    # First column is the X-axis transform, second for Y, and the third column for Z
    M = torch.cat([torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
                   torch.cat([torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)], dim=-1),
                   torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1)], dim=-1)
    M = M.view(bs, T, 3, 3)
    new_xyz = torch.matmul(xyz, M)

    return new_xyz


class TrajPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)         # joints (23*3) and smpl latents (3+3+32)
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                            n_latent=config.output_seq_len + config.input_seq_len,
                                            n_latent_channels=config.motion_latent_dim,
                                            n_self_att_heads=config.motion_n_heads,
                                            n_self_att_layers=config.motion_n_layers,
                                            dropout=config.dropout)

        self.gaze_fp_layer = MyFPModule()
        self.gaze_encoder = PerceiveEncoder(n_input_channels=config.scene_feats_dim,
                                            n_latent=config.output_seq_len + config.input_seq_len,
                                            n_latent_channels=config.gaze_hidden_dim,
                                            n_self_att_heads=config.motion_n_heads,
                                            n_self_att_layers=config.gaze_n_layers,
                                            dropout=config.dropout)
        if self.config.occ:
            self.occupancy_emb = Mlp(128*128, config.motion_hidden_dim, drop=0.2)

        self.tia_blocks = nn.ModuleList([TIABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)

    def forward(self, motions, joints, scene_xyz, gazes, occupancy_map):
        """
        :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent]
        :param scene_xyz: (bs, n, 3)
        :param gazes: (bs, seq_len, 1, 3)
        :return:
        """
        bs, seq_len, gaze_n, _ = gazes.shape
        assert gaze_n == 1
        gazes = gazes.squeeze(2)

        scene_feats, scene_global_feats = self.pointnet(scene_xyz.repeat(1, 1, 2))      # B x 256 x 30w, B x 256
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat([motions, joints.view(bs, seq_len, -1)], dim=-1)

        if self.config.occ:
            motions = self.motion_emb(motions) + self.occupancy_emb(occupancy_map.view(bs, -1)).unsqueeze(1)
        else:
            motions = self.motion_emb(motions)

        motions = self.motion_encoder(motions)
        gaze_embedding = self.gaze_fp_layer(gazes, scene_xyz, scene_feats.transpose(1, 2).contiguous()).transpose(1, 2)
        gaze_embedding = self.gaze_encoder(gaze_embedding)

        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_feats, gaze_embedding)
        out_tia = out.clone()
        ori, trans = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)

        return ori, trans

class AEGIS(nn.Module):
    def __init__(self, config, vposer, smplx_model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.out_unit = config.out_unit
        self.use_window = True
        
        self.vposer = vposer
        self.smplx_model = smplx_model

        self.extractor = HumanGazeSceneUpSample(config)
        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        
        self.motion_emb = nn.Linear( 38, config.motion_hidden_dim)         # joints (23*3) and smpl latents (3+3+32)
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                              n_latent=self.input_seq_len+self.out_unit,   # T seq_len
                                              n_latent_channels=config.motion_latent_dim,
                                              n_self_att_heads=config.motion_n_heads,
                                              n_self_att_layers=config.motion_n_layers,
                                              dropout=config.dropout,
                                              use_rotary_pos_emb=config.rope)

        
        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))

        self.traj_predictor = TrajPredictor(config)
        self.load_ckpts()
        
        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.motion_decoder = GCN(config, node_n=69)
    
    def load_ckpts(self):
        state_dict = torch.load(self.config.traj_ckpts)
        print(self.config.traj_ckpts)
        self.traj_predictor.load_state_dict(state_dict, strict=False)
        
    def _extract_scene_features(self, scene_points):
        scene_feats, _ = self.pointnet(scene_points.repeat(1, 1, 2))
        return scene_feats.transpose(1, 2).contiguous()
    
    # TODO: 使用mlp去替代，减少性能消耗
    def _encode_motion_sequence(self, motions):
        motion_tokens = self.motion_emb(motions)
        return self.motion_encoder(motion_tokens)
    
    def _update_motion_sequence(self, motions, step):
        start = step * self.out_unit
        end = start + self.input_seq_len + self.out_unit
        return motions[:, start:end], start, end
    
    def _compute_spatial_prior(self, scene_points):
        distance = torch.norm(scene_points, dim=-1)
        distance_salience = -((distance * 1.5) ** 3)
        direction_salience = scene_points[:, :, 0] / torch.sqrt((scene_points[:, :, [0, 2]] ** 2).sum(dim=-1))
        base_prior = self.spatial_att(scene_points)
        return distance_salience * self.dist_factor + direction_salience * self.dir_factor + base_prior

    def _apply_sca_blocks(self, motion, scene_feats, spatial_prior):
        out = motion.clone()
        for block in self.sca_blocks:
            out = block(out, scene_feats, spatial_prior)
        return out.clone()
    
    def combine_smplx_rt(
        self,
        trans_global:    torch.Tensor,  # (B, T, 3) 全局平移
        trans_residual:  torch.Tensor,  # (B, T, 3) 平移残差
        ori_global:      torch.Tensor,  # (B, T, 3) 全局旋转 (axis-angle, SMPLX global_orient)
        ori_residual:    torch.Tensor,  # (B, T, 3) 旋转残差 (axis-angle)
    ):
        """
        合并 SMPLX 的全局平移/旋转 与 残差，得到最终平移 & 最终旋转（axis-angle）

        返回:
            final_trans:   (B, T, 3) 最终平移 = trans_global + trans_residual
            final_orient:  (B, T, 3) 最终旋转 (axis-angle)
        """
        # 1. 平移：逐元素相加即可
        final_trans = trans_global + trans_residual       # (B, T, 3)

        # 2. 旋转：先把两个 axis-angle 转成矩阵
        Rg = axis_angle_to_matrix(ori_global)             # (B, T, 3, 3) 全局
        Rr = axis_angle_to_matrix(ori_residual)           # (B, T, 3, 3) 残差

        # 假设“最终旋转 = 残差 * 全局”
        R_final = torch.matmul(Rr, Rg)                    # (B, T, 3, 3)

        # 再从矩阵转回 axis-angle
        final_orient = matrix_to_axis_angle(R_final)      # (B, T, 3)

        return final_trans, final_orient

    def forward(self, motions, joints, scene_xyz, gazes,occ):
        """
        :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent] dim:38
        :param scene_xyz: (bs, n, 3)
        :param gazes: unused; kept for compatibility with existing dataloaders/scripts
        :return:
        """
        B, _, J, C = joints.shape
        # with torch.no_grad():
        ori, trans = self.traj_predictor(motions, joints, scene_xyz, gazes, occ)     # B,T,3  B,T,3
        
        # motions_cat = torch.cat([motions, joints.view(B, 6, -1)], dim=-1)   # B，6，69 + 38
        motion_extend = motions[:, [-1]].repeat(1, self.output_seq_len, 1)
        motions_cat = torch.cat([motions, motion_extend], dim=1)        #B，16，69 + 38

        human_points = self.extractor.cut_by_step(scene_xyz, trans, self.out_unit, ori)      # B,10 // unit,N,3
        scene_feats_batch = self._extract_scene_features(human_points.view(B * self.output_seq_len // self.out_unit, -1, 3))        # [B, num_points, feat_dim]
        scene_feats_batch = scene_feats_batch.view(B, self.output_seq_len // self.out_unit, -1, self.config.scene_feats_dim)
        # scene_feats = self._extract_scene_features(scene_xyz) 
        # print(scene_feats_batch.shape)
        latent_full = motions_cat.clone()
        for step in range(self.output_seq_len // self.out_unit):
            scene_feats = scene_feats_batch[:, step]
            
            motions_cat, start, end = self._update_motion_sequence(latent_full, step)
            motions = self._encode_motion_sequence(motions_cat)  #B,8,dim

            #scene_xyz_normalized = human_centered_scene(human_points[:, step],ori[:, start:end],trans[:,start:end])          # bs * len * N * 3
            spatial_prior = self._compute_spatial_prior(human_points[:, step])
            out_sca = self._apply_sca_blocks(motions, scene_feats, spatial_prior)
            pose = self.pose_fc(out_sca)
            
            latent_pred = torch.cat([ori[:, start:end], trans[:, start:end], pose], dim=-1) 
            # Keep the full sequence length fixed; only replace the current future block.
            latent_full = torch.cat(
                [latent_full[:, :end - self.out_unit], latent_pred[:, -self.out_unit :], latent_full[:, end:]],
                dim=1,
            )
          
        recons_joints = latent_to_joints(latent_full, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()
        pred_joints = recons_joints[:, :, :23]
        
        if self.config.traj_fusion:    
            pred_joints = self.pose_traj_fusion(trans.unsqueeze(2), pred_joints)
            
        pred_joints = self.motion_decoder(pred_joints)
        return latent_full, pred_joints
        
    def pose_traj_fusion(self, traj, joints):
        predictions = joints.clone()
        distance = predictions[:,:,[0],:] - traj
        predictions =  predictions - distance
        predictions[:, :, [0], :] = traj
        return predictions

class TIABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 3, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, gaze_embedding):
        motion_tia = self.motion_encoder(motion)
        sm_feature = self.ia_attention(motion_tia, scene_feats)
        out = torch.cat([motion_tia, sm_feature, gaze_embedding], dim=-1)
        out = self.mlp(out)
        return out


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)        # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))       # bs * head * len * N
        att = F.softmax(att, dim=-1)                                # bs * head * N
        out = torch.matmul(att, v).view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, spatial_prior):
        """
        motion: B,T,256
        scene_feats: B,N,256
        spatial_prior: B,N
        """
        motion_sca = self.motion_encoder(motion)    # B，T，256
        sm_feature = self.sa_attention(motion_sca, scene_feats, spatial_prior)  # B，T，256
        out = torch.cat([motion_sca, sm_feature], dim=-1)   # B，T，512
        out = self.mlp(out) #B,T,256
        return out


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats, spatial_prior):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)           # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

        #TODO: scaled dot-product attention
        att = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.feat_dim)       # bs * head * len * N
        att = att + spatial_prior[:, None, None, :]
        att = F.softmax(att, dim=-1)                         # bs * head * len * N
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz):
        out = F.relu(self.fc1(xyz))
        spatial_prior = self.fc2(out)
        return spatial_prior.squeeze(-1)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out
