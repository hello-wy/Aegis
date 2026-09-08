import os
import torch
import time
import smplx
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.aegis import AEGIS, load_compatible_state_dict
from utils.logger import create_logger, MetricTracker


def expand_semantic_labels(labels, out_unit, output_seq_len):
    labels = labels.repeat_interleave(out_unit, dim=1)
    if labels.shape[1] != output_seq_len:
        raise ValueError(
            f"Semantic labels expand to {labels.shape[1]} frames, expected {output_seq_len}."
        )
    return labels


def body_scene_contact_targets(joints, scene_points, threshold, max_scene_points):
    point_count = min(scene_points.shape[1], max_scene_points)
    indices = torch.linspace(
        0, scene_points.shape[1] - 1, point_count, device=scene_points.device
    ).long()
    sampled_scene = scene_points.index_select(1, indices)
    distances = torch.cdist(joints.flatten(1, 2), sampled_scene).amin(dim=-1)
    return (distances.reshape(joints.shape[:-1]) < threshold).to(joints.dtype)


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = vposer.to(self.device)
        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.batch_size * 16,
                                       ).to(self.device)
        
        self.model = AEGIS(config, self.vposer, self.body_model).to(self.device)

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir, map_location=self.device)
            state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
            load_compatible_state_dict(self.model, state_dict, "model checkpoint")
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        # exit(0)
        self.best_results = {
            "loss_trans": float("inf"),
            "loss_des_trans": float("inf"),
            "mpjpe": float("inf"),
            "des_mpjpe": float("inf")
        }
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        # We set drop_last=True to fit the SMPL body model during training, test results could be a little inaccurate.
        # For accurate test results, run eval.sh with checkpoint.
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )

        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        os.makedirs(f"runs/{self.config.save_path}", exist_ok=True)

    def train(self):
        train_metrics = MetricTracker(
            'loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe',
            'latent', 'velocity', 'acceleration', 'semantic', 'contact',
        )

        for epoch in range(self.config.epoch):
            for data in tqdm(self.train_loader, dynamic_ncols=True):
                (
                    gazes, poses_input, poses_label, joints_input, joints_label,
                    scene_points, seq, scene, occ, semantic_labels,
                ) = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)
                semantic_labels = semantic_labels.to(self.device)

                poses_predict, joints_predict, auxiliary = self.model(
                    poses_input, joints_input[:, :, :23], scene_points, gazes, occ, return_aux=True
                )
                losses = self.calculate_training_loss(
                    poses_predict,
                    joints_predict,
                    auxiliary,
                    poses_label,
                    joints_label[:, :, :23],
                    scene_points,
                    semantic_labels,
                )
                self.optim.zero_grad()
                losses['total'].backward()
                self.optim.step()

                batch_size = gazes.shape[0]
                train_metrics.update("loss_trans", losses['translation'], batch_size)
                train_metrics.update("loss_des_trans", losses['destination_translation'], batch_size)
                train_metrics.update("mpjpe", losses['mpjpe'], batch_size)
                train_metrics.update("des_mpjpe", losses['destination_mpjpe'], batch_size)
                for name in ('latent', 'velocity', 'acceleration', 'semantic', 'contact'):
                    train_metrics.update(name, losses[name], batch_size)

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % self.config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ, _ = data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            occ = occ.to(self.device)

            poses_predict, joints_predict = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)

            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

        test_metrics.log(self.logger, epoch)
        current_results = {
            "loss_trans": test_metrics['loss_trans'],
            "loss_des_trans": test_metrics['loss_des_trans'],
            "mpjpe":test_metrics['mpjpe'],
            "des_mpjpe": test_metrics['des_mpjpe']
        }
        test_metrics.reset()
        if current_results['mpjpe'] < self.best_results['mpjpe']:
            self.best_results = current_results
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), f"{self.config.save_path}/best.pth")
            print(f"New best results at epoch {epoch}: {self.best_results}")


    def calculate_training_loss(
        self, poses_predict, joints_predict, auxiliary, poses_label, joints_label,
        scene_points, semantic_labels,
    ):
        future_pose = poses_predict[:, self.config.input_seq_len:]
        future_joints = joints_predict[:, self.config.input_seq_len:]
        pred_relative = future_joints - future_joints[:, :, [0]]
        target_relative = joints_label - joints_label[:, :, [0]]

        latent = F.l1_loss(future_pose[..., 6:], poses_label[..., 6:])
        orientation = F.l1_loss(future_pose[..., :3], poses_label[..., :3])
        translation_per_frame = torch.norm(
            future_pose[..., 3:6] - poses_label[..., 3:6], dim=-1
        )
        mpjpe_per_frame = torch.norm(pred_relative - target_relative, dim=-1).mean(dim=-1)
        velocity = F.l1_loss(
            pred_relative[:, 1:] - pred_relative[:, :-1],
            target_relative[:, 1:] - target_relative[:, :-1],
        )
        acceleration = F.l1_loss(
            pred_relative[:, 2:] - 2 * pred_relative[:, 1:-1] + pred_relative[:, :-2],
            target_relative[:, 2:] - 2 * target_relative[:, 1:-1] + target_relative[:, :-2],
        )

        frame_labels = expand_semantic_labels(
            semantic_labels, self.config.out_unit, self.config.output_seq_len
        )
        semantic = F.cross_entropy(
            auxiliary['semantic_logits'].reshape(-1, self.config.num_semantic_classes),
            frame_labels.reshape(-1),
        )
        contact_targets = body_scene_contact_targets(
            joints_label,
            scene_points,
            self.config.contact_threshold,
            self.config.contact_scene_points,
        )
        positive = contact_targets.sum().clamp_min(1)
        pos_weight = ((contact_targets.numel() - positive) / positive).clamp(1, 20)
        contact = F.binary_cross_entropy_with_logits(
            auxiliary['contact_logits'], contact_targets, pos_weight=pos_weight
        )

        mpjpe = mpjpe_per_frame.mean()
        root = orientation + translation_per_frame.mean()
        total = (
            root
            + self.config.mpjpe_loss_weight * mpjpe
            + self.config.latent_loss_weight * latent
            + self.config.velocity_loss_weight * velocity
            + self.config.acceleration_loss_weight * acceleration
            + self.config.semantic_loss_weight * semantic
            + self.config.contact_loss_weight * contact
        )
        return {
            'total': total,
            'translation': translation_per_frame.mean(),
            'destination_translation': translation_per_frame[:, -1].mean(),
            'mpjpe': mpjpe,
            'destination_mpjpe': mpjpe_per_frame[:, -1].mean(),
            'latent': latent,
            'velocity': velocity,
            'acceleration': acceleration,
            'semantic': semantic,
            'contact': contact,
        }

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)             # bs * seq * J

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    print("ckpts save path:"+ config.save_path)
    evaluator = SMPLX_evalutor(config)
    evaluator.train()
