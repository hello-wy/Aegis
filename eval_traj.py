import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.config import MotionFromGazeConfig
from dataset import gimo_dataset
from model.aegis import TrajPredictor,load_compatible_state_dict
from utils.logger import MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        self.model = TrajPredictor(self.config).to(self.device)
        self._load_checkpoint(self.config.load_model_dir)

        self.test_metrics = MetricTracker('loss_trans', 'loss_des_trans')

    def _load_checkpoint(self, ckpt_path):
        if ckpt_path is None:
            raise ValueError('load_model_dir must be provided for trajectory evaluation.')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        print('loading pretrained model from ', ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        if not isinstance(state_dict, dict):
            raise TypeError(f'Unsupported checkpoint format in {ckpt_path!r}.')

        state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if ('smplx' not in k and 'vposer' not in k)}
        load_compatible_state_dict(self.model, state_dict, 'traj checkpoint')
        print('load done!')

    def eval(self):
        self.model.eval()
        self.test_metrics.reset()

        with torch.no_grad():
            for data in self.test_loader:
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                occ = occ.to(self.device)

                ori, trans = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)
                _, loss_trans, _, loss_des_trans = self.calculate_loss(
                    ori,
                    trans,
                    poses_label,
                    poses_input,
                )

                self.test_metrics.update('loss_trans', loss_trans[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                self.test_metrics.update('loss_des_trans', loss_des_trans, gazes.shape[0])

        results = self.test_metrics.result()
        print(results)
        self.test_metrics.reset()
        return results

    def calculate_loss(self, ori, trans, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(ori[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(trans[:, -1] - poses_label[:, -1, 3:6], dim=-1).mean()

        loss_all = F.l1_loss(ori, poses_label[:, :, :3], reduction='none')
        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(trans - poses_label[:, :, 3:6], dim=-1)

        return loss_ori, loss_trans, loss_des_ori, loss_des_trans


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    evaluator = SMPLX_evalutor(config)
    evaluator.eval()
