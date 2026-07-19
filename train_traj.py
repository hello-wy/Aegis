import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import MotionFromGazeConfig
from dataset import gimo_dataset
from model.aegis import TrajPredictor,load_compatible_state_dict
from utils.logger import MetricTracker, create_logger


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        self.model = TrajPredictor(self.config).to(self.device)
        if self.config.load_model_dir is not None:
            self._load_checkpoint(self.config.load_model_dir)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        os.makedirs(self.config.save_path, exist_ok=True)
        self.logger = create_logger(self.config.save_path)
        self.train_metrics = MetricTracker('loss_trans', 'loss_des_trans')
        self.test_metrics = MetricTracker('loss_trans', 'loss_des_trans')
        self.best_results = {
            'loss_trans': float('inf'),
            'loss_des_trans': float('inf'),
        }
        self.best_epoch = -1

    def _load_checkpoint(self, ckpt_path):
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

    def _save_checkpoint(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, name))

    def train(self):
        for epoch in range(self.config.epoch):
            self.train_metrics.reset()
            self.training(self.train_loader, is_train=True)

            self.lr_adjuster.step()
            self.train_metrics.log(self.logger, epoch, train=True)

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()

            if epoch % self.config.save_fre == 0:
                self._save_checkpoint(f'{epoch}.pth')

    def test(self, epoch):
        current_results = self.training(self.test_loader, is_train=False)
        self.test_metrics.log(self.logger, epoch, train=False)

        if all(current_results[key] < self.best_results[key] for key in current_results):
            self.best_results = current_results
            self.best_epoch = epoch
            self._save_checkpoint('best.pth')
            print(f'New best traj results at epoch {epoch}: {self.best_results}')

        self.test_metrics.reset()
        return current_results

    def training(self, dataloader, is_train):
        metrics = self.train_metrics if is_train else self.test_metrics
        self.model.train(is_train)

        for data in tqdm(dataloader, dynamic_ncols=True):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, occ = data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)
            occ = occ.to(self.device)

            ori, trans = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes, occ)
            loss_ori, loss_trans, loss_des_ori, loss_des_trans = self.calculate_loss(
                ori,
                trans,
                poses_label,
                poses_input,
            )

            if is_train:
                loss = loss_ori.mean() + loss_trans.mean() + loss_des_ori + loss_des_trans
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            batch_size = gazes.shape[0]
            metrics.update('loss_trans', loss_trans[:, self.config.input_seq_len:].mean(), batch_size)
            metrics.update('loss_des_trans', loss_des_trans, batch_size)

        return metrics.result()

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
    evaluator.train()
