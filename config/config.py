from argparse import ArgumentParser


class MotionFromGazeConfig(ArgumentParser):
    def __init__(self):
        super().__init__()

        self.input_configs = self.add_argument_group('input')
        self.input_configs.add_argument('--batch_size', default=16, type=int)
        self.input_configs.add_argument('--num_workers', default=4, type=int)
        self.input_configs.add_argument('--input_seq_len', default=6, type=int)
        self.input_configs.add_argument('--output_seq_len', default=10, type=int)
        self.input_configs.add_argument('--dataroot', default='/data/wuyang/human/GIMO/', type=str)
        self.input_configs.add_argument('--fps', default=2, type=int)
        self.input_configs.add_argument('--sample_points', default=4096, type=int)

        self.add_argument('--model_type', default='cross', choices=['cross', 'mint', 'rnn', 'st_transformer', 'simple'])
        self.motion_configs = self.add_argument_group('motion_transformer')
        self.motion_configs.add_argument('--dropout', default=0.3, type=float)
        self.motion_configs.add_argument('--motion_n_heads', default=8, type=int)
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int)
        self.motion_configs.add_argument('--motion_n_layers', default=3, type=int)
        self.motion_configs.add_argument('--motion_latent_dim', default=256, type=int)
        self.motion_configs.add_argument('--scene_feats_dim', default=256, type=int)
        self.motion_configs.add_argument('--num_heads', default=4, type=int)

        self.gaze_configs = self.add_argument_group('gaze_transformer')
        self.gaze_configs.add_argument('--gaze_n_heads', default=8, type=int)
        self.gaze_configs.add_argument('--gaze_hidden_dim', default=256, type=int)
        self.gaze_configs.add_argument('--gaze_n_layers', default=3, type=int)

        self.train_configs = self.add_argument_group('train')
        self.train_configs.add_argument('--save_path', type=str, default='/data/wuyang/output/motion_ckpts')
        self.train_configs.add_argument('--save_fre', type=int, default=1)
        self.train_configs.add_argument('--val_fre', type=int, default=1)
        self.train_configs.add_argument('--load_model_dir', type=str, default=None)
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None)

        self.train_configs.add_argument('--epoch', type=int, default=150)
        self.train_configs.add_argument('--lr', type=float, default=3e-4)
        self.train_configs.add_argument('--weight_decay', type=float, default=1e-4)
        self.train_configs.add_argument('--gamma', type=float, default=0.98)
        self.train_configs.add_argument('--activation', type=str, default='gelu')
        self.train_configs.add_argument('--traj_vel_loss_weight', type=float, default=0.1)
        self.train_configs.add_argument('--traj_acc_loss_weight', type=float, default=0.05)

        self.eval_configs = self.add_argument_group('eval')
        self.eval_configs.add_argument('--output_path', default='results', type=str)
        self.eval_configs.add_argument('--smplx_path', default='/data/wuyang/smpl_model', type=str)
        self.eval_configs.add_argument('--vposer_path', default='vposer_v1_0', type=str)
        self.eval_configs.add_argument('--modelName', default='', type=str)
        self.eval_configs.add_argument('--traj_ckpts', default='/data/wuyang/output/traj_ckpts/best.pth', type=str)
        self.eval_configs.add_argument('--dataset_csv', default='dataset_with_motion_label', type=str)
        self.eval_configs.add_argument('--is_vis', action='store_true', help='whether to visualize the results')
        
        self.eval_configs.add_argument('--beta', default=0.5, type=float, help='beta for region adjustment')
        
        # 消融实验指标 默认是不加 occ 的
        self.eval_configs.add_argument('--eval_len', default=10, type=int,help='number of frames to eval,Optional numbers[1,4,10]')
        self.eval_configs.add_argument('--occ', action='store_true', help='whether to use occupancy grid')
        self.eval_configs.add_argument('--rope', action='store_true', help='whether to use occupancy grid')
        self.eval_configs.add_argument('--out_unit', default=2, type=int)
        self.eval_configs.add_argument('--traj_fusion', action='store_true', help='whether to use trajectory fusion')
        self.eval_configs.add_argument('--seg', type=int, default=1024, help='whether to use segmented points')
        self.eval_configs.add_argument('--region_size', type=int, default=2, help='region size for scene cutting')
        self.eval_configs.add_argument('--cube', action='store_true', help='whether to use cube region')
        self.eval_configs.add_argument('--adaptive', action='store_true', help='whether to use adaptive region size')
        self.eval_configs.add_argument('--canonical', action='store_true', help='whether to use human centered scene')
        self.eval_configs.add_argument('--folder', type=str, default='318_4096', help='which folder to use for data loading')
        self.eval_configs.add_argument('--eval_scene', type=str, default='all', help='which scene to eval on, options: all, bedroom, livingroom, office, kitchen')
        
        self.mogaze_configs = self.add_argument_group('mogaze')
        self.mogaze_configs.add_argument('--mogaze_data_dir', default='/home/customer/IMP3DS/data/preprocessed_mogaze', type=str)
        
    def get_configs(self):
        return self.parse_args()


if __name__ == '__main__':
    config = MotionFromGazeConfig()
    print(config.get_configs())
