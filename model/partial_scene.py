import torch
import torch.nn as nn
from utils.coord_helps import scene_to_local, rot_scene
import trimesh
import os
import numpy as np

class HumanGazeSceneUpSample(nn.Module):
    def __init__(self, config):
        super(HumanGazeSceneUpSample, self).__init__()
        self.nh = config.seg
        self.is_cube = config.cube
        self.fps = config.fps
        self.beta = config.beta
        self.adaptive = config.adaptive
        self.canonical = config.canonical
        self.region_size = config.region_size
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        
    def cutHumanScene_parallel(self, 
                                scene_data, 
                                pelvis_seq, 
                                num_points=None,
                                is_cube=True): # "cube" or "cylinder"
        """
        Vectorized version of cutHumanScene_adapt that keeps computations on torch tensors so
        every sample in the batch can be processed in parallel.

        Args:
            scene_data: (B, N, 3) torch tensor or numpy array. B can be 1.
            pelvis_seq: (B, T, 3) torch tensor or numpy array describing pelvis trajectory.
            num_points: number of sampled points per scene. Defaults to self.nh.
        """
        num_points = num_points or self.nh
        TRAJ_PAD = 2.0
        REGION_SIZE = 10

        B, N, _ = scene_data.shape
        pelvis_xz = pelvis_seq[:, :, [0, 2]]
        traj_max = pelvis_xz.max(dim=1).values
        traj_min = pelvis_xz.min(dim=1).values
        traj_size = traj_max - traj_min
        traj_size = traj_size + TRAJ_PAD * torch.exp(-traj_size)

        pad = ((REGION_SIZE - traj_size) / 2).clamp_min(0.0)
        center = (traj_max + traj_min) / 2
        center_region_max = center + pad
        center_region_min = center - pad
        sample_xy = (center_region_min + center_region_max) / 2
        sample_region_max = sample_xy + REGION_SIZE / 2
        sample_region_min = sample_xy - REGION_SIZE / 2

        scene_xz = scene_data[:, :, [0, 2]]
        if is_cube:
            point_in_region = (scene_xz >= sample_region_min.unsqueeze(1)) & \
                              (scene_xz <= sample_region_max.unsqueeze(1))
            point_in_region = point_in_region.all(dim=-1)
        else:
            # For cylinder, check if points are within the radius in the xz plane
            center_xy = (sample_region_min + sample_region_max) / 2
            radius = REGION_SIZE / 2
            dist = (scene_xz - center_xy.unsqueeze(1)).norm(dim=-1)
            point_in_region = dist <= radius

        # rerun program if any sample has no points in the region
        if torch.any(point_in_region.sum(dim=1) == 0):
            raise ValueError("No points in the region for at least one sample.")

        weights = point_in_region.float()
        replace_mask = weights.sum(dim=1) < num_points
        indices = torch.empty(B, num_points, dtype=torch.long, device=scene_data.device)
        if torch.any(replace_mask):
            indices[replace_mask] = torch.multinomial(weights[replace_mask], num_points, replacement=True)
        if torch.any(~replace_mask):
            indices[~replace_mask] = torch.multinomial(weights[~replace_mask], num_points, replacement=False)

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, scene_data.size(-1))
        sampled = torch.gather(scene_data, 1, gather_idx)

        return sampled

    def cutScene_adaptive(self, scene_data, traj_data, num_points=None,beta=0.5):
        """
            Args:
                scene_data: (B,step, N, 3) torch tensor or numpy array. B can be 1.
                traj_data: (B, step, unit, 3) torch tensor or numpy array describing pelvis trajectory.
                num_points: number of sampled points per scene. Defaults to self.nh.
        """
        B, T, _, _ = traj_data.shape
        num_points = num_points or self.nh
        REGION_SIZE = self.region_size
        
        vel_vec = (traj_data[:, :, [0], :] - traj_data[:, :, [-1], :]) * self.fps
        vel = torch.norm(vel_vec, dim=-1)       # [B,step, 1]
        
        radius = REGION_SIZE + self.beta * vel   # (B, T)
        center_xz = traj_data[:,:,-1,[0,2]]

        scene_xz = scene_data[:, :, [0, 2]]
        center_xz_flat = center_xz.reshape(B*T, 2)      # (BT, 2)
        radius_flat = radius.reshape(B*T)
        
        if self.is_cube:
            # Square region: |x - cx| <= r, |z - cz| <= r
            min_xy = center_xz_flat - radius_flat.unsqueeze(-1)  # (BT, 2)
            max_xy = center_xz_flat + radius_flat.unsqueeze(-1)  # (BT, 2)

            point_in_region = (scene_xz >= min_xy.unsqueeze(1)) & \
                            (scene_xz <= max_xy.unsqueeze(1))   # (BT, N, 2)
            point_in_region = point_in_region.all(dim=-1)         # (BT, N)
        else:
            # Circular region: sqrt((x-cx)^2 + (z-cz)^2) <= r
            dist = (scene_xz - center_xz_flat.unsqueeze(1)).norm(dim=-1)  # (BT, N)
            point_in_region = dist <= radius_flat.unsqueeze(-1)           # (BT, N)

        if torch.any(point_in_region.sum(dim=1) == 0):
            raise ValueError("No points in the region for at least one (B, T) sample.")
        weights = point_in_region.float()          # (BT, N)
        replace_mask = weights.sum(dim=1) < num_points  # (BT,)

        indices = torch.empty(B*T, num_points, dtype=torch.long, device=scene_data.device)

        if torch.any(replace_mask):
            indices[replace_mask] = torch.multinomial(
                weights[replace_mask], num_points, replacement=True
            )
        if torch.any(~replace_mask):
            indices[~replace_mask] = torch.multinomial(
                weights[~replace_mask], num_points, replacement=False
            )

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, scene_data.size(-1))  # (BT, num_points, 3)
        sampled = torch.gather(scene_data, 1, gather_idx)                  # (BT, num_points, 3)

        return sampled
    
    def cut_by_step(self, scene_data, traj_data, unit, ori=None, num_points=None):
        """
        scene_data: B,N,3
        traj_data: B,T,3
        ori: B,3 (optional, required if self.canonical is True)
        """
        num_points = num_points or self.nh

        bs, npts, _ = scene_data.shape
        if self.output_seq_len % unit != 0:
            raise ValueError(f"output_seq_len={self.output_seq_len} must be divisible by unit={unit}.")

        future_start = self.input_seq_len
        future_end = future_start + self.output_seq_len
        if traj_data.shape[1] < future_end:
            raise ValueError(
                f"traj_data has length {traj_data.shape[1]}, but cut_by_step needs at least {future_end} frames."
            )

        num_segments = self.output_seq_len // unit
        segment_starts = torch.arange(num_segments, device=traj_data.device) * unit + future_start
        segment_ends = segment_starts + unit
        pelvis_segments = []
        for start, end in zip(segment_starts.tolist(), segment_ends.tolist()):
            pelvis_segments.append(traj_data[:, start:end])
        pelvis_segments = torch.stack(pelvis_segments, dim=1)  # B, S, unit, 3
        
        scene_repeated = scene_data.unsqueeze(1).expand(-1, num_segments, -1, -1)
        scene_repeated = scene_repeated.reshape(bs * num_segments, npts, 3)
        if self.adaptive:
            sampled = self.cutScene_adaptive(scene_repeated, pelvis_segments, num_points, self.is_cube)
        else:
            pelvis_repeated = pelvis_segments.reshape(bs * num_segments, unit, 3)
            sampled = self.cutHumanScene_parallel(scene_repeated, pelvis_repeated, num_points, self.is_cube)
        
        sampled = sampled.view(bs, num_segments, num_points, 3)
        
        if self.canonical:
            if ori is None:
                raise ValueError("ori must be provided when canonical is True")
            reference_idx = segment_ends - 1
            sampled_flat = sampled.view(bs * num_segments, num_points, 3)
            trans_ref = traj_data[:, reference_idx].reshape(bs * num_segments, 3)
            ori_ref = ori[:, reference_idx].reshape(bs * num_segments, 3)
            sampled_flat = scene_to_local(sampled_flat, trans_ref, ori_ref)
            sampled = sampled_flat.view(bs, num_segments, num_points, 3)
        
        return sampled

def human_centered_scene(xyz, ori, trans):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        ori: input global orient of the human, [B, 3]
        trans: input translation of the human, [B, 3]
    Return:
        new_xyz: normalized points position data, [B, N, 3]
    """
    bs, _ = ori.shape
    sin_ori = torch.sin(ori[:, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, 1]).unsqueeze(-1)
    xyz = xyz - trans.unsqueeze(1)        # [B, N, 3]

    # First column is the X-axis transform, second for Y, and the third column for Z
    M = torch.cat([torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
                   torch.cat([torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)], dim=-1),
                   torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1)], dim=-1)
    M = M.view(bs, 3, 3)
    new_xyz = torch.matmul(xyz, M)

    return new_xyz

class EgoSceneExtractor(nn.Module):
    def __init__(self, config):
        super(EgoSceneExtractor, self).__init__()
        self.nh = config.seg
        self.is_cube = config.cube      #False
        self.fps = config.fps
        self.beta = config.beta     #0.5
        self.adaptive = config.adaptive     #false
        self.canonical = config.canonical
        self.unit = config.out_unit
    
    def cutScene_adaptive(self, scene_data, traj_data, num_points=None, beta=0.5):
        """
        The motion input within the window is fixed to 8 frames.
        Args:
            scene_data: (B, N, 3) torch tensor or numpy array. B can be 1.
            traj_data: (B, step, 3) torch tensor or numpy array describing pelvis trajectory.
            ori: (B, 3)
            num_points: number of sampled points per scene. Defaults to self.nh.
        """
        # 1. Preprocess parameters
        num_points = num_points or self.nh
        REGION_SIZE = 1.0
        B, _, _ = traj_data.shape
        
        current_pos = traj_data[:, -1]        # (B, 3)
        prev_pos = traj_data[:, 0]           # (B, 3)
        
        vel = torch.norm((prev_pos - current_pos) * (self.fps), dim=-1)
        
        radius = (REGION_SIZE + self.beta * vel).unsqueeze(1) 

        center_xz = current_pos[:, [0, 2]]      # (B, 2) y-up coordinate system
        scene_xz = scene_data[:, :, [0, 2]]     # (B, N, 2)

        if self.is_cube:
            # Square region optimization: use abs() to avoid two comparisons
            # |x - cx| <= r  AND  |z - cz| <= r
            diff = torch.abs(scene_xz - center_xz.unsqueeze(1)) # (B, N, 2)
            # If any dimension exceeds the radius, the result is False; use all(dim=-1) after comparison
            point_in_region = (diff <= radius.unsqueeze(-1)).all(dim=-1)      # (B, N)
        else:
            # Circular region optimization: compare squared distances to avoid sqrt
            # (x-cx)^2 + (z-cz)^2 <= r^2
            dist_sq = (scene_xz - center_xz.unsqueeze(1)).pow(2).sum(dim=-1) # (B, N)
            radius_sq = radius.pow(2).squeeze(1) # (B,) -> pay attention to dimensions for broadcasting comparison
            point_in_region = dist_sq <= radius_sq.unsqueeze(1) # (B, N)

        valid_counts = point_in_region.sum(dim=1)
        if (valid_counts == 0).any():
            raise ValueError("No points in the region for at least one sample.")

        # 6. Sampling logic
        weights = point_in_region.float() # (B, N)
        indices = torch.empty((B, num_points), dtype=torch.long, device=scene_data.device)
        
        # Determine which batches need sampling with replacement (insufficient points)
        replace_mask = valid_counts < num_points # (B,)
        
        # Branch 1: insufficient points, must sample with replacement (Replacement=True)
        if replace_mask.any():
            # Perform multinomial sampling only on the rows that need it
            indices[replace_mask] = torch.multinomial(
                weights[replace_mask], num_points, replacement=True
            )
            
        # Branch 2: sufficient points, sample without replacement (Replacement=False)
        if (~replace_mask).any():
            # Perform sampling without replacement only on the rows that need it
            indices[~replace_mask] = torch.multinomial(
                weights[~replace_mask], num_points, replacement=False
            )
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, 3)
        
        return torch.gather(scene_data, 1, gather_idx)
    
    def forward(self, scene_data, traj_data, ori):
        """
        window traj data: B,T,3   
        traj_data: B,2,3    Only two trajectory coordinates are needed to compute an instantaneous velocity
        scene_data: B,N,3
        ori: B,3    Only the current frame orientation is needed
        """
        sampled_scene = self.cutScene_adaptive(scene_data, traj_data)  # B,nh,3        
        if self.canonical:
            sampled_scene = scene_to_local(sampled_scene, traj_data[:,-1], ori)
        
        return sampled_scene

def visualize_points(point1):
    pcd = trimesh.PointCloud(vertices=point1)
    npoints, _ = point1.shape
    colors = np.ones((npoints, 3)) * 255

    colors[:, 0] = 0  # R channel
    colors[:, 1] = 255   # G channel
    colors[:, 2] = 0    # B channel
    
    pcd.visual.vertex_colors = colors
    pcd.export(f"/data/wuyang/points.obj")

def visualize_Scene_wo_color(scene_data,name="scene"):
    scene_data = scene_data.cpu().numpy()
    output_path = "/data/wuyang"
    # ndarray
    # Input: scene_data [bs npoints 3] Or [npoints 3]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if len(scene_data.shape) == 3:
        bs, npoints, _ = scene_data.shape
    elif len(scene_data.shape) == 2:
        npoints, _ = scene_data.shape
        scene_data = scene_data.reshape(1,npoints,3)
        bs = 1
    for i in range(bs):
        pcd = trimesh.PointCloud(vertices=scene_data[i, :,:3])
        
        pcd.export(f"{output_path}/{name}{i}.obj")
