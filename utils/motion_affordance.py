from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

# SMPL-X body edges restricted to the first 23 joints used by this repo.
SMPLX_BODY_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 22),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
)

DEFAULT_JOINT_WEIGHTS: Tuple[float, ...] = (
    1.00, 0.90, 0.90, 0.85, 0.95, 0.95, 0.85, 1.05, 1.05, 0.90, 1.10, 1.10,
    0.75, 0.70, 0.70, 0.70, 0.72, 0.72, 0.66, 0.66, 0.85, 0.85, 0.60,
)
DEFAULT_TRAIL_JOINTS: Tuple[int, ...] = (0, 7, 8, 10, 11, 20, 21)


@dataclass
class MotionAffordanceConfig:
    frame_stride: int = 1
    support_joints: Tuple[int, ...] = tuple(range(23))
    bone_edges: Tuple[Tuple[int, int], ...] = SMPLX_BODY_EDGES
    trail_joints: Tuple[int, ...] = DEFAULT_TRAIL_JOINTS
    joint_weights: Tuple[float, ...] = DEFAULT_JOINT_WEIGHTS
    bone_samples: int = 3
    trail_samples: int = 4
    temporal_start_weight: float = 0.9
    temporal_end_weight: float = 1.1
    sigma_joint_scale: float = 3.2
    sigma_bone_scale: float = 4.4
    sigma_trail_scale: float = 5.2
    vertical_sigma_scale: float = 1.8
    gamma: float = 2.0
    bone_weight: float = 0.65
    trail_weight: float = 0.45
    diffusion_k: int = 16
    diffusion_eta: float = 0.35
    diffusion_iters: int = 4
    max_points_for_diffusion: int = 8192
    clip_percentile: float = 97.0
    max_chunk_points: int = 4096
    color_low: Tuple[float, float, float] = (0.10, 0.28, 0.88)
    color_mid: Tuple[float, float, float] = (0.16, 0.75, 0.88)
    color_high: Tuple[float, float, float] = (0.20, 0.84, 0.34)
    color_midpoint: float = 0.55
    metadata: Dict[str, float] = field(default_factory=dict)


def _as_float_tensor(values: torch.Tensor | np.ndarray | Sequence[float], device: torch.device | None = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32) if device is not None else values.float()
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _linspace_indices(length: int, max_count: int, device: torch.device) -> torch.Tensor:
    if length <= max_count:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=max_count, device=device).round().long()


def estimate_point_spacing(scene_points: torch.Tensor, neighbor_rank: int = 6, max_samples: int = 1024) -> float:
    points = _as_float_tensor(scene_points)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f'scene_points must have shape [N, 3], got {tuple(points.shape)}')
    if points.shape[0] < 2:
        return 0.25

    sample_ids = _linspace_indices(points.shape[0], max_samples, points.device)
    sampled = points.index_select(0, sample_ids)
    distances = torch.cdist(sampled, sampled)
    distances.fill_diagonal_(float('inf'))
    rank = min(neighbor_rank, max(1, sampled.shape[0] - 1))
    kth = distances.topk(rank, largest=False).values[:, -1]
    spacing = kth.median().item()
    if not np.isfinite(spacing) or spacing <= 1e-6:
        return 0.25
    return spacing


def _temporal_weights(num_frames: int, config: MotionAffordanceConfig, device: torch.device) -> torch.Tensor:
    if num_frames <= 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    return torch.linspace(
        config.temporal_start_weight,
        config.temporal_end_weight,
        steps=num_frames,
        dtype=torch.float32,
        device=device,
    )


def build_motion_affordance_anchors(
    motion_joints: torch.Tensor | np.ndarray,
    config: MotionAffordanceConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    joints = _as_float_tensor(motion_joints)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f'motion_joints must have shape [T, J, 3], got {tuple(joints.shape)}')

    joints = joints[:: max(1, config.frame_stride)]
    if joints.shape[1] < 1:
        raise ValueError('motion_joints must contain at least one joint.')

    support = [joint_idx for joint_idx in config.support_joints if joint_idx < joints.shape[1]]
    if not support:
        raise ValueError('No valid support joints remain after filtering against motion_joints.')

    joint_weights = _as_float_tensor(config.joint_weights, device=joints.device)
    if joint_weights.numel() < joints.shape[1]:
        padded = torch.ones(joints.shape[1], dtype=torch.float32, device=joints.device)
        padded[: joint_weights.numel()] = joint_weights
        joint_weights = padded

    frame_weights = _temporal_weights(joints.shape[0], config, joints.device)
    anchors: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    joint_anchor = joints[:, support].reshape(-1, 3)
    joint_weight = (frame_weights[:, None] * joint_weights[support][None, :]).reshape(-1)
    anchors.append(joint_anchor)
    weights.append(joint_weight)

    if config.bone_samples > 0:
        ts = torch.linspace(0.0, 1.0, steps=config.bone_samples + 2, device=joints.device, dtype=torch.float32)[1:-1]
        for start_idx, end_idx in config.bone_edges:
            if start_idx >= joints.shape[1] or end_idx >= joints.shape[1]:
                continue
            start = joints[:, start_idx]
            end = joints[:, end_idx]
            sampled = start[:, None, :] * (1.0 - ts[None, :, None]) + end[:, None, :] * ts[None, :, None]
            anchors.append(sampled.reshape(-1, 3))
            edge_weight = 0.5 * (joint_weights[start_idx] + joint_weights[end_idx])
            weights.append((frame_weights[:, None] * (edge_weight * config.bone_weight)).reshape(-1).repeat_interleave(config.bone_samples))

    if config.trail_samples > 0 and joints.shape[0] > 1:
        ts = torch.linspace(0.0, 1.0, steps=config.trail_samples + 2, device=joints.device, dtype=torch.float32)[1:-1]
        for joint_idx in config.trail_joints:
            if joint_idx >= joints.shape[1]:
                continue
            start = joints[:-1, joint_idx]
            end = joints[1:, joint_idx]
            sampled = start[:, None, :] * (1.0 - ts[None, :, None]) + end[:, None, :] * ts[None, :, None]
            anchors.append(sampled.reshape(-1, 3))
            trail_frames = 0.5 * (frame_weights[:-1] + frame_weights[1:])
            trail_weight = trail_frames * joint_weights[joint_idx] * config.trail_weight
            weights.append(trail_weight.repeat_interleave(config.trail_samples))

    all_anchors = torch.cat(anchors, dim=0)
    all_weights = torch.cat(weights, dim=0)
    return all_anchors, all_weights


def _accumulate_field(
    scene_points: torch.Tensor,
    anchors: torch.Tensor,
    anchor_weights: torch.Tensor,
    sigma_xy: float,
    sigma_y: float,
    max_chunk_points: int,
) -> torch.Tensor:
    scores = torch.zeros(scene_points.shape[0], dtype=torch.float32, device=scene_points.device)
    anchor_xy = anchors[:, [0, 2]]
    anchor_y = anchors[:, 1]
    sigma_xy_sq = max(float(sigma_xy) ** 2, 1e-8)
    sigma_y_sq = max(float(sigma_y) ** 2, 1e-8)

    for start in range(0, scene_points.shape[0], max_chunk_points):
        end = min(start + max_chunk_points, scene_points.shape[0])
        chunk = scene_points[start:end]
        dist_xy_sq = torch.cdist(chunk[:, [0, 2]], anchor_xy).pow(2)
        dist_y_sq = (chunk[:, [1]] - anchor_y[None, :]).pow(2)
        logits = -0.5 * (dist_xy_sq / sigma_xy_sq + dist_y_sq / sigma_y_sq)
        scores[start:end] = (torch.exp(logits) * anchor_weights[None, :]).sum(dim=1)
    return scores


def _diffuse_scores(scene_points: torch.Tensor, scores: torch.Tensor, config: MotionAffordanceConfig, spacing: float) -> torch.Tensor:
    if config.diffusion_iters <= 0 or scene_points.shape[0] > config.max_points_for_diffusion:
        return scores
    if scene_points.shape[0] <= 1:
        return scores

    k = min(config.diffusion_k, scene_points.shape[0] - 1)
    if k <= 0:
        return scores

    sigma_neighbor = max(spacing * 2.0, 1e-6)
    distances = torch.cdist(scene_points, scene_points)
    distances.fill_diagonal_(float('inf'))
    knn_dist, knn_idx = distances.topk(k, largest=False)
    weights = torch.exp(-0.5 * (knn_dist / sigma_neighbor) ** 2)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    current = scores
    for _ in range(config.diffusion_iters):
        neighbor_scores = current.index_select(0, knn_idx.reshape(-1)).view_as(knn_idx)
        smooth = (neighbor_scores * weights).sum(dim=1)
        current = (1.0 - config.diffusion_eta) * current + config.diffusion_eta * smooth
    return current


def normalize_affordance(scores: torch.Tensor | np.ndarray, clip_percentile: float = 97.0) -> torch.Tensor:
    tensor = _as_float_tensor(scores)
    if tensor.numel() == 0:
        return tensor
    scale = torch.quantile(tensor, min(max(clip_percentile / 100.0, 0.0), 1.0))
    if not torch.isfinite(scale) or scale <= 1e-8:
        scale = tensor.max().clamp_min(1e-8)
    normalized = (tensor / scale).clamp(0.0, 1.0)
    return normalized


def affordance_to_colors(
    scores: torch.Tensor | np.ndarray,
    low_color: Sequence[float] = (0.10, 0.28, 0.88),
    high_color: Sequence[float] = (0.20, 0.84, 0.34),
    mid_color: Sequence[float] | None = (0.16, 0.75, 0.88),
    midpoint: float = 0.55,
) -> torch.Tensor:
    normalized = normalize_affordance(scores)
    low = _as_float_tensor(low_color, device=normalized.device)
    high = _as_float_tensor(high_color, device=normalized.device)
    if mid_color is None:
        return low[None, :] * (1.0 - normalized[:, None]) + high[None, :] * normalized[:, None]

    mid = _as_float_tensor(mid_color, device=normalized.device)
    midpoint = float(np.clip(midpoint, 1e-4, 1.0 - 1e-4))
    colors = torch.empty(normalized.shape[0], 3, dtype=torch.float32, device=normalized.device)
    left_mask = normalized <= midpoint
    right_mask = ~left_mask
    if left_mask.any():
        local = (normalized[left_mask] / midpoint).unsqueeze(1)
        colors[left_mask] = low[None, :] * (1.0 - local) + mid[None, :] * local
    if right_mask.any():
        local = ((normalized[right_mask] - midpoint) / (1.0 - midpoint)).unsqueeze(1)
        colors[right_mask] = mid[None, :] * (1.0 - local) + high[None, :] * local
    return colors


def compute_motion_affordance(
    scene_points: torch.Tensor | np.ndarray,
    motion_joints: torch.Tensor | np.ndarray,
    config: MotionAffordanceConfig,
    return_details: bool = False,
) -> torch.Tensor | Dict[str, torch.Tensor | float | Dict[str, float]]:
    points = _as_float_tensor(scene_points)
    joints = _as_float_tensor(motion_joints, device=points.device)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f'scene_points must have shape [N, 3], got {tuple(points.shape)}')
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f'motion_joints must have shape [T, J, 3], got {tuple(joints.shape)}')

    anchors, anchor_weights = build_motion_affordance_anchors(joints, config)
    spacing = estimate_point_spacing(points)
    sigma_joint = spacing * config.sigma_joint_scale
    sigma_bone = spacing * config.sigma_bone_scale
    sigma_trail = spacing * config.sigma_trail_scale
    sigma_xy = max(sigma_bone, sigma_joint, sigma_trail)
    sigma_y = sigma_xy * config.vertical_sigma_scale

    raw_scores = _accumulate_field(
        scene_points=points,
        anchors=anchors,
        anchor_weights=anchor_weights,
        sigma_xy=sigma_xy,
        sigma_y=sigma_y,
        max_chunk_points=max(256, config.max_chunk_points),
    )
    bounded_scores = 1.0 - torch.exp(-config.gamma * raw_scores)
    diffused_scores = _diffuse_scores(points, bounded_scores, config, spacing)
    normalized_scores = normalize_affordance(diffused_scores, config.clip_percentile)
    colors = affordance_to_colors(
        normalized_scores,
        low_color=config.color_low,
        high_color=config.color_high,
        mid_color=config.color_mid,
        midpoint=config.color_midpoint,
    )

    if not return_details:
        return normalized_scores

    return {
        'scores': normalized_scores,
        'colors': colors,
        'raw_scores': raw_scores,
        'anchors': anchors,
        'anchor_weights': anchor_weights,
        'spacing': spacing,
        'parameters': {
            **asdict(config),
            'sigma_joint': sigma_joint,
            'sigma_bone': sigma_bone,
            'sigma_trail': sigma_trail,
            'sigma_xy': sigma_xy,
            'sigma_y': sigma_y,
        },
    }


def save_colored_point_cloud_obj(
    points: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    output_path: str | Path,
    title: str = 'motion affordance',
) -> None:
    xyz = _as_float_tensor(points).detach().cpu().numpy()
    rgb = _as_float_tensor(colors).detach().cpu().numpy().clip(0.0, 1.0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        handle.write(f'# {title}\n')
        handle.write('# Per-vertex colors encoded as RGB in [0, 1].\n')
        for point, color in zip(xyz, rgb):
            handle.write(
                f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                f'{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n'
            )


def save_affordance_npz(
    output_path: str | Path,
    scene_points: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    motion_joints: torch.Tensor | np.ndarray,
    anchors: torch.Tensor | np.ndarray | None = None,
    anchor_weights: torch.Tensor | np.ndarray | None = None,
    parameters: Dict[str, float] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'scene_points': _as_float_tensor(scene_points).detach().cpu().numpy(),
        'scores': _as_float_tensor(scores).detach().cpu().numpy(),
        'colors': _as_float_tensor(colors).detach().cpu().numpy(),
        'motion_joints': _as_float_tensor(motion_joints).detach().cpu().numpy(),
    }
    if anchors is not None:
        payload['anchors'] = _as_float_tensor(anchors).detach().cpu().numpy()
    if anchor_weights is not None:
        payload['anchor_weights'] = _as_float_tensor(anchor_weights).detach().cpu().numpy()
    if parameters is not None:
        payload['parameters'] = np.array([parameters], dtype=object)
    np.savez_compressed(output, **payload)


__all__ = [
    'MotionAffordanceConfig',
    'SMPLX_BODY_EDGES',
    'affordance_to_colors',
    'build_motion_affordance_anchors',
    'compute_motion_affordance',
    'estimate_point_spacing',
    'normalize_affordance',
    'save_affordance_npz',
    'save_colored_point_cloud_obj',
]
