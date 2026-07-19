import trimesh
from easydict import EasyDict
from typing import List
import numpy as np
import os
SKELETON_CHAIN = EasyDict({
    'SMPL': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
    },
    'SMPLH': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]],
        'left_hand_chain': [[20, 34, 35, 36], [20, 22, 23, 24], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]],
        'right_hand_chain': [[21, 49, 50, 51], [21, 37, 38, 39], [21, 40, 41, 42], [21, 46, 47, 48],[21, 43, 44, 45]],
    },
    'SMPLX': {
        'kinematic_chain': [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20], [15, 23], [15, 24]],
        'left_hand_chain': [[20, 37, 38, 39], [20, 25, 26, 27], [20, 28, 29, 30], [20, 34, 35, 36], [20, 31, 32, 33]],
        'right_hand_chain': [[21, 52, 53, 54], [21, 40, 41, 42], [21, 43, 44, 45], [21, 49, 50, 51], [21, 46, 47, 48]],
    },
    'CUSTOM': {
        'kinematic_chain': [
            [0, 1, 2, 3, 4],       # spine
            [2, 5, 6, 7, 8],       # left arm
            [2, 9, 10, 11, 12],    # right arm
            [0, 13, 14, 15, 16],   # left leg
            [0, 17, 18, 19, 20],   # right leg
        ]
    },
    'GTA': {
        'kinematic_chain': [
            [0, 1, 2, 3, 4, 5],        # head -> neck -> right arm
            [1, 6, 7, 8, 9],           # neck -> left arm
            [1, 10, 11, 12, 13, 14],   # neck -> spine
            [14, 15, 16, 17],          # spine4 -> right leg
            [14, 18, 19, 20],          # spine4 -> left leg
        ]
    }
})

JOINTS_NAME = EasyDict({
    'SMPLH': [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ]
})

colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]



def latent_to_joints(pose_latent, vposer, smplx_model):
    bs, T, c = pose_latent.shape
    body_pose = vposer.decode(pose_latent[:, :, 6:], output_type='aa').view(-1, 63)
    smplx_output = smplx_model(return_verts=True, body_pose=body_pose,
                               global_orient=pose_latent[:, :, :3].view(-1, 3),
                               transl=pose_latent[:, :, 3:6].view(-1, 3),
                               pose_embedding=pose_latent[:, :, 6:].view(-1, 32))
    joints = smplx_output.joints
    return joints.reshape(bs, T, 127, 3)

def visualize_points(point1):
    pcd = trimesh.PointCloud(vertices=point1)
    npoints, _ = point1.shape
    colors = np.ones((npoints, 3)) * 255

    colors[:, 0] = 0  # R 通道
    colors[:, 1] = 255   # G 通道
    colors[:, 2] = 0    # B 通道
    
    pcd.visual.vertex_colors = colors
    pcd.export(f"/data/wuyang/points.obj")

def visualize_Scene_wo_color(scene_data,name="scene",k=0):
    scene_data = scene_data.cpu().numpy()
    output_path = "/data/wuyang"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if len(scene_data.shape) == 3:
        bs, npoints, _ = scene_data.shape
    elif len(scene_data.shape) == 2:
        npoints, _ = scene_data.shape
        scene_data = scene_data.reshape(1,npoints,3)
        bs = 1
    for i in range(bs):
        pcd = trimesh.PointCloud(vertices=scene_data[i, :, :3])
        pcd.export(f"{output_path}/{name}{i}_{k}.obj")

        
def skeleton_to_mesh(skeleton: np.ndarray, kinematic_chain: List, njoints: int=22) -> List:
    """ Convert skeleton to meshes

    Args:
        skeleton: skeleton array, joints position, <L, njoints, 3>
        kinematic_chain: kinematic chain, can be none
        njoints: joints number
    
    Return:
        Skeleton mesh list
    """
    meshes = []
    if kinematic_chain is None:
        for f in range(skeleton.shape[0]):
            joints = skeleton[f]
            joints_mesh = []
            for i, joint in enumerate(joints):
                joint_mesh = trimesh.creation.uv_sphere(radius=0.005)
                joint_mesh.apply_translation(joint)
                joints_mesh.append(joint_mesh)
            meshes.append(trimesh.util.concatenate(joints_mesh))
    else:
        colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
        meshes = []
        def plot3D(joints, color=None, linewidth=0.01) -> trimesh.Trimesh:
            if color is not None and color.startswith('#'):
                color = trimesh.visual.color.hex_to_rgba(color)
            else:
                color = np.array([128, 0.0, 0.0, 255], dtype=np.uint8)
            
            lines = []
            for i in range(len(joints) - 1):
                line = trimesh.creation.cylinder(
                    radius=linewidth,
                    segment=joints[i:i+2],
                )
                line.visual.vertex_colors = color
                lines.append(line)
            
            return trimesh.util.concatenate(lines)

        for f in range(skeleton.shape[0]):
            joints = skeleton[f]
            joints_mesh = []
            for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
                if i < 5:
                    linewidth = 0.02
                else:
                    linewidth = 0.01
                
                lines = plot3D(joints[chain], color=color, linewidth=linewidth)
                joints_mesh.append(lines)
            
            meshes.append(trimesh.util.concatenate(joints_mesh))
    
    return meshes

def visualize_SMPLXjoints(joints, pc_xyz,i, output_path="/data/wuyang/MM/output/vis_debug"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    kinematic_chain = SKELETON_CHAIN.SMPLX['kinematic_chain']
    
    skeleton = joints[:,:,:].reshape(-1,23,3)
    meshes = skeleton_to_mesh(skeleton, kinematic_chain, 23)
    scene = trimesh.Scene(meshes)
    scene.export(f"{output_path}/{i}_skeleton.obj")
    if pc_xyz is not None:
        pcd = trimesh.PointCloud(vertices=pc_xyz)
        pcd.export(f"{output_path}/{i}_scene.obj")
