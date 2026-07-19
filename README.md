# AGEIS

AGEIS is a research codebase for 3D human motion forecasting that uses gaze, scene context, and trajectory cues to predict future human motion. The implementation is based on a AEGIS-style training and evaluation workflow, and provides both the main motion forecasting model and a robot-oriented two-stage trajectory / motion pipeline.

This repository is intended for researchers and developers who want to reproduce, evaluate, or extend the model. This document describes the project structure, environment setup, external assets, data preparation, pretrained weights, and common training/evaluation commands.

中文版文档: [zh.md](zh.md)

> **Note**
> The current code still keeps some legacy default paths from the development environment. For stable reproduction on a new machine, explicitly pass data, model, and output paths in commands, or directly update the defaults in scripts/configuration.

---

## Contents

- [Project structure](#project-structure)
- [Environment setup](#environment-setup)
- [External assets](#external-assets)
- [Data preparation](#data-preparation)
- [Pretrained weights](#pretrained-weights)
- [Training and evaluation](#training-and-evaluation)
- [References](#references)

---

## Project structure

```text
AGEIS/
├── README.md
├── zh.md
├── requirements.txt
├── environment.yml
├── train.py
├── eval_mod.py
├── train_traj.py
├── eval_traj.py
├── scripts/
│   ├── train.sh
│   ├── eval_mod.sh
│   ├── train_traj.sh
│   └── eval_traj.sh
├── config/
│   └── config.py
├── dataset/
│   └── gimo_dataset.py
├── model/
├── utils/
├── vposer_v1_0/
├── data/
│   └── smplx/
└── weights/
    ├── traj_ckpts/
    │   ├── best.pth
    │   └── best_128.pth
    ├── motion_ckpts/
    │   └── 45.pth
    └── temp_ckpts/
```

Main entry points:

| Task | Shell script | Python entry |
|---|---|---|
| Main model training | `scripts/train.sh` | `train.py` |
| Main model evaluation | `scripts/eval_mod.sh` | `eval_mod.py` |
| Robot trajectory/motion training | `scripts/train_traj.sh` | `train_traj.py` |
| Robot trajectory/motion evaluation | `scripts/eval_traj.sh` | `eval_traj.py` |

---

## Environment setup

### Recommended platform

This project is designed for Linux + NVIDIA GPU environments. Training, evaluation, and PointNet++ CUDA operators all depend on CUDA.

Recommended baseline environment:

```text
Linux
Python 3.10
PyTorch 1.13.1
CUDA 11.7
torchvision 0.14.1
```

Create a conda environment:

```bash
conda create -n ageis python=3.10 -y
conda activate ageis
```

Install PyTorch + CUDA 11.7:

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Verify PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Install common Python dependencies:

```bash
pip install numpy scipy pandas tqdm trimesh einops easydict termcolor tabulate
pip install smplx human-body-prior
```

The repository also provides `requirements.txt` and `environment.yml`. They can be used as references, but `environment.yml` may contain machine-specific packages or channels. It is recommended to install PyTorch/CUDA first, then manually install the remaining dependencies.

### PointNet++ CUDA extension

`model/pointnet_plus2.py` depends on PointNet++ CUDA operators:

```python
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
```

One common installation method:

```bash
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
```

Check after installation:

```bash
python -c "from pointnet2_ops import pointnet2_utils; from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG; print('PointNet++ OK')"
```

If compilation fails, check whether the CUDA toolkit, `nvcc`, PyTorch CUDA version, compiler version, and Python environment are consistent. It is recommended to rebuild PointNet++ after the final PyTorch environment has been installed.

---

## External assets

### VPoser

The code expects a VPoser 1.0 directory by default:

```text
vposer_v1_0/
```

Key files include:

```text
vposer_v1_0/snapshots/TR00_E096.pt
vposer_v1_0/TR00_004_00_WO_accad.ini
vposer_v1_0/vposer_smpl.py
vposer_v1_0/version.txt
```

Default argument:

```bash
--vposer_path vposer_v1_0
```

The Python package `human-body-prior` is also required.

### SMPL-X

The code creates the SMPL-X body model as follows:

```python
smplx.create(smplx_path, model_type='smplx', gender='neutral', ext='npz')
```

Therefore, the directory structure should satisfy:

```text
<smplx_path>/smplx/SMPLX_NEUTRAL.npz
```

If using the repository-local default layout, pass:

```bash
--smplx_path ./data
```

- https://smpl-x.is.tuebingen.mpg.de/
- https://github.com/vchoutas/smplx

---

## Data preparation

The current implementation uses GIMO-style data and cached tensors. Specify the data root with:

```bash
--dataroot /path/to/GIMO
```

`config/config.py` uses the following defaults:

```text
--dataset_csv dataset_with_motion_label
--folder 318_4096
```

The minimum cached directory layout expected by the current data loader is:

```text
/path/to/GIMO/
├── dataset_with_motion_label.csv
├── 318_4096/
│   ├── train/
│   │   ├── gazes.pth
│   │   ├── poses_input.pth
│   │   ├── poses_label.pth
│   │   ├── joints_input.pth
│   │   ├── joints_label.pth
│   │   └── occ_map.pth
│   └── test/
│       ├── gazes.pth
│       ├── poses_input.pth
│       ├── poses_label.pth
│       ├── joints_input.pth
│       ├── joints_label.pth
│       └── occ_map.pth
└── 315_32768/
    ├── train/
    │   └── scene_points_32768.pth
    └── test/
        └── scene_points_32768.pth
```

If you need to rerun preprocessing, scene transforms, penetration metrics, mesh evaluation, or visualization, additional original GIMO assets may also be required, such as scene meshes, sequence transform JSON files, `scene_obj/`, `eye_pc/`, `smplx_local/`, and so on.

GIMO resources:

- https://github.com/y-zheng18/GIMO
- https://geometry.stanford.edu/projects/gimo/

---

## Pretrained weights

If pretrained checkpoints are provided, it is recommended to place them under:

```text
weights/
├── traj_ckpts/
│   ├── best.pth
│   └── best_128.pth
└── motion_ckpts/
    └── 45.pth
```

Usage:

| Weight | Purpose |
|---|---|
| `weights/traj_ckpts/best.pth` | Trajectory predictor in the main AEGIS model |
| `weights/traj_ckpts/best_128.pth` | Legacy hard-coded trajectory checkpoint for the robot `ProgressivePredictor` |
| `weights/motion_ckpts/45.pth` | Example motion checkpoint for evaluation |

Public redistribution of checkpoints may be restricted by dataset or model licenses. If pretrained weights cannot be publicly redistributed, provide separate download instructions and exclude `weights/` from the public repository.

---

## Training and evaluation

It is recommended to run all commands from the repository root:

```bash
cd AGEIS
```

Replace `/path/to/GIMO` with the actual GIMO data root.

### Stage 1: trajectory training

```bash
bash scripts/train_traj.sh 1
```

This trains the standalone `TrajPredictor` and writes checkpoints to `outputs/temp_ckpts/`.

### Stage 1: trajectory evaluation

Evaluate the saved `best.pth` checkpoint:

```bash
bash scripts/eval_traj.sh
```

Run Python directly:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_traj.py \
  --batch_size 1 \
  --sample_points 4096 \
  --load_model_dir outputs/temp_ckpts/best.pth \
  --modelName TrajPredictor \
  --dataroot /path/to/GIMO \
  --occ
```

### Main model training

```bash
bash scripts/train.sh 0 \
  --dataroot /path/to/GIMO \
  --smplx_path ./data \
  --vposer_path vposer_v1_0 \
  --traj_ckpts weights/traj_ckpts/best.pth \
  --save_path outputs/motion_ckpts
```

### Main model evaluation

```bash
bash scripts/eval_mod.sh \
  --load_model_dir weights/motion_ckpts/45.pth \
  --traj_ckpts weights/traj_ckpts/best.pth \
  --dataroot /path/to/GIMO \
  --smplx_path ./data \
  --vposer_path vposer_v1_0 \
  --eval_len 10 \
  --occ
```

---

## References

- AEGIS official repository: https://github.com/kjle6/AEGIS-master
- AEGIS CVPR 2024 paper: https://openaccess.thecvf.com/content/CVPR2024/html/Lou_Multimodal_Sense-Informed_Forecasting_of_3D_Human_Motions_CVPR_2024_paper.html
- GIMO repository: https://github.com/y-zheng18/GIMO
- GIMO project page: https://geometry.stanford.edu/projects/gimo/
- SMPL-X repository: https://github.com/vchoutas/smplx
- SMPL-X model download: https://smpl-x.is.tuebingen.mpg.de/
- human_body_prior / VPoser: https://github.com/nghorbani/human_body_prior
- PointNet++ PyTorch: https://github.com/erikwijmans/Pointnet2_PyTorch
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- PyTorch3D installation: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- Open3D documentation: https://www.open3d.org/docs/release/getting_started.html
- Trimesh documentation: https://trimesh.org/install
