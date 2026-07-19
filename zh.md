# AGEIS

AGEIS 是一个用于 3D 人体运动预测的研究代码库，利用 gaze、场景上下文和轨迹线索预测未来人体运动。该实现基于 AEGIS 风格的训练与评估流程，同时提供主运动预测模型和面向 robot 场景的两阶段 trajectory / motion pipeline。

本仓库面向希望复现、评估或扩展该模型的研究者与开发者。本文档说明项目结构、环境配置、外部资源、数据准备、预训练权重以及常用训练/评估命令。

英文文档: [README.md](README.md)

> **说明**
> 当前代码仍保留了一些来自开发环境的 legacy 默认路径。为了在新机器上稳定复现，建议在命令中显式传入数据、模型和输出路径，或直接修改脚本/配置中的默认值。

---

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [外部资源](#外部资源)
- [数据准备](#数据准备)
- [预训练权重](#预训练权重)
- [训练与评估](#训练与评估)
- [已知路径与配置说明](#已知路径与配置说明)
- [参考资料](#参考资料)

---

## 项目结构

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

主要入口：

| 任务 | Shell 脚本 | Python 入口 |
|---|---|---|
| 主模型训练 | `scripts/train.sh` | `train.py` |
| 主模型评估 | `scripts/eval_mod.sh` | `eval_mod.py` |
| Robot 轨迹/运动训练 | `scripts/train_traj.sh` | `train_traj.py` |
| Robot 轨迹/运动评估 | `scripts/eval_traj.sh` | `eval_traj.py` |

---

## 环境配置

### 推荐平台

本项目面向 Linux + NVIDIA GPU 环境。训练、评估和 PointNet++ CUDA 算子都依赖 CUDA。

推荐基础环境：

```text
Linux
Python 3.10
PyTorch 1.13.1
CUDA 11.7
torchvision 0.14.1
```

创建 conda 环境：

```bash
conda create -n ageis python=3.10 -y
conda activate ageis
```

安装 PyTorch + CUDA 11.7：

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

验证 PyTorch 与 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

安装常用 Python 依赖：

```bash
pip install numpy scipy pandas tqdm trimesh einops easydict termcolor tabulate
pip install smplx human-body-prior
```

仓库中也提供了 `requirements.txt` 和 `environment.yml`。它们可以作为参考，但 `environment.yml` 可能包含机器相关的包或 channel。建议先安装 PyTorch/CUDA，再手动安装其余依赖。

### PointNet++ CUDA 扩展

`model/pointnet_plus2.py` 依赖 PointNet++ CUDA 算子：

```python
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
```

一种常见安装方式：

```bash
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
```

安装后检查：

```bash
python -c "from pointnet2_ops import pointnet2_utils; from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG; print('PointNet++ OK')"
```

如果编译失败，请检查 CUDA toolkit、`nvcc`、PyTorch CUDA 版本、编译器版本和 Python 环境是否一致。建议在最终 PyTorch 环境安装完成后重新编译 PointNet++。

---

## 外部资源

### VPoser

代码默认需要 VPoser 1.0 目录：

```text
vposer_v1_0/
```

关键文件包括：

```text
vposer_v1_0/snapshots/TR00_E096.pt
vposer_v1_0/TR00_004_00_WO_accad.ini
vposer_v1_0/vposer_smpl.py
vposer_v1_0/version.txt
```

默认参数：

```bash
--vposer_path vposer_v1_0
```

同时需要安装 Python 包 `human-body-prior`。

### SMPL-X

代码通过以下方式创建 SMPL-X body model：

```python
smplx.create(smplx_path, model_type='smplx', gender='neutral', ext='npz')
```

因此目录结构应满足：

```text
<smplx_path>/smplx/SMPLX_NEUTRAL.npz
```

如果使用仓库内的默认布局，可传入：

```bash
--smplx_path ./data
```

- https://smpl-x.is.tuebingen.mpg.de/
- https://github.com/vchoutas/smplx

---

## 数据准备

当前实现使用 GIMO 风格数据和缓存张量。通过以下参数指定数据根目录：

```bash
--dataroot /path/to/GIMO
```

`config/config.py` 中默认使用：

```text
--dataset_csv dataset_with_motion_label
--folder 318_4096
```

当前 data loader 期望的最小缓存目录结构如下：

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

如果需要重新预处理、场景变换、penetration 指标、mesh 评估或可视化，还可能需要 GIMO 原始资源，例如 scene mesh、sequence transform JSON、`scene_obj/`、`eye_pc/`、`smplx_local/` 等。

GIMO 资源：

- https://github.com/y-zheng18/GIMO
- https://geometry.stanford.edu/projects/gimo/

---

## 预训练权重

如果提供预训练 checkpoint，建议放置在：

```text
weights/
├── traj_ckpts/
│   ├── best.pth
│   └── best_128.pth
└── motion_ckpts/
    └── 45.pth
```

用途：

| 权重 | 用途 |
|---|---|
| `weights/traj_ckpts/best.pth` | 主 AEGIS 模型中的 trajectory predictor |
| `weights/traj_ckpts/best_128.pth` | Robot `ProgressivePredictor` legacy 硬编码 trajectory checkpoint |
| `weights/motion_ckpts/45.pth` | 示例 motion checkpoint，用于评估 |

Checkpoint 的公开分发可能受数据集或模型许可证限制。如果不能公开分发预训练权重，请单独提供下载说明，并将 `weights/` 排除在公开仓库之外。

---

## 训练与评估

所有命令建议从仓库根目录执行：

```bash
cd AGEIS
```

请将 `/path/to/GIMO` 替换为实际 GIMO 数据根目录。


### Stage 1：轨迹训练

```bash
bash scripts/train_traj.sh 1
```

这会训练独立的 `TrajPredictor`，并把 checkpoint 写到 `outputs/temp_ckpts/`。

### Stage 1：轨迹评估

评估保存的 `best.pth` checkpoint：

```bash
bash scripts/eval_traj.sh
```

直接运行 Python：

```bash
CUDA_VISIBLE_DEVICES=0 python eval_traj.py \
  --batch_size 1 \
  --sample_points 4096 \
  --load_model_dir outputs/temp_ckpts/best.pth \
  --modelName TrajPredictor \
  --dataroot /path/to/GIMO \
  --occ
```



### 主模型训练

```bash
bash scripts/train.sh 0 \
  --dataroot /path/to/GIMO \
  --smplx_path ./data \
  --vposer_path vposer_v1_0 \
  --traj_ckpts weights/traj_ckpts/best.pth \
  --save_path outputs/motion_ckpts
```


### 主模型评估


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

## 参考资料

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
