
CUDA_VISIBLE_DEVICES=2 python eval.py --batch_size 1 --sample_points 4096 \
 --load_model_dir /data/wuyang/output/motion_ckpts/best.pth --traj_fusion"$@"



