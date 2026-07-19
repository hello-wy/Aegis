DEVICE=1
CKPT=best

if [[ "$1" =~ ^[0-9]+$ ]]; then
    CKPT=$1
    shift
fi

if [[ "$CKPT" =~ \.pth$ ]]; then
    LOAD_MODEL_DIR="/data/wuyang/output/traj_ckpts/$CKPT"
else
    LOAD_MODEL_DIR="/data/wuyang/output/traj_ckpts/${CKPT}.pth"
fi

export CUDA_VISIBLE_DEVICES=$DEVICE

python eval_traj.py --batch_size 1 --sample_points 4096 \
 --load_model_dir "$LOAD_MODEL_DIR" \
 --modelName TrajPredictor "$@"
