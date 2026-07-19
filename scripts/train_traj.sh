DEVICE=3

if [[ "$1" =~ ^[0-9]+$ ]]; then
    DEVICE=$1
    shift
fi

export CUDA_VISIBLE_DEVICES=$DEVICE

python train_traj.py --batch_size 8 --sample_points 4096 \
 --epoch 100 --dropout 0.4 \
 --save_path /data/wuyang/output/temp_ckpts/ \
 --modelName TrajPredictor "$@"
