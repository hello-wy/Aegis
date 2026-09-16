DEVICE=0

if [[ "$1" =~ ^[0-9]+$ ]]; then
    DEVICE=$1
    shift  
fi
export CUDA_VISIBLE_DEVICES=$DEVICE

python train.py --save_fre 1 --val_fre 1 --batch_size 8 --sample_points 4096 \
    --dropout 0.3 --epoch 100 --occ --traj_fusion "$@"