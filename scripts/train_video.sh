#!/usr/bin/env bash
# Pure VIDEO training.
base_dir="./output_video"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
./train.py \
    --model RelayFormer \
    --world_size 1 \
    --batch_size 2 \
    --find_unused_parameters \
    --data_path ./configs/train_video.json \
    --test_data_path ./configs/test_datasets.json \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 1024 \
    --if_resizing \
    --clip_len 4 \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 4 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
