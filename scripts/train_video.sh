#!/usr/bin/env bash
# Pure VIDEO training at 512 resolution with resize preprocessing.
# Matches the training config of the released video checkpoint (checkpoint-88.pth).
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
    --batch_size 8 \
    --find_unused_parameters \
    --data_path ./configs/train_video.json \
    --test_data_path ./configs/test_video_datasets.json \
    --epochs 200 \
    --lr 1e-4 \
    --min_lr 5e-7 \
    --vit_pretrain_path /data/huangwen/worklab/ps-detection/vit_base_patch16_224.mae.bin \
    --image_size 512 \
    --patch_size 272 \
    --feature_patch_size 17 \
    --grid_size 2 \
    --tokens_per_patch 2 \
    --edge_lambda 2 \
    --if_resizing \
    --clip_len 5 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --num_workers 1 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
