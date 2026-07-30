#!/usr/bin/env bash
# Evaluate a video checkpoint on video test sets (e.g. MOSE100).
# Matches the released video checkpoint (checkpoint-88.pth) eval configuration.
base_dir="./output_test_video"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
./test.py \
    --model RelayFormer \
    --world_size 1 \
    --test_batch_size 2 \
    --test_data_json ./configs/test_video_datasets.json \
    --checkpoint_path /data/huangwen/worklab/ps-detection/weights/checkpoint-88.pth \
    --image_size 512 \
    --patch_size 272 \
    --feature_patch_size 17 \
    --grid_size 2 \
    --tokens_per_patch 2 \
    --edge_lambda 2 \
    --vit_pretrain_path /data/huangwen/worklab/ps-detection/vit_base_patch16_224.mae.bin \
    --if_resizing \
    --clip_len 9 \
    --edge_mask_width 7 \
    --num_workers 1 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
