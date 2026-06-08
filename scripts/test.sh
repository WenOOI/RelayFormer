#!/usr/bin/env bash
# Evaluate a checkpoint on the image/video test sets in configs/test_datasets.json.
base_dir="./output_test"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
./test.py \
    --model RelayFormer \
    --world_size 1 \
    --test_batch_size 2 \
    --test_data_json ./configs/test_datasets.json \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --image_size 1024 \
    --if_resizing \
    --clip_len 4 \
    --edge_mask_width 7 \
    --merge_lora \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
