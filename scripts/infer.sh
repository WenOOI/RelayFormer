#!/usr/bin/env bash
# Localize manipulations in a folder of images / videos.
python ./infer.py \
    --model_path /path/to/checkpoint-best.pth \
    --input_dir  /path/to/input \
    --output_dir ./output_infer \
    --device cuda \
    --input_size 1024 \
    --clip_len 4 \
    --mask_threshold 0.5
