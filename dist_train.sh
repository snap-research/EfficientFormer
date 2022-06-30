#!/usr/bin/env bash

MODEL=$1
nGPUs=$2

python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model $MODEL \
--data-path /path/to/imagenet \
--output_dir efficientformer_l1_300d
