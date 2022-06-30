#!/usr/bin/env bash

MODEL=$1
nGPUs=$2
CHECKPOINT=$3

python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model $MODEL \
--resume $CHECKPOINT --eval \
--data-path /path/to/imagenet \
--output_dir efficientformer_test
