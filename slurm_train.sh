#!/usr/bin/env bash

MODEL=$1

export NCCL_P2P_DISABLE=1

python run_with_submitit.py --model $MODEL \
--batch-size 128 \
--job_dir efficientformer_l1_300d \
--partition your-partition \
--ngpus 2 --nodes 4 --cpus 32 --mem 90 \
--epochs 300 \
--data-path /path/to/imagenet
