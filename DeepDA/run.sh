#!/usr/bin/env bash
GPU_ID=0
data_dir=/home/data/office31
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python dsan.py --config ./DSAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam