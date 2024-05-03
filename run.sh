#! /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba

export CUDA_VISIBLE_DEVICES=0

ntfy "resnet50" python main.py --model resnet50 --epochs 100 
