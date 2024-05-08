#! /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vim-2

export CUDA_VISIBLE_DEVICES=1

# ntfy "resnet18" python main.py --epochs 5 --arch resnet18
ntfy "resnet50" python main.py --epochs 1 --arch resnet50
# ntfy "convnext" python main.py --epochs 1 --arch convnext
# ntfy "vit" python main.py --epochs 1 --arch vit --batch-size 128
