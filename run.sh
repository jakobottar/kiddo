#! /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba

export CUDA_VISIBLE_DEVICES=0

# ntfy "resnet18" python main.py --epochs 5 --arch resnet18
# ntfy "resnet50" python main.py --epochs 25 --arch resnet50 --name rn50
# ntfy "convnext" python main.py --epochs 25 --arch convnext --name convnext
# ntfy "vit" python main.py --epochs 25 --arch vit --batch-size 128
ntfy "vim" python main.py --epochs 1 --arch vim # --name vim
