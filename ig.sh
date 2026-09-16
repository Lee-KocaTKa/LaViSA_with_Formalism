#!/bin/bash

#SBATCH --job-name=amr_qwen35_27b
#SBATCH --partition=public
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 python -m scripts.ig_expl \
    --model Qwen/Qwen3.5-4B \
    --image "/mnt/home/sangmyeong-l/research/LaViSA_with_Formalism/data/lavisa/original/images/vp/vp-11-b-i.png" \
    --amr "(a / argue-01 :ARG0 (g / girl) :ARG1 (b / boy :ARG0-of (h / hold-01 :ARG1 (c / cake :mod (b2 / birthday))))))" \
    --steps 50 \
    --output outputs/ig/vp-11-b