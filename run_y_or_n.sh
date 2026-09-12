#!/bin/bash

#SBATCH --job-name=amr_qwen35_27b
#SBATCH --partition=public
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

MODEL_CARD="Qwen/Qwen3.5-27B"

echo "Running on:"
hostname



python -m scripts.exp_y_or_n "$MODEL_CARD"