#!/bin/bash

#SBATCH --job-name=amr_qwen35_27b
#SBATCH --partition=public
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

MODEL_CARD="Qwen/Qwen3.6-27B"
#MODEL_CARD="google/gemma-4-E4B-it"

echo "================================"
echo "Node:"
hostname

echo "CUDA_VISIBLE_DEVICES:"
echo "$CUDA_VISIBLE_DEVICES"

echo "GPU status:"
nvidia-smi

echo "PyTorch CUDA status:"
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

if torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"Expected 2 GPUs, found {torch.cuda.device_count()}"
    )

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "================================"
echo "Starting experiment"
echo "Model: $MODEL_CARD"
echo "================================"

#python -m scripts.exp_amr_generation "$MODEL_CARD"
python -m scripts.exp_ablation "$MODEL_CARD"