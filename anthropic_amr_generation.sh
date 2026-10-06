#!/bin/bash

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

#MODEL_CARD=${1:-"gemini-3.1-flash-lite"}
MODEL_CARD=${1:-"gemini-3.5-flash"}
#MODEL_CARD=${1:-"claude-sonnet-5"}
#MODEL_CARD=${1:-"claude-opus-5-5"}
MODEL_CARD=${1:-"gpt-5.6-terra"}

echo "Running AMR Generation Experiment with: $MODEL_CARD"

#python -m scripts.exp_amr_generation_anthropic "$MODEL_CARD"
#python -m scripts.exp_ablation_anthropic "$MODEL_CARD"
#python -m scripts.exp_ablation_gemini "$MODEL_CARD"
python -m scripts.exp_ablation_openai "$MODEL_CARD"