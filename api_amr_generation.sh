#!/bin/bash

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

#MODEL_CARD=${1:-"gemini-3.1-flash-lite"}
MODEL_CARD=${1:-"gemini-3.5-flash"}

echo "Running AMR Generation Experiment with: $MODEL_CARD"

python -m scripts.exp_amr_generation_gemini "$MODEL_CARD"