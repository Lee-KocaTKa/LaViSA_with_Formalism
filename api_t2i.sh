#!/bin/bash

set -e

cd /mnt/home/sangmyeong-l/research/LaViSA_with_Formalism

#MODEL_CARD=${1:-"gemini-3.1-flash-lite"}
MODEL_CARD=${1:-"gemini-3.5-flash"}
MODEL_CARD=${1:-"claude-opus-5-5"}
#MODEL_CARD=${1:-"claude-sonnet-5"}

echo "Running AMR Generation Experiment with: $MODEL_CARD"

python -m scripts.api_t2i "$MODEL_CARD"