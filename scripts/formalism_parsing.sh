#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_CARD=${1:-"gemini-3.1-flash-lite"}

echo "Running formalism augmentation with: $MODEL_CARD"

PYTHONPATH="$PROJECT_ROOT" python scripts/formalism_parsing.py "$MODEL_CARD"