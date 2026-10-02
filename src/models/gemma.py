from __future__ import annotations

from pathlib import Path
from typing import Sequence 

import torch

from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

class GemmaModel:
    def __init__(
        self,
        model_card: str,
        max_output_tokens: int = 256, 
    ) -> None: 
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_card
        )