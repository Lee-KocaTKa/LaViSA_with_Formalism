from __future__ import annotations 

import mimetypes

from google import genai 
from google.genai import types 

from pathlib import Path 
from typing import Any 

class GeminiModel: 
    def __init__(
        self,
        model_card: str,
        api_key: str,
        max_output_tokens: int = 128
    ) -> None: 
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=api_key) 
        
    def _image_to_part(self, image_path: str | Path) -> types.Part:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            raise ValueError(f"Could not determine MIME type for: {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )