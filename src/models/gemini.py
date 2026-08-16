from __future__ import annotations 

from google import genai 

class GeminiModel: 
    def __init__(
        self,
        model_card: str,
        max_output_tokens: int = 128
    ) -> None: 
        