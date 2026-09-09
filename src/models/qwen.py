from __future__ import annotations

from pathlib import Path

import torch

from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


class QwenModel:
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

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_card,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()

    def _load_image(
        self,
        image_path: str | Path,
    ) -> Image.Image:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return Image.open(image_path).convert("RGB")

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        content = []

        if image_path is not None:
            image = self._load_image(image_path)

            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False 
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(
                    max_output_tokens
                    or self.max_output_tokens
                ),
            )

        generated_tokens = outputs[
            :,
            inputs["input_ids"].shape[-1]:
        ]

        response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return response.strip()