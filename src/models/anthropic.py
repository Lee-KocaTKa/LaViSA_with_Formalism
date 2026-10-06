from __future__ import annotations

import base64
import mimetypes

from anthropic import Anthropic

from pathlib import Path


class ClaudeModel:
    def __init__(
        self,
        model_card: str,
        api_key: str,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens

        self.client = Anthropic(
            api_key=api_key
        )

    def _image_to_content(
        self,
        image_path: str | Path,
    ) -> dict:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(image_path)

        if mime_type not in {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }:
            raise ValueError(
                f"Unsupported image type: {mime_type}"
            )

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(
                f.read()
            ).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_data,
            },
        }

    def generate(
        self,
        prompt: str,
        image_paths: str | Path | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        content = []

        if image_paths is not None:
            if isinstance(image_paths, (str, Path)):
                image_paths = [image_paths]
            
            for image_path in image_paths:
                content.append(
                    self._image_to_content(image_path)
                )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        response = self.client.messages.create(
            model=self.model_card,
            max_tokens=(
                max_output_tokens
                or self.max_output_tokens
            ),
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        texts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]

        if not texts:
            raise RuntimeError(
                "Claude returned no text in the response."
            )

        return "\n".join(texts).strip()