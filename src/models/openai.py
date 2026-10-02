from __future__ import annotations

import base64
import mimetypes

from pathlib import Path

from openai import OpenAI


class OpenAIModel:
    def __init__(
        self,
        model_card: str,
        api_key: str,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens

        self.client = OpenAI(
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

        mime_type, _ = mimetypes.guess_type(
            image_path
        )

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
            image_data = base64.b64encode(
                f.read()
            ).decode("utf-8")

        return {
            "type": "input_image",
            "image_url": (
                f"data:{mime_type};base64,"
                f"{image_data}"
            ),
        }

    def generate(
        self,
        prompt: str,
        image_paths: str | Path | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        content = []

        if image_paths is not None:
            content.append(
                self._image_to_content(
                    image_paths
                )
            )

        content.append(
            {
                "type": "input_text",
                "text": prompt,
            }
        )

        response = self.client.responses.create(
            model=self.model_card,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_output_tokens=(
                max_output_tokens
                or self.max_output_tokens
            ),
        )

        text = response.output_text

        if not text:
            raise RuntimeError(
                "OpenAI returned no text "
                "in the response."
            )

        return text.strip()