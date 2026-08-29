from __future__ import annotations

from typing import Any, Dict

from src.data_augmentation.prompts import (
    build_formalism_augmentation_prompt,
)
from src.models.gemini import GeminiModel


def generate_formalisms(
    sample: Dict[str, Any],
    model: GeminiModel,
) -> Dict[str, Any]:
    """
    Generate one S-expression for each interpretation
    of an ambiguous caption.

    Expected sample format:
    {
        "ambiguous_caption": "...",
        "interpretations": [
            "...",
            "..."
        ]
    }
    """

    prompt = build_formalism_augmentation_prompt(sample)
    response = model.generate(prompt=prompt)
    sample["formalism"] = response 

    return sample