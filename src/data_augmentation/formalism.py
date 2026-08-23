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

    prompts = build_formalism_augmentation_prompt(sample)

    interpretations = sample["interpretations"]

    if len(prompts) != len(interpretations):
        raise ValueError(
            "The number of prompts does not match "
            "the number of interpretations."
        )

    formalism_results = []

    for interpretation, prompt in zip(
        interpretations,
        prompts,
    ):
        response = model.generate(prompt=prompt)

        formalism_results.append(
            {
                "interpretation": interpretation,
                "formalism": response,
            }
        )

    return {
        "ambiguous_caption": sample["ambiguous_caption"],
        "formalisms": formalism_results,
    }