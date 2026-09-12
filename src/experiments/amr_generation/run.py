from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_amr_generation_prompt


def run_amr_generation(
    sample: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    

    
    result = {
        "interpretation_id": sample["interpretation_id"],
        "ambiguous_caption": sample["ambiguous_caption"],
        "interpretation": sample["interpretation"],
        "image_path": sample["image_path"],
        "gold_amr": sample["formalism"],
    }

    try:
        prompt = build_amr_generation_prompt(sample)

        generated_amr = model.generate(
            prompt=prompt,
            image_path=sample["image_path"],
        )

        result["generated_amr"] = generated_amr
        result["error"] = None

    except Exception as e:
        result["generated_amr"] = None
        result["error"] = str(e)

    

    return result