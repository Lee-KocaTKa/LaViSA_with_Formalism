from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_yes_or_no_prompt


def run_yes_or_no(
    sample: list[dict[str, Any]],
    model: Any,
    correct_or_incorrect: bool 
) -> list[dict[str, Any]]:
    
    
    result = {
        #"interpretation_id": sample["interpretation_id"],
        #"ambiguous_caption": sample["ambiguous_caption"],
        #"interpretation": sample["interpretation"],
        "image_path": sample["image_path"],
        "gold_amr": sample["formalism"],
        "correct_or_incorrect": correct_or_incorrect
    }

    try:
        prompt = build_yes_or_no_prompt(sample)

        answer = model.generate(
            prompt=prompt,
            image_path=sample["image_path"],
        )

        result["yes_or_no"] = answer
        result["error"] = None

    except Exception as e:
        result["yes_or_no"] = None
        result["error"] = str(e)

    

    return result