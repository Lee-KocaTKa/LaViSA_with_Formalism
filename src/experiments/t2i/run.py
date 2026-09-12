from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_t2i_matching_prompt


def run_t2i(
    sample: list[dict[str, Any]],
    model: Any
) -> list[dict[str, Any]]:
    
    
    result = {
        #"interpretation_id": sample["interpretation_id"],
        #"ambiguous_caption": sample["ambiguous_caption"],
        #"interpretation": sample["interpretation"],
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "amr": sample["formalism"],
        #"amr2": sample["amr_2"],
        #"amr3": sample["amr_3"],
        "correct_answer": sample["correct_answer"],
    }

    try:
        prompt = build_t2i_matching_prompt(
            sample, 
            )

        answer = model.generate(
            prompt=prompt,
            image_paths=sample["image_path"],
        )

        result["model_choice"] = answer
        result["error"] = None

    except Exception as e:
        result["model_choice"] = None
        result["error"] = str(e)

    

    return result