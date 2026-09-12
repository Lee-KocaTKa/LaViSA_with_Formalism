from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_i2t_matching_prompt


def run_i2t(
    sample: list[dict[str, Any]],
    model: Any
) -> list[dict[str, Any]]:
    
    
    result = {
        #"interpretation_id": sample["interpretation_id"],
        #"ambiguous_caption": sample["ambiguous_caption"],
        #"interpretation": sample["interpretation"],
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "amr1": sample["amr_1"],
        "amr2": sample["amr_2"],
        "amr3": sample["amr_3"],
        "correct_answer": sample["correct_answer"],
    }

    num_option = 3 if result["amr3"] != None else 2 

    try:
        prompt = build_i2t_matching_prompt(
            sample,
            num_option=num_option 
            )

        answer = model.generate(
            prompt=prompt,
            image_path=sample["image_path"],
        )

        result["model_choice"] = answer
        result["error"] = None

    except Exception as e:
        result["model_choice"] = None
        result["error"] = str(e)

    

    return result