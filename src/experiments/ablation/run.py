from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_ablation_generation_prompt


def run_amr_generation(
    samples: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    
    results = [] 
    
    for i in range(len(samples)):
        sample = samples[i]  
    
        result = {
            "interpretation_id": sample["interpretation_id"],
            "ambiguous_caption": sample["ambiguous_caption"],
            "interpretation": sample["interpretation"],
            "image_path": sample["image_path"],
            "gold_amr": sample["formalism"],
        }

        if i == 0:
            try:
                prompt = build_ablation_generation_prompt(sample)

                generated_amr = model.generate(
                    prompt=prompt,
                    image_paths=None,
                )

                result["generated_amr"] = generated_amr
                result["error"] = None

            except Exception as e:
                result["generated_amr"] = None
                result["error"] = str(e)
                
        else: 
            result["generated_amr"] = generated_amr
            result["error"] = None
            
        results.append(result)
    

    return results