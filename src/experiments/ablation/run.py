from __future__ import annotations

from typing import Any


from src.experiments.prompts import build_ablation_generation_prompt


def run_amr_generation(
    samples: list[dict[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:

    results = []
    generated_amr = None
    generation_error = None

    for i, sample in enumerate(samples):
        if i == 0:
            try:
                prompt = build_ablation_generation_prompt(sample)
                generated_amr = model.generate(
                    prompt=prompt,
                    image_paths=None,
                )
            except Exception as e:
                generation_error = str(e)
                print(f"Generation failed: {generation_error}")

        result = {
            "interpretation_id": sample["interpretation_id"],
            "ambiguous_caption": sample["ambiguous_caption"],
            "interpretation": sample["interpretation"],
            "image_path": sample["image_path"],
            "gold_amr": sample["formalism"],
            "generated_amr": generated_amr,
            "error": generation_error,
        }
        results.append(result)

    return results