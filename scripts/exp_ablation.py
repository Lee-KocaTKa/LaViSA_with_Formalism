from __future__ import annotations

import argparse

from tqdm import tqdm
from collections import defaultdict

import json 

from src.data.writer import append_json
from src.experiments.ablation.run import run_amr_generation
from src.models.qwen import QwenModel
from src.paths import ABLATION_OUTPUT_DIR, AUGMENTED_JSONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "card",
        help="Hugging Face model card, e.g. Qwen/Qwen3.5-27B",
    )

    args = parser.parse_args()
    model_card = args.card

    model = QwenModel(
        model_card=model_card,
        max_output_tokens=1024,
    )

    model_name = model_card.split("/")[-1]

    output_dir = ABLATION_OUTPUT_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{model_name}.json"

    with open(AUGMENTED_JSONS, "r", encoding="utf-8") as f:
        lavisa_samples = json.load(f) 

    print(f"LaViSA {len(lavisa_samples)} samples")
    print(f"Task: Ablation")
    print(f"Model: {model_card}")
    print(f"Output: {output_path}")
    
    lavisa_grouped = defaultdict(list)
    
    for sample in lavisa_samples:
        lavisa_grouped[sample["group_id"]].append(sample)
    
    lavisa_group_keys = list(lavisa_grouped.keys()) 

    for group_key in tqdm(
        lavisa_group_keys,
        desc="AMR ablation",
    ):
        samples = lavisa_grouped[group_key]
        
        results = run_amr_generation(
            samples=samples,
            model=model,
        )

        for result in results: 

            append_json(
                data=result,
                output_path=output_path,
            )

    print("Experiment Complete")


if __name__ == "__main__":
    main()