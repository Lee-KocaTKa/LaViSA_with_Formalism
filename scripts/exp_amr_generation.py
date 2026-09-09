from __future__ import annotations

import argparse

from tqdm import tqdm

import json 

from src.data.writer import append_json
from src.experiments.amr_generation.run import run_amr_generation
from src.models.qwen import QwenModel
from src.paths import AMR_GENERATION_OUTPUT_DIR, AUGMENTED_JSONS


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
        max_output_tokens=512,
    )

    model_name = model_card.split("/")[-1]

    output_dir = AMR_GENERATION_OUTPUT_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{model_name}.json"

    with open(AUGMENTED_JSONS, "r", encoding="utf-8") as f:
        lavisa_samples = json.load(f) 

    print(f"LaViSA {len(lavisa_samples)} samples")
    print(f"Task: AMR Generation")
    print(f"Model: {model_card}")
    print(f"Output: {output_path}")

    for sample in tqdm(
        lavisa_samples,
        desc="AMR generation",
    ):
        result = run_amr_generation(
            sample=sample,
            model=model,
        )

        append_json(
            data=result,
            output_path=output_path,
        )

    print("Experiment Complete")


if __name__ == "__main__":
    main()