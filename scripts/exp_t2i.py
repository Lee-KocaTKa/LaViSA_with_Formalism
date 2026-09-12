from __future__ import annotations

import argparse

from tqdm import tqdm
from collections import defaultdict

import json 

from src.data.writer import append_json
from src.experiments.t2i.run import run_t2i
from src.models.qwen import QwenModel
from src.paths import T2I_OUTPUT_DIR, AUGMENTED_JSONS


def group_by_group_id(data): 
    groups = defaultdict(list) 
    
    for sample in data: 
        groups[sample["group_id"]].append(sample)
    
    return groups 


def make_matching_samples(group):
    matching_samples = []
    
    image_paths = [sample["image_path"] for sample in group]
    
    for amr_sample in group:
        sample = {
            "sample_id": amr_sample["sample_id"],
            "group_id": amr_sample["group_id"],
            "category": amr_sample["category"],
            "formalism": amr_sample["formalism"],
            "image_path": image_paths,
            "correct_answer": amr_sample["sample_id"][-1],
        }

        matching_samples.append(sample)

    return matching_samples


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

    output_dir = T2I_OUTPUT_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{model_name}.json"

    with open(AUGMENTED_JSONS, "r", encoding="utf-8") as f:
        lavisa_samples = json.load(f) 

    lavisa_grouped = group_by_group_id(lavisa_samples)
    grouped_ids = lavisa_grouped.keys()

    print(f"LaViSA {len(grouped_ids)} samples")
    print(f"Task: T2I")
    print(f"Model: {model_card}")
    print(f"Output: {output_path}")

    for id in tqdm(
        grouped_ids,
        desc="T2I",
    ):
        group = lavisa_grouped[id]  
        group_samples = make_matching_samples(group)
        
        for sample in group_samples:
            result = run_t2i(
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