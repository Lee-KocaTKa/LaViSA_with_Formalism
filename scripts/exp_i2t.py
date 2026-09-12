from __future__ import annotations

import argparse

from tqdm import tqdm
from collections import defaultdict

import json 

from src.data.writer import append_json
from src.experiments.i2t.run import run_i2t
from src.models.qwen import QwenModel
from src.paths import I2T_OUTPUT_DIR, AUGMENTED_JSONS


def group_by_group_id(data): 
    groups = defaultdict(list) 
    
    for sample in data: 
        groups[sample["group_id"]].append(sample)
    
    return groups 


def make_matching_samples(group):
    matching_samples = []
    
    amr1 = None 
    amr2 = None 
    amr3 = None 
    
    for sample in group:
        id = sample["sample_id"][-1]
        if id == "1":
            amr1 = sample["formalism"]
        elif id == "2":
            amr2 = sample["formalism"]
        elif id == "3": 
            amr3 = sample["formalism"]
        else: 
            pass 

    for image_sample in group:
        sample = {
            "sample_id": image_sample["sample_id"],
            "group_id": image_sample["group_id"],
            "category": image_sample["category"],
            "amr_1": amr1,
            "amr_2": amr2, 
            "amr_3": amr3, 
            "image_path": image_sample["image_path"],
            "correct_answer": image_sample["sample_id"][-1],
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

    output_dir = I2T_OUTPUT_DIR
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
    print(f"Task: I2T")
    print(f"Model: {model_card}")
    print(f"Output: {output_path}")

    for id in tqdm(
        grouped_ids,
        desc="Yes or No",
    ):
        group = lavisa_grouped[id]  
        group_samples = make_matching_samples(group)
        
        for sample in group_samples:
            result = run_i2t(
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