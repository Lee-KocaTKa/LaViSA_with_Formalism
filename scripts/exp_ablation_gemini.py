from __future__ import annotations

import argparse

from tqdm import tqdm
from collections import defaultdict

import json 
import os 

from src.data.writer import append_json
from src.experiments.ablation.run import run_amr_generation
from src.models.gemini import GeminiModel
from src.paths import ABLATION_OUTPUT_DIR, AUGMENTED_JSONS


def main() -> None: 
    parser = argparse.ArgumentParser() 
    parser.add_argument("card")
    
    args = parser.parse_args() 
    model_card = args.card 
    
    model = GeminiModel(
        model_card=model_card,
        api_key=os.environ["GEMINI_API_KEY"], 
        max_output_tokens=1024 
    )
    
    output_dir = ABLATION_OUTPUT_DIR
    output_path = output_dir / f"{model_card}.json" 
    
    with open(AUGMENTED_JSONS, "r", encoding="utf-8") as f:
        lavisa_samples = json.load(f) 
    
    if output_path.is_file(): 
        with open(output_path, "r", encoding="utf-8") as f: 
            anchor = json.load(f) 
    
        anchor_point = len(anchor) 
        lavisa_samples = lavisa_samples[anchor_point:] 
    
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