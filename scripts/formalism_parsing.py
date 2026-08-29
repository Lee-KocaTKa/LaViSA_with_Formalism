from __future__ import annotations 
from tqdm import tqdm 

import os 
import argparse 
import time 

from src.data.writer import append_json
from src.data.loader import build_all_lavisa_samples 
from src.data_augmentation.formalism import generate_formalisms 
from src.models.gemini import GeminiModel
from src.paths import ORIG_JSONS, AUGMENTED_JSONS
 

def main() -> None: 
    parser = argparse.ArgumentParser() 
    parser.add_argument("card")
    
    args = parser.parse_args() 
    model_card = args.card 
    
    model = GeminiModel(
        model_card=model_card,
        api_key=os.environ["GEMINI_API_KEY"],
        max_output_tokens=512 
    )
    
    output_path = AUGMENTED_JSONS / "augmented.json" 
    #jsons_paths = sorted(ORIG_JSONS.iterdir()) 
    
    #for json_path in jsons_paths:
    #    if not json_path.is_file(): 
    #        print(f"file {json_path} inexistent")
    #        continue 
        
    #    print(f"Processing Category: {json_path.name}")
    
    lavisa_samples = build_all_lavisa_samples() 
    lavisa_sample_cnt = len(lavisa_samples)
    print(f"LaViSA {lavisa_sample_cnt} samples") # 1503 
    
    for i in tqdm(range(lavisa_sample_cnt)):
        sample = lavisa_samples[i] 
        formalism_augmented_sample = generate_formalisms(sample=sample, model=model)
        append_json(data=formalism_augmented_sample, output_path=output_path)
        
    print("Add-on Complete")
    
         
    
        
        
        
    
    return 


if __name__ == "__main__": 
    main() 