# Generating Ambiguous Captions for Real Images (Assuming JCRE3)

from __future__ import annotations

import json 
from pathlib import Path
from typing import Any, Dict, List

from src.data_augmentation.prompts import (
    build_realimage_based_caption_augmentation_prompts 
)


def parse_json_response(response: str) -> Dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON: {e}")
        
        
def augment_real_image_with_captions(
    image_path: str | Path, 
    model: Any, 
) -> List[Dict[str, Any]]: 
    prompts = build_realimage_based_caption_augmentation_prompts() 
    
    results = [] 
    
    for category, prompt in prompts.items(): 
        response = model.generate(
            prompt=prompt, 
            image_path=image_path,
        )
        
        try: 
            parsed = parse_json_response(response)
            
            result = {
                "category": category, 
                "success": True, 
                "result": parsed, 
                "raw_response": response, 
            }
        
        except ValueError as e: 
            result = {
                "category": category, 
                "success": False, 
                "error": str(e), 
                "raw_response": response, 
                "result": None, 
            }
            
        results.append(result) 
        
    return results
