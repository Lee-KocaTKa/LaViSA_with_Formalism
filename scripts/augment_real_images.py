from __future__ import annotations 

import os 

from src.data.writer import append_jsonl
from src.data_augmentation.real_image import augment_real_image_with_captions 
from src.models.gemini import GeminiModel 
from src.paths import JCRE3_IMAGE_DIR, JCRE3_JSON_DIR 


def main() -> None: 
    model = GeminiModel(
        model_card="",
        api_key=os.environ["GEMINI_API_KEY"],
        max_output_tokens=256 
    ) 
    
    output_path = (
        JCRE3_JSON_DIR / "augmented_captions.jsonl" 
    )
    
    image_paths = sorted(JCRE3_IMAGE_DIR.iterdir()) 
    
    for image_path in image_paths: 
        if not image_path.is_file():
            continue
        
        print(f"Processing image: {image_path.name}") 
        
        generations = augment_real_image_with_captions(
            image_path=image_path,
            model=model 
        )
        
        record = {
            "image_id": image_path.stem,
            "image_path": str(image_path), 
            "model": model.model_card, 
            "generations": generations
        }
        
        append_jsonl(record, output_path) 


if __name__ == "__main__": 
    main() 
    