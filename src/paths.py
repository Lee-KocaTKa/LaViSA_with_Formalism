from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Resolve : pathabsolutisation 


DATA_DIR = PROJECT_ROOT / "data" 
LAVISA_DIR = DATA_DIR / "lavisa" 
JCRE3_DIR = DATA_DIR / "jcre3"

JCRE3_IMAGE_DIR = JCRE3_DIR / "images" 
JCRE3_JSON_DIR = JCRE3_DIR / "jsons" 

ORIG_DIR = LAVISA_DIR / "original" 
ORIG_JSONS = ORIG_DIR / "jsons" 
ORIG_IMAGES = ORIG_DIR / "images"

AUGMENTED_DIR = LAVISA_DIR / "augmented"
AUGMENTED_JSONS = AUGMENTED_DIR / "jsons" / "augmented_annotated.json"
AUGMENTED_IMAGES = AUGMENTED_DIR / "images"

REVISED_DIR = LAVISA_DIR / "revised"
REVISED_JSONS = REVISED_DIR / "jsons"
REVISED_IMAGES = REVISED_DIR / "images"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" 
AMR_GENERATION_OUTPUT_DIR = OUTPUTS_DIR / "amr_generation"
YES_OR_NO_OUTPUT_DIR = OUTPUTS_DIR / "yes_or_no"
I2T_OUTPUT_DIR = OUTPUTS_DIR / "i2t"
T2I_OUTPUT_DIR = OUTPUTS_DIR / "t2i"

CONFIGS_DIR = PROJECT_ROOT / "configs" 

