from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Resolve : pathabsolutisation 


DATA_DIR = PROJECT_ROOT / "data" 
LAVISA_DIR = DATA_DIR / "lavisa" 
ORIG_DIR = LAVISA_DIR / "original" 
AUGMENTED_DIR = LAVISA_DIR / "augmented"
REVISED_DIR = LAVISA_DIR / "revised"


OUTPUTS_DIR = PROJECT_ROOT / "outputs" 
CONFIGS_DIR = PROJECT_ROOT / "configs" 

