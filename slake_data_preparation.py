import os
import json
from datasets import load_dataset

# CONFIG
DATA_ROOT = "SLAKE"
IMG_DIR = os.path.join(DATA_ROOT, "imgs")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")

os.makedirs(ANNOTATION_DIR, exist_ok=True)

# LOAD SLAKE FOR ANNOTATIONS
slake = load_dataset("BoKelvin/SLAKE")
print("SLAKE Dataset loaded from Hugging Face.")

 