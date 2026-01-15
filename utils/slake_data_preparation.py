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

def extract_annotations(splits):
    en_data = []
    zh_data = []

    for split in slake[splits]:
        entry = {
            "img_name": split["img_name"],
            "question": split["question"],
            "answer": split["answer"]
        }

        if split["q_lang"] == "en":
            en_data.append(entry)
        elif split["q_lang"] == "zh":
            zh_data.append(entry)
    
    return en_data, zh_data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saving {len(data)} samples to {path} complete.")

# PROCESSING SPLITS
train_en, train_zh = extract_annotations("train")
print("Processing training split complete.")

validation_en, _ = extract_annotations("validation")
print("Processing validation split complete.")

test_en, _ = extract_annotations("test")
print("Processing test split complete.")

# SAVING ANNOTATIONS
save_json(train_en, os.path.join(ANNOTATION_DIR, "train_en.json"))
save_json(train_zh, os.path.join(ANNOTATION_DIR, "train_zh.json"))
save_json(validation_en, os.path.join(ANNOTATION_DIR, "validation_en.json"))
save_json(test_en, os.path.join(ANNOTATION_DIR, "test_en.json"))

print("Saving annotations file complete.")

# /////////////// #
print("\n===== SLAKE DATA PREPARATION SUMMARY =====")
print(f"Train EN: {len(train_en)}")
print(f"Train ZH: {len(train_zh)}")
print(f"Validation EN: {len(validation_en)}")
print(f"Test EN: {len(test_en)}")
print("==========================================")
