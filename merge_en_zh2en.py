import os
import json

# CONFIG
DATA_ROOT = "SLAKE/annotations"
EN_PATH = os.path.join(DATA_ROOT, "train_en.json")
ZH2EN_PATH = os.path.join(DATA_ROOT, "train_zh2en.json")
OUT_PATH = os.path.join(DATA_ROOT, "train_merged.json")

# LOAD DATA
with open(EN_PATH, "r") as f:
    en_file = json.load(f)

with open(ZH2EN_PATH, "r") as f:
    zh2en_file = json.load(f)

print("Loading EN and ZH2EN files complete.")
print(f"EN Train: {len(en_file)} samples.")
print(f"ZH2EN Train: {len(zh2en_file)} samples.")

# ANSWER NORMALIZATION
def normalize_answer(ans):
    ans = ans.lower().strip()

    if ans in ["yes", "y", "true"]:
        return "yes"
    
    if ans in ["no", "n", "False"]:
        return "no"
    
    if ans in ["none", "na", "n/a"]:
        return "no"
    
    return ans 

for sample in en_file:
    sample["answer"] = normalize_answer(sample["answer"])
    sample["source"] = "en"

for sample in zh2en_file:
    sample["answer"] = normalize_answer(sample["answer"])
    sample["source"] = "zh2en-translated"

# CHECK AND REMOVE DUPLICATES
def check_duplicates(samples):
    seen = set()
    unique_samples = []

    for sample in samples:
        data = (
            sample["img_name"], sample["question"], sample["queation"].lower().strip()
        )

        if data not in seen:
            seen.add(data)
            unique_samples.append(sample)
    
    return unique_samples

# MERGING
merged = en_file + zh2en_file
print(f"Merged successful: {len(merged)} samples.")

unique_merged = check_duplicates(merged)
print(f"Unique Merged: {len(unique_merged)} samples.")

# SAVING DATA
with open(OUT_PATH, "w") as f:
    json.dump(unique_merged, f, indent=2)

print("\n===== MERGE SUMMARY =====")
print(f"Final training samples: {len(unique_merged)}")
print(f"File path: {OUT_PATH}")
print("==========================")