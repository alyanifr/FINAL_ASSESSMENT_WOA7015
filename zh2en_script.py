import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)

def translate_batch(texts, batch_size=8):
    outputs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=64)
        outputs.extend(tokenizer.batch_decode(translated, skip_special_token=True))
    
    return outputs

with open("SLAKE/annotations/train_zh.json", "r", encoding="utf-8") as f:
    zh_samples = json.load(f)

questions_zh = [sample["question"] for sample in zh_samples]
answers_zh = [sample["answer"] for sample in zh_samples]

questions_zh2en = translate_batch(questions_zh)
answers_zh2en = translate_batch(answers_zh)

# SAVING ZH2EN DATA

translated_samples = []

for sample, q_zh2en, a_zh2en in zip(zh_samples, questions_zh2en, answers_zh2en):
    translated_samples.append({
        "img_name": sample["img_name"],
        "question": q_zh2en,
        "answer": a_zh2en,
        "source": "zh2en-translated"
    })

with open("train_zh2en.json", "w", encoding="utf-8") as f:
    json.dump(translated_samples, f, indent=2)

print("Translation complete.")