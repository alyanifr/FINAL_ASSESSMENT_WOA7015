import sys
import os

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.vqa_rad_dataset import VQARadVisualBERTDataset
from VisualBertModel.VisualBert_vqa import (
    VisualFeatureExtract,
    vqa_collate_fn,
    evaluate_excluding_unk,
    evaluate_open_ended
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    test_data  = VQARadVisualBERTDataset("data/VQA_RAD/test.json",  "data/VQA_RAD/images", tokenizer, transform)
    test_loader  = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=vqa_collate_fn)

    # --- ANSWER VOCAB ---
    answers = sorted({sample["answer"] for sample in test_data.samples})

    if "UNK" not in answers:
        answers.append("UNK")

    answer_to_idx = {ans: i for i, ans in enumerate(answers)}
    # answer_to_idx["UNK"] = len(answer_to_idx)

    idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}

    state_dict = torch.load("outputs/checkpoints/visualbert_best.pt", map_location=device)

    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        num_labels=len(answer_to_idx)
    ).to(device)


    model.load_state_dict(state_dict, strict=False)

    extractor = VisualFeatureExtract().to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_loss, test_acc = evaluate_excluding_unk(model, extractor, test_loader, criterion, answer_to_idx, device)
    print(f"TEST_LOSS: {test_loss:.4f} | TEST ACC: {test_acc:.2f}%")

    evaluate_open_ended(model, extractor, test_loader, idx_to_answer, device)


if __name__ == "__main__":
    main()
