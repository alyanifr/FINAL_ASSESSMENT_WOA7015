import sys
import os

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.slake_datasets import VisualBERTSlakeDataset
from VisualBertModel.VisualBert import (
    VisualFeatureExtract,
    evaluate_excluding_unk,
    evaluate_failures_with_images,
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
            std=[0.229, 0.224, 0.225]
        )
    ])

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # --- DATASET ---
    test_data = VisualBERTSlakeDataset(
        annotations_path="data/SLAKE/annotations/test_en.json",
        image_root="data/SLAKE/imgs",
        tokenizer=tokenizer,
        transform=transform
    )

    # --- DATALOADERS ---
    test_loader = DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
    )

    # --- ANSWER VOCAB ---
    answers = sorted({sample["answer"] for sample in test_data.samples})

    if "UNK" not in answers:
        answers.append("UNK")

    answer_to_idx = {ans: i for i, ans in enumerate(answers)}
    # answer_to_idx["UNK"] = len(answer_to_idx)

    idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}

    # --- MODELS ---
    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        num_labels=len(answer_to_idx)).to(device)
    
    # --- LOAD BEST MODEL ---
    model.load_state_dict(torch.load("outputs/checkpoints/visualbert_best.pt", map_location=device))

    model.to(device)
    model.eval()

    visual_extractor = VisualFeatureExtract().to(device)
    criterion = nn.CrossEntropyLoss()

    # --- TESTING ---
    test_loss, test_accuracy = evaluate_excluding_unk(
        model=model,
        visual_features_extractor=visual_extractor,
        loader=test_loader,
        criterion=criterion,
        answer_to_idx=answer_to_idx,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    # --- Failure-only qualitative analysis ---
    evaluate_failures_with_images(
        model=model,
        visual_features_extractor=visual_extractor,
        loader=test_loader,
        idx_to_answer=idx_to_answer,
        tokenizer=tokenizer,
        device=device,
        max_failures=5
    )

    # --- OPEN-ENDED EVAL ---
    evaluate_open_ended(
        model=model,
        visual_features_extractor=visual_extractor,
        loader=test_loader,
        idx_to_answer=idx_to_answer,
        device=device,
        num_examples=5
    )


if __name__ == "__main__":
    main()