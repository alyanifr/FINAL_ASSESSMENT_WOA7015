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
    EarlyStopping,
    train_for_one_epoch,
    plot_training_curves,
    evaluate_excluding_unk,
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
    train_data = VisualBERTSlakeDataset(
        annotations_path="data/SLAKE/annotations/train_en.json",
        image_root="data/SLAKE/imgs",
        tokenizer=tokenizer,
        transform=transform
    )

    validation_data = VisualBERTSlakeDataset(
        annotations_path="data/SLAKE/annotations/validation_en.json",
        image_root="data/SLAKE/imgs",
        tokenizer=tokenizer,
        transform=transform
    )

    test_data = VisualBERTSlakeDataset(
        annotations_path="data/SLAKE/annotations/test_en.json",
        image_root="data/SLAKE/imgs",
        tokenizer=tokenizer,
        transform=transform
    )

    # --- DATALOADERS ---
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=8,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
    )

    # --- ANSWER VOCAB ---
    answers = sorted({sample["answer"] for sample in train_data.samples})

    if "UNK" not in answers:
        answers.append("UNK")

    answer_to_idx = {ans: i for i, ans in enumerate(answers)}
    # answer_to_idx["UNK"] = len(answer_to_idx)

    idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}

    # --- MODELS ---
    model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre", 
                                                           num_labels=len(answer_to_idx)).to(device)

    visual_extractor = VisualFeatureExtract().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    criterion = nn.CrossEntropyLoss()

    # --- EARLY STOPPING ---
    early_stopping = EarlyStopping(patience=3, checkpoint_path="outputs/checkpoints/visualbert_best.pt")

    # --- TRAINING ---
    epochs = 10
    train_losses, validation_losses, train_accs, validation_accs = [], [], [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_for_one_epoch(
            model=model,
            visual_features_extractor=visual_extractor,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            answer_to_idx=answer_to_idx,
            device=device
        )

        validation_loss, validation_accuracy = evaluate_excluding_unk(
            model=model,
            visual_features_extractor=visual_extractor,
            loader=validation_loader,
            criterion=criterion,
            answer_to_idx=answer_to_idx,
            device=device
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_accs.append(train_accuracy)
        validation_accs.append(validation_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy (Excluding UNK): {validation_accuracy:.2f}%"
        )

        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # --- PLOT CURVES ---
    plot_training_curves(train_losses, validation_losses, train_accs, validation_accs)

    pass

if __name__ == "__main__":
    main()