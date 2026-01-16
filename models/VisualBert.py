import sys
import os

sys.path.append(os.path.abspath("."))

import os
import json
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.slake_datasets import VisualBERTSlakeDataset
from PIL import Image

class VisualFeatureExtract(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)            # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)     # [B, 2048]

        return features
    
# TRAINING PIPELINE
def train_for_one_epoch(model, visual_features_extractor, loader, optimizer, criterion, answer_to_idx, device):

    model.train()

    total_loss = 0
    total_training = 0
    training_error = 0

    visual_sequence_length = 10

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = torch.tensor(
            [answer_to_idx.get(ans, answer_to_idx["UNK"]) for ans in batch["answer"]],
            dtype=torch.long,
            device=device
        )

        # VISUAL EMBEDDINGS
        visual_features = visual_features_extractor(images)
        visual_embeds = visual_features.unsqueeze(1).expand(
            visual_features.size(0), visual_sequence_length, visual_features.size(1)
        )

        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long, device=device
        )

        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float, device=device
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        training_error += (preds != labels).sum().item()
        total_training += labels.size(0)

    return  total_loss / len(loader), 100 * (1 - training_error / total_training)


# EVALUATION PIPELINE
@torch.no_grad()
def evaluate_excluding_unk(model, visual_features_extractor, loader, criterion, answer_to_idx, device):
    
    model.eval()

    total_loss = 0
    total_eval = 0
    eval_error = 0

    visual_sequence_length = 10
    unk_idx = answer_to_idx["UNK"]

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Map unseen answers to "UNK"
        labels = [
            answer_to_idx.get(ans, unk_idx) for ans in batch["answer"]
        ]
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Skip samples with UNK labels
        mask = labels != unk_idx
        if mask.sum() == 0:
            continue

        # VISUAL EMBEDDINGS
        visual_features = visual_features_extractor(images)
        visual_embeds = visual_features.unsqueeze(1).expand(
            visual_features.size(0), visual_sequence_length, visual_features.size(1)
        )

        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long, device=device
        )

        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float, device=device
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )

        logits = outputs.logits
        loss = criterion(logits[mask], labels[mask])

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        eval_error += (preds[mask] != labels[mask]).sum().item()
        total_eval += mask.sum().item()

    return  total_eval / len(loader), 100 * (1 - eval_error / total_eval)

@torch.no_grad()
def evaluate_open_ended(model, visual_features_extractor, loader, idx_to_answer, device, num_examples=10):
    model.eval()
    visual_sequence_length = 10

    # examples = []

    print("\n===== GROUND TRUTH EVALUATION =====\n")

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # VISUAL EMBEDDINGS
        visual_features = visual_features_extractor(images)
        visual_embeds = visual_features.unsqueeze(1).expand(
            visual_features.size(0), visual_sequence_length, visual_features.size(1)
        )
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float, device=device
        )
        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long, device=device
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids
        )

        preds = outputs.logits.argmax(dim=1).cpu().tolist()

        for i in range(len(preds)):
            print(f"Question: {batch['question'][i]}")
            print(f"Ground Truth: {batch['answer'][i]}")
            print(f"Prediction: {idx_to_answer.get(preds[i], 'UNK')}")
            print("-" * 50)

            collected += 1
            if collected >= num_examples:
                return


# MAIN
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
        batch_size=16,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=16,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=True,
    )

    # --- ANSWER VOCAB ---
    answers = sorted({sample["answer"] for sample in train_data.samples})
    answer_to_idx = {ans: i for i, ans in enumerate(answers)}
    answer_to_idx["UNK"] = len(answer_to_idx)

    idx_to_answer = {v: k for k, v in answer_to_idx.items()}

    # --- MODELS ---
    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        num_labels=len(answer_to_idx)
    )

    model.to(device)

    visual_extractor = VisualFeatureExtract().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    # --- TRAINING ---
    epochs = 3

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

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy (Excluding UNK): {validation_accuracy:.2f}%"
        )
    
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


    

