import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.vqa_rad_dataset import VQARadVisualBERTDataset
import matplotlib.pyplot as plt

# -------------------------------
# Visual Feature Extractor
# -------------------------------
class VisualFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            feats = self.backbone(images)
        return feats.squeeze(-1).squeeze(-1)  # [B, 2048]


# -------------------------------
# Early Stopping
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, checkpoint_path="outputs/checkpoints/visualbert_best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"✓ Validation improved → checkpoint saved")
        else:
            self.counter += 1
            print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def vqa_collate_fn(batch):
    """
    Custom collate function to handle text fields (question, answer)
    without converting them to tensors.
    """
    images = torch.stack([b["image"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    questions = [b["question"] for b in batch]   # keep as list[str]
    answers = [b["answer"] for b in batch]       # keep as list[str]

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question": questions,
        "answer": answers,
    }

# -------------------------------
# Training
# -------------------------------
def train_for_one_epoch(model, extractor, loader, optimizer, criterion, answer_to_idx, device):
    model.train()
    total_loss, total, errors = 0, 0, 0
    seq_len = 10

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = torch.tensor(
            [answer_to_idx.get(str(a).lower().strip(), answer_to_idx["UNK"]) for a in batch["answer"]],
            dtype=torch.long, device=device
        )

        visual_feats = extractor(images)
        visual_embeds = visual_feats.unsqueeze(1).expand(-1, seq_len, -1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=torch.ones(visual_embeds.shape[:-1], device=device),
            visual_token_type_ids=torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
        )

        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.logits.argmax(1)
        total_loss += loss.item()
        errors += (preds != labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * (1 - errors / total)


# -------------------------------
# Evaluation (excluding UNK)
# -------------------------------
@torch.no_grad()
def evaluate_excluding_unk(model, extractor, loader, criterion, answer_to_idx, device):
    model.eval()
    total_loss, total, errors, batches = 0, 0, 0, 0
    seq_len = 10
    unk = answer_to_idx["UNK"]

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = torch.tensor(
            [answer_to_idx.get(str(a).lower().strip(), unk) for a in batch["answer"]],
            dtype=torch.long, device=device
        )

        mask = labels != unk
        if mask.sum() == 0:
            continue

        visual_feats = extractor(images)
        visual_embeds = visual_feats.unsqueeze(1).expand(-1, seq_len, -1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=torch.ones(visual_embeds.shape[:-1], device=device),
            visual_token_type_ids=torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
        )

        loss = criterion(outputs.logits[mask], labels[mask])

        preds = outputs.logits.argmax(1)
        total_loss += loss.item()
        errors += (preds[mask] != labels[mask]).sum().item()
        total += mask.sum().item()
        batches += 1

    return total_loss / max(1, batches), 100 * (1 - errors / total)


# -------------------------------
# Failure-only qualitative eval
# -------------------------------
@torch.no_grad()
def evaluate_open_ended(model, extractor, loader, idx_to_answer, device, num_examples=10):
    model.eval()
    seq_len = 10
    collected = 0

    print("\n===== FAILURE-ONLY QUALITATIVE =====\n")

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        visual_feats = extractor(images)
        visual_embeds = visual_feats.unsqueeze(1).expand(-1, seq_len, -1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=torch.ones(visual_embeds.shape[:-1], device=device),
            visual_token_type_ids=torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
        )

        preds = outputs.logits.argmax(1).cpu().tolist()

        for i, p in enumerate(preds):
            gt = str(batch["answer"][i]).lower().strip()
            pred = idx_to_answer.get(p, "UNK")

            if pred == gt:
                continue

            print(f"Q: {batch['question'][i]}")
            print(f"GT: {gt}")
            print(f"PRED: {pred}")
            print("-" * 50)

            collected += 1
            if collected >= num_examples:
                return


# -------------------------------
# MAIN
# -------------------------------
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

    train_data = VQARadVisualBERTDataset("data/VQA_RAD/train.json", "data/VQA_RAD/images", tokenizer, transform)
    val_data   = VQARadVisualBERTDataset("data/VQA_RAD/val.json",   "data/VQA_RAD/images", tokenizer, transform)
    test_data  = VQARadVisualBERTDataset("data/VQA_RAD/test.json",  "data/VQA_RAD/images", tokenizer, transform)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=vqa_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=vqa_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=vqa_collate_fn)

    answers = sorted({str(s["answer"]).lower().strip() for s in train_data.samples})
    answer_to_idx = {a: i for i, a in enumerate(answers)}
    answer_to_idx["UNK"] = len(answer_to_idx)
    idx_to_answer = {i: a for a, i in answer_to_idx.items()}

    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        num_labels=len(answer_to_idx)
    ).to(device)

    extractor = VisualFeatureExtract().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=5)

    for epoch in range(10):
        tr_loss, tr_acc = train_for_one_epoch(model, extractor, train_loader, optimizer, criterion, answer_to_idx, device)
        va_loss, va_acc = evaluate_excluding_unk(model, extractor, val_loader, criterion, answer_to_idx, device)

        print(f"Epoch {epoch+1}: Train {tr_acc:.2f}% | Val {va_acc:.2f}%")

        stopper(va_loss, model)
        if stopper.early_stop:
            break

    model.load_state_dict(torch.load("outputs/checkpoints/visualbert_best.pt", map_location=device))

    test_loss, test_acc = evaluate_excluding_unk(model, extractor, test_loader, criterion, answer_to_idx, device)
    print(f"TEST ACC: {test_acc:.2f}%")

    evaluate_open_ended(model, extractor, test_loader, idx_to_answer, device)


if __name__ == "__main__":
    main()
