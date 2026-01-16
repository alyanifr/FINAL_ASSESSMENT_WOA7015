import sys
import os

sys.path.append(os.path.abspath("."))

import json
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.slake_datasets import VisualBERTSlakeDataset
from PIL import Image
import matplotlib.pyplot as plt

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
    
# EARLY STOPPING PIPELINE + CHECKPOINT
class EarlyStopping:
    def __init__(self,
                patience=3,
                min_delta=0.0,
                checkpoint_path="outputs/checkpoints/best_model.pt",
                verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    def __call__(self, validation_loss, model):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early Stopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        if self.verbose:
            print(f"Validation loss improved. Model saved to {self.checkpoint_path}")

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

        labels = torch.tensor([answer_to_idx.get(ans, answer_to_idx["UNK"]) 
                               for ans in batch["answer"]], 
                               dtype=torch.long, 
                               device=device)

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
        labels = torch.tensor([answer_to_idx.get(ans, unk_idx) 
                               for ans in batch["answer"]], 
                               dtype=torch.long, 
                               device=device)

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

# TRAINING CURVES
def plot_training_curves(train_losses, validation_losses, train_accuracies, validation_accuracies, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses)+1)

    # LOSS PLOT
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, validation_accuracies, label="Validation Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

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

# FAILURE QUALITATIVE ANALYSIS
@torch.no_grad()
def evaluate_failures_with_images(model, visual_features_extractor, loader, idx_to_answer, tokenizer, device, max_failures=10, save_dirs="outputs/failures"):
    os.makedirs(save_dirs, exist_ok=True)

    model.eval()

    visual_sequence_length = 10
    failures = 0

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

        for i, pred in enumerate(preds):
            ground_truth = batch["answer"][i]
            pred_ans = idx_to_answer[pred]

            if pred_ans != ground_truth:
                question = batch["question"][i] if "quqestion" in batch else tokenizer.decode(input_ids[i], skip_special_tokens=True)
                img = images[i].cpu().permute(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())

                plt.figure(figsize=(4,4))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Question: {question}\nGround Truth: {ground_truth}\nPred: {pred_ans}", fontsize=8)
                filename = f"failure_{failures+1}.png"
                plt.savefig(os.path.join(save_dirs, filename), dpi=300)
                plt.close

                print(f"[FAILURE {failures+1}] Question: {question} | Ground Truth: {ground_truth} | Pred: {pred_ans}")
                failures += 1
                if failures >= max_failures:
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
    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
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

    # --- LOAD BEST MODEL ---
    model.load_state_dict(torch.load("outputs/checkpoints/visualbert_best.pt", map_location=device))

    # --- PLOT CURVES ---
    plot_training_curves(train_losses, validation_losses, train_accs, validation_accs)

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
    evaluate_failures_with_images(model, visual_extractor, test_loader, idx_to_answer, tokenizer, device, max_failures=10)

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


    

