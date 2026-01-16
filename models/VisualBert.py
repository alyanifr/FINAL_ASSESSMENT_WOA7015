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
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)            # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)     # [B, 2048]

        return features
    
# TRAINING PIPELINE
def train_for_one_epoch(model, visual_features_extractor, loader, optimizer, answer_to_idx, device):

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
            [answer_to_idx[ans] for ans in batch["answer"]],
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
        loss = nn.CrossEntropyLoss()(logits, labels)

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
def evaluate(model, visual_features_extractor, loader, answer_to_idx, device):
    
    model.eval()

    total_loss = 0
    total_eval = 0
    eval_error = 0

    visual_sequence_length = 10

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = torch.tensor(
            [answer_to_idx[ans] for ans in batch["answer"]],
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
            visual_token_type_ids=visual_token_type_ids,
        )

        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        eval_error += (preds != labels).sum().item()
        total_eval += labels.size(0)

    return  total_eval / len(loader), 100 * (1 - eval_error / total_eval)

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
    # idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}

    # --- MODELS ---
    model = VisualBertForQuestionAnswering.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        num_labels=len(answer_to_idx)
    )

    model.to(device)

    visual_extractor = VisualFeatureExtract().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # --- TRAINING ---
    epochs = 3

    for epoch in range(epochs):
        train_loss, train_accuracy = train_for_one_epoch(
            model=model,
            visual_features_extractor=visual_extractor,
            loader=train_loader,
            optimizer=optimizer,
            answer_to_idx=answer_to_idx,
            device=device
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            visual_features_extractor=visual_extractor,
            loader=validation_loader,
            answer_to_idx=answer_to_idx,
            device=device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )
    
    # --- TESTING ---
    test_loss, test_accuracy = evaluate(
        model=model,
        visual_features_extractor=visual_extractor,
        loader=test_loader,
        answer_to_idx=answer_to_idx,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")




if __name__ == "__main__":
    main()


    

"""
Example: 

response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
image = Image.open(BytesIO(response.content))


visual_embeds = get_visual_embeddings_simple(image, visual_feature_extractor, device)
    
inputs = tokenizer("What is shown in this image?", return_tensors="pt")
    
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
})
    
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer_idx = logits.argmax(-1).item()

print(f"Predicted answer: {predicted_answer_idx}")
"""