import os
import json
import requests
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
from utils.slake_datasets import VisualBERTSlakeDataset
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            features = self.backbone(images)
        features = features.squeeze(-1).squeeze(-1)

        return features
    
def get_visual_embeddings_simple(image, extractor, visual_seq_length=10, device=None):

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Image must be a PIL Image or path to image file")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = extractor(image_tensor)
    
    batch_size = features.shape[0]
    feature_dim = features.shape[1]
    visual_seq_length = 10
    
    visual_embeds = features.squeeze(-1).squeeze(-1).unsqueeze(1).expand(batch_size, visual_seq_length, feature_dim)
    
    return visual_embeds

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

train_data = VisualBERTSlakeDataset(
    annotations_path="data/SLAKE/annotations/train_en.json",
    image_root="data/SLAKE/imgs",
    tokenizer=tokenizer,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

### Answer vocab
answers = sorted({sample["answer"] for sample in train_data.samples})
answer_to_idx = {ans: i for i, ans in enumerate(answers)}
idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}

### Training pipeline
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre", num_labels=len(answer_to_idx))
model.to(device)

visual_feature_extractor = VisualFeatureExtract().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

### Variables for plotting
epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0
    total_training = 0
    training_error = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = torch.tensor(
            [answer_to_idx[ans] for ans in batch["answer"]],
            dtype=torch.long,
            device=device
        )

        visual_embeds = get_visual_embeddings_simple(images, visual_feature_extractor, device=device)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        training_error += (preds != labels).sum().item()
        total_training += labels.size(0)

    avg_training_loss = round(total_loss / len(train_loader), 4)
    training_accuracy = round(100 * (1 - training_error / total_training), 2)

    print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {avg_training_loss:.4f} | Training Accuracy: {training_accuracy:.2f}%")



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