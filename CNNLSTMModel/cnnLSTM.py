import sys
import os

sys.path.append(os.path.abspath("."))

import os
import re
import json
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 2.1 Image encoder (CNN)
class ImageEncoder(nn.Module):

    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.cnn(x).squeeze()
        return self.image_fc(x)
    
# 2.2 Text input encoder (LSTM)
class TextInputEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return h[-1]
    
# 2.3 VQA Model (Fusion)
class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_answers):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_input_encoder = TextInputEncoder(vocab_size)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_answers)
        )

    def forward(self, image, question):
        img_features = self.image_encoder(image)
        ques_feature = self.text_input_encoder(question)
        fused = torch.cat([img_features, ques_feature], dim=1)
        return self.classifier(fused)
    
# 3.1 Evaluation function
def eval(model, criterion, loader):
    model.eval()
    total_eval_loss = 0
    total_eval = 0
    eval_error = 0

    with torch.no_grad():
        for images, questions, answers in loader:
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            total_eval_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            eval_error += (preds != answers).sum().item()
            total_eval += answers.size(0)

    avg_eval_loss = round(total_eval_loss / len(loader), 4)
    accuracy = round(100 * (1 - eval_error / total_eval), 2)
    return accuracy, avg_eval_loss

# 1.1 Image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 1.2 Text tokenization
# Simple word-level tokenizer
def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9 ]", "", sentence)
    return sentence.split()

# Build vocab
def build_vocab(questions, min_freq=1):
    counter = Counter()
    for q in questions:
        counter.update(tokenize(q))

    vocab = {
        "<pad>": 0, 
        "<unk>": 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# Encode question
def encode_question(question, vocab, max_len=20):
    tokens = tokenize(question)
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    ids = ids[:max_len]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))

# Answer vocab
def build_answer_vocab(answers):
    answers = [str(a).lower().strip() for a in answers]
    unique_answers = sorted(set(answers))

    answer_to_idx = {"<unk_ans>": 0}        # Solving missing val answers problem
    for ans in unique_answers:
        answer_to_idx[ans] = len(answer_to_idx)

    idx_to_answer = {i: a for a, i in answer_to_idx.items()}
    return answer_to_idx, idx_to_answer

# 1.4 Initialize dataset
DATA_ROOT = "VQA_RAD"
IMAGE_DIR = f"{DATA_ROOT}/images"

train_dataset = VqaRadDataset(
    json_path=f"{DATA_ROOT}/train.json",
    image_dir=IMAGE_DIR,
    vocab=vocab,
    answer_to_idx=answer_to_idx,
    transform=image_transform
)

val_dataset = VqaRadDataset(
    json_path=f"{DATA_ROOT}/val.json",
    image_dir=IMAGE_DIR,
    vocab=vocab,
    answer_to_idx=answer_to_idx,
    transform=image_transform
)

test_dataset = VqaRadDataset(
    json_path=f"{DATA_ROOT}/test.json",
    image_dir=IMAGE_DIR,
    vocab=vocab,
    answer_to_idx=answer_to_idx,
    transform=image_transform
)

# 1.5 Initialize DataLoader
BATCH_SIZE = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# 3.2 Training pipeline
model = VQAModel(vocab_size=len(vocab), num_answers=len(answer_to_idx))
# model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 35

# Variables for plotting
training_losses, training_accuracies = [], []
validation_losses, validation_accuracies = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_training = 0
    training_error = 0

    for images, questions, answers in train_loader:
        #images, questions, answers = images.to(device), questions.to(device), answers.to(device)

        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        training_error += (preds != answers).sum().item()
        total_training += answers.size(0)

    
    avg_training_loss = round(total_loss / len(train_loader), 4)
    training_accuracy = round(100 * (1 - training_error / total_training), 2)

    training_losses.append(avg_training_loss)
    training_accuracies.append(training_accuracy)

    validation_accuracy, validation_loss = eval(model, criterion, val_loader)
    validation_accuracies.append(validation_accuracy)
    validation_losses.append(validation_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {avg_training_loss:.4f} | Training Accuracy: {training_accuracy:.2f}% | Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.2f}%")