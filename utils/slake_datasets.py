import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

# Dataset class for baseline CNN + LSTM model
class CNNLSTMSlakeDataset(Dataset):

    def __init__(self, annotations_path, image_root, tokenizer, answer_to_idx, transform=None, max_question_len=30):
        self.samples = json.load(open(annotations_path, "r"))
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.answer_to_idx = answer_to_idx
        self.transform = transform
        self.max_question_len = max_question_len

    def __len__(self):
        """
        Docstring for __len__
        
        To handle batching
        """
        return len(self.samples)
    
    def padding_question(self, tokens):
        """
        Docstring for padding_question
        
        Truncate long questions
        """

        if len(tokens) >= self.max_question_len:
            return tokens[: self.max_question_len]
        
        return tokens + [0] * (self.max_question_len - len(tokens))
    
    def __getitem__(self, index):
        sample = self.samples[index]

        # Transform image
        image_path = os.path.join(self.image_root, sample["img_name"])
        image = Image.open(image_path).convert("RGB")   # 3-channel input

        if self.transform:
            image = self.transform(image)

        # Question tensor
        question_tokens = self.tokenizer(sample["question"])
        question_tokens = self.padding_question(question_tokens)
        question = torch.tensor(question_tokens, dtype=torch.long)

        # Answer tensor
        answer = torch.tensor(self.answer_to_idx[sample["answer"]], dtype=torch.long)

        return image, question, answer
    
# Dataset class for ViT + BERT model
class VisualBERTSlakeDataset(Dataset):

    def __init__(self, annotations_path, image_root, tokenizer, transform=None, max_length=32):
        self.samples = json.load(open(annotations_path, "r"))
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        #Transform image
        image_path = os.path.join(self.image_root, sample["img_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Question tensor
        encode = self.tokenizer(
            sample["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encode["input_ids"].squeeze(0)
        attention_mask = encode["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": encode["input_ids"].squeeze(0),
            "attention_mask": encode["attention_mask"].squeeze(0),
            "answer": sample["answer"],
            "question": sample["question"]
        }