import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class VQARadVisualBERTDataset(Dataset):
    """
    PyTorch Dataset for VQA-RAD compatible with VisualBERT.
    """

    def __init__(
        self,
        annotations_path: str,
        image_root: str,
        tokenizer,
        transform=None,
        max_question_length: int = 32
    ):
        """
        Args:
            annotations_path (str): Path to train/val/test JSON
            image_root (str): Directory containing images
            tokenizer: HuggingFace tokenizer (BERT)
            transform: torchvision transforms
            max_question_length (int): Max tokens for question
        """

        with open(annotations_path, "r") as f:
            self.samples = json.load(f)

        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_question_length = max_question_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(self.image_root, sample["image_name"])
        question = sample["question"]
        answer = sample["answer"]

        # --- Load Image ---
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # --- Tokenize Question ---
        tokens = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt"
        )

        return {
            "image": image,  # Tensor [3, H, W]
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "question": question,
            "answer": answer
        }
