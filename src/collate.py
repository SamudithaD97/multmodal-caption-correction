# src/collate.py
from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase
from torchvision import transforms


class CaptionCorrectionCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([self.image_transform(item["image"]) for item in batch])

        input_captions = [item["input_caption"] for item in batch]
        target_captions = [item["target_caption"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

        input_tok = self.tokenizer(
            input_captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        target_tok = self.tokenizer(
            target_captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": images,
            "input_ids": input_tok["input_ids"],
            "attention_mask": input_tok["attention_mask"],
            "target_ids": target_tok["input_ids"],
            "target_attention_mask": target_tok["attention_mask"],
            "labels": labels,
            "pair_type": [item["pair_type"] for item in batch],
            "sample_id": [item["sample_id"] for item in batch],
            "source_image_id": [item["source_image_id"] for item in batch],
        }