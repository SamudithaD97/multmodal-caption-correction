# src/dataloaders.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import CaptionCorrectionDataset
from .collate import CaptionCorrectionCollator


def build_dataloaders(
    data_dir: str = "coco_correction_data",
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    data_dir = Path(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = CaptionCorrectionDataset(
        jsonl_path=str(data_dir / "manifests" / "train.jsonl"),
        image_root=None,
    )
    val_ds = CaptionCorrectionDataset(
        jsonl_path=str(data_dir / "manifests" / "val.jsonl"),
        image_root=None,
    )
    test_ds = CaptionCorrectionDataset(
        jsonl_path=str(data_dir / "manifests" / "test.jsonl"),
        image_root=None,
    )

    collate_fn = CaptionCorrectionCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, tokenizer