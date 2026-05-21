# src/dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class CaptionCorrectionDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: Optional[str] = None):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root is not None else None

        self.rows: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        image_path = Path(row["image_path"])
        if self.image_root is not None and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "input_caption": row["input_caption"],
            "target_caption": row["target_caption"],
            "label": int(row["label"]),
            "pair_type": row["pair_type"],
            "sample_id": row["sample_id"],
            "source_image_id": row["source_image_id"],
            "image_path": str(image_path),
        }