# tune.py
from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch

from .train import TrainConfig, train


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_cfg = TrainConfig(
        data_dir="data/coco_correction",
        tokenizer_name="bert-base-uncased",
        batch_size=32,
        num_workers=0,          # better for Quartz
        max_length=64,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        pretrained_resnet=False,   # set according to your current setup
        freeze_image_encoder=True,
        lr=1e-4,
        weight_decay=1e-4,
        num_epochs=5,               # short sweep
        grad_clip=1.0,
        lambda_match=1.0,
        lambda_caption=1.5,
        caption_dropout_prob=0.3,
        caption_dropout_only_negatives=True,
        log_every=50,
        save_every=5,
        seed=42,
        device=device,
        out_dir="runs/tuning",
    )

    search_space = [
        {"lr": 1e-4, "caption_dropout_prob": 0.2, "lambda_caption": 1.0, "d_model": 256, "dropout": 0.1},
        {"lr": 1e-4, "caption_dropout_prob": 0.3, "lambda_caption": 1.5, "d_model": 256, "dropout": 0.1},
        {"lr": 2e-4, "caption_dropout_prob": 0.3, "lambda_caption": 1.5, "d_model": 256, "dropout": 0.1},
        {"lr": 1e-4, "caption_dropout_prob": 0.4, "lambda_caption": 1.5, "d_model": 384, "dropout": 0.1},
        {"lr": 2e-4, "caption_dropout_prob": 0.2, "lambda_caption": 2.0, "d_model": 384, "dropout": 0.2},
    ]

    results: List[Dict[str, object]] = []

    for i, params in enumerate(search_space, start=1):
        cfg = deepcopy(base_cfg)
        for k, v in params.items():
            setattr(cfg, k, v)

        cfg.out_dir = f"runs/tuning/exp_{i:02d}"
        print("\n" + "=" * 80)
        print(f"Running experiment {i}/{len(search_space)}")
        print(params)
        print("=" * 80)

        metrics = train(cfg)

        row = {
            "exp": i,
            **params,
            "val_match_loss": metrics["val_match_loss"],
            "val_caption_loss": metrics["val_caption_loss"],
            "val_match_acc": metrics["val_match_acc"],
            "test_match_loss": metrics["test_match_loss"],
            "test_caption_loss": metrics["test_caption_loss"],
            "test_match_acc": metrics["test_match_acc"],
            "out_dir": cfg.out_dir,
        }
        results.append(row)

    # Pick best by lowest caption loss, tie-break by highest match acc
    best = min(results, key=lambda r: (r["val_caption_loss"], -r["val_match_acc"]))

    out_dir = Path("runs/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "tuning_results.csv", results)
    with (out_dir / "tuning_results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    print("\nBest config:")
    print(best)


if __name__ == "__main__":
    main()