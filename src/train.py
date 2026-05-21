# train.py
from __future__ import annotations

import argparse
import math
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from .model import CaptionCorrectionModel
from .dataloaders import build_dataloaders


@dataclass
class TrainConfig:
    data_dir: str = "data/coco_correction"
    tokenizer_name: str = "bert-base-uncased"
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 64

    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    pretrained_resnet: bool = True
    freeze_image_encoder: bool = True

    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    grad_clip: float = 1.0

    lambda_match: float = 1.0
    lambda_caption: float = 1.0

    log_every: int = 50
    save_every: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/coco_correction"

    caption_dropout_prob: float = 0.6
    caption_dropout_only_negatives: bool = True


def apply_caption_dropout_on_negatives(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    p: float = 0.3,
):
    """
    Randomly masks caption tokens only for negative samples (label == 0).
    Keeps attention_mask unchanged.
    """
    if p <= 0.0:
        return input_ids, attention_mask

    if tokenizer.mask_token_id is None:
        # If tokenizer has no [MASK], do nothing.
        return input_ids, attention_mask

    corrupted = input_ids.clone()
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    for i in neg_indices.tolist():
        if random.random() < p:
            real_positions = attention_mask[i].bool()
            corrupted[i, real_positions] = tokenizer.mask_token_id

    return corrupted, attention_mask

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(
    model: CaptionCorrectionModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.eval()

    match_criterion = nn.BCEWithLogitsLoss()
    caption_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_match_loss = 0.0
    total_caption_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_caption_tokens = 0
    total_caption_loss_tokens = 0.0

    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        labels = batch["labels"].to(device)

        decoder_input_ids = target_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != pad_id).long()
        lm_labels = target_ids[:, 1:]

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        match_logits = outputs["match_logits"].squeeze(-1)
        caption_logits = outputs["caption_logits"]

        match_loss = match_criterion(match_logits, labels)
        caption_loss = caption_criterion(
            caption_logits.reshape(-1, caption_logits.size(-1)),
            lm_labels.reshape(-1),
        )

        total_match_loss += match_loss.item()
        total_caption_loss += caption_loss.item()

        preds = (torch.sigmoid(match_logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_examples += labels.numel()

        non_pad = (lm_labels != pad_id).sum().item()
        total_caption_tokens += non_pad
        total_caption_loss_tokens += caption_loss.item() * max(1, non_pad)

    return {
        "match_loss": total_match_loss / max(1, len(loader)),
        "caption_loss": total_caption_loss / max(1, len(loader)),
        "match_acc": total_correct / max(1, total_examples),
    }


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: CaptionCorrectionModel,
    optimizer: torch.optim.Optimizer,
    tokenizer_name: str,
    config: TrainConfig,
) -> None:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_name": tokenizer_name,
            "config": config.__dict__,
        },
        path,
    )

def apply_caption_dropout(input_ids, attention_mask, tokenizer, p=0.3):
    if random.random() > p:
        return input_ids, attention_mask

    # Replace the whole caption with [MASK] tokens
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        # fallback: keep only pad tokens
        input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(attention_mask)
        return input_ids, attention_mask

    corrupted = input_ids.clone()
    corrupted = torch.where(attention_mask.bool(), torch.full_like(corrupted, mask_id), corrupted)
    return corrupted, attention_mask



def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataloaders...", flush=True)
    train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
        data_dir=cfg.data_dir,
        tokenizer_name=cfg.tokenizer_name,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_length=cfg.max_length,
    )

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    if pad_id is None:
        raise ValueError("Tokenizer must have a pad token.")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must have BOS/CLS and EOS/SEP tokens.")

    vocab_size = len(tokenizer)

    print("Building model...", flush=True)
    model = CaptionCorrectionModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_length,
        pretrained_resnet=cfg.pretrained_resnet,
        freeze_image_encoder=cfg.freeze_image_encoder,
        vision_backbone='resnet',
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}", flush=True)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    match_criterion = nn.BCEWithLogitsLoss()
    caption_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_match_loss = 0.0
        running_caption_loss = 0.0
        running_total_loss = 0.0
        running_correct = 0
        running_examples = 0

        print(f"\nEpoch {epoch}/{cfg.num_epochs}", flush=True)

        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)
            input_ids, attention_mask = input_ids, attention_mask = apply_caption_dropout_on_negatives(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                tokenizer=tokenizer,
                p=cfg.caption_dropout_prob,
            )


            # Teacher forcing:
            # decoder input = everything except last token
            # label sequence = everything except first token
            decoder_input_ids = target_ids[:, :-1]
            decoder_attention_mask = (decoder_input_ids != pad_id).long()
            lm_labels = target_ids[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            match_logits = outputs["match_logits"].squeeze(-1)
            caption_logits = outputs["caption_logits"]

            match_loss = match_criterion(match_logits, labels)
            caption_loss = caption_criterion(
                caption_logits.reshape(-1, caption_logits.size(-1)),
                lm_labels.reshape(-1),
            )

            total_loss = cfg.lambda_match * match_loss + cfg.lambda_caption * caption_loss
            total_loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(match_logits) >= 0.5).float()
                running_correct += (preds == labels).sum().item()
                running_examples += labels.numel()

            running_match_loss += match_loss.item()
            running_caption_loss += caption_loss.item()
            running_total_loss += total_loss.item()

            if step % cfg.log_every == 0 or step == 1 or step == len(train_loader):
                avg_match = running_match_loss / step
                avg_caption = running_caption_loss / step
                avg_total = running_total_loss / step
                acc = running_correct / max(1, running_examples)

                print(
                    f"  step {step:4d}/{len(train_loader)} | "
                    f"total={avg_total:.4f} | "
                    f"match={avg_match:.4f} | "
                    f"caption={avg_caption:.4f} | "
                    f"acc={acc:.4f}",
                    flush=True,
                )

        train_match_acc = running_correct / max(1, running_examples)
        train_match_loss = running_match_loss / max(1, len(train_loader))
        train_caption_loss = running_caption_loss / max(1, len(train_loader))
        train_total_loss = running_total_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device, pad_id)

        print(
            f"Epoch {epoch} summary | "
            f"train_total={train_total_loss:.4f} | "
            f"train_match={train_match_loss:.4f} | "
            f"train_caption={train_caption_loss:.4f} | "
            f"train_acc={train_match_acc:.4f} | "
            f"val_match={val_metrics['match_loss']:.4f} | "
            f"val_caption={val_metrics['caption_loss']:.4f} | "
            f"val_acc={val_metrics['match_acc']:.4f}",
            flush=True,
        )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(out_dir, epoch, model, optimizer, cfg.tokenizer_name, cfg)

        val_total = val_metrics["match_loss"] + val_metrics["caption_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = out_dir / "checkpoints" / "best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tokenizer_name": cfg.tokenizer_name,
                    "config": cfg.__dict__,
                },
                best_path,
            )
            print(f"  saved best checkpoint -> {best_path}", flush=True)

    test_metrics = evaluate(model, test_loader, device, pad_id)
    print(
        f"\nFinal test | match_loss={test_metrics['match_loss']:.4f} | "
        f"caption_loss={test_metrics['caption_loss']:.4f} | "
        f"match_acc={test_metrics['match_acc']:.4f}",
        flush=True,
    )
    summary = {
        "val_match_loss": val_metrics["match_loss"],
        "val_caption_loss": val_metrics["caption_loss"],
        "val_match_acc": val_metrics["match_acc"],
        "test_match_loss": test_metrics["match_loss"],
        "test_caption_loss": test_metrics["caption_loss"],
        "test_match_acc": test_metrics["match_acc"],
        "config": cfg.__dict__,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train multimodal caption correction model on COCO.")
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrained-resnet", action="store_true")
    parser.add_argument("--freeze-image-encoder", action="store_true")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lambda-match", type=float, default=1.0)
    parser.add_argument("--lambda-caption", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/coco_correction_with_highdropout")

    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_resnet=args.pretrained_resnet,
        freeze_image_encoder=args.freeze_image_encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        grad_clip=args.grad_clip,
        lambda_match=args.lambda_match,
        lambda_caption=args.lambda_caption,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    train(parse_args())