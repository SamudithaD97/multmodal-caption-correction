# saliency_aggregate.py
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from .model import CaptionCorrectionModel
from .dataset import CaptionCorrectionDataset
from .collate import CaptionCorrectionCollator


@dataclass
class TokenStats:
    saliency_sum: float = 0.0
    abs_saliency_sum: float = 0.0
    count: int = 0

    def add(self, saliency: float) -> None:
        self.saliency_sum += saliency
        self.abs_saliency_sum += abs(saliency)
        self.count += 1

    @property
    def mean_saliency(self) -> float:
        return self.saliency_sum / self.count if self.count > 0 else 0.0

    @property
    def mean_abs_saliency(self) -> float:
        return self.abs_saliency_sum / self.count if self.count > 0 else 0.0


def get_special_ids(tokenizer):
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    if pad_id is None:
        raise ValueError("Tokenizer must have a pad token.")
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must have CLS/BOS and SEP/EOS tokens.")
    if mask_id is None:
        raise ValueError("Tokenizer must have a mask token.")

    return pad_id, cls_id, sep_id, mask_id


def build_model_from_checkpoint(ckpt, tokenizer):
    cfg = ckpt.get("config", {})
    pad_id, bos_id, eos_id, _ = get_special_ids(tokenizer)

    model = CaptionCorrectionModel(
        vocab_size=len(tokenizer),
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 2),
        num_decoder_layers=cfg.get("num_decoder_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        max_len=cfg.get("max_length", 64),
        pretrained_resnet=cfg.get("pretrained_resnet", True),
        freeze_image_encoder=cfg.get("freeze_image_encoder", True),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def pretty_token(tok: str) -> str:
    # BERT-style subword cleanup for readability
    return tok.replace("##", "")


@torch.no_grad()
def saliency_for_one_sample(
    model: CaptionCorrectionModel,
    tokenizer,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> List[Tuple[str, float]]:
    """
    Returns list of (token, saliency) for one example.
    Saliency = original_prob - masked_prob
    """
    _, cls_id, sep_id, mask_id = get_special_ids(tokenizer)

    image = image.unsqueeze(0).to(device)                # (1, 3, H, W)
    input_ids = input_ids.unsqueeze(0).to(device)        # (1, T)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    # Original probability
    orig_logits = model.forward_match(image, input_ids, attention_mask)
    orig_prob = torch.sigmoid(orig_logits).item()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    candidate_positions = []

    for i, tok_id in enumerate(input_ids.squeeze(0).tolist()):
        if tok_id in {tokenizer.pad_token_id, cls_id, sep_id}:
            continue
        if attention_mask[0, i].item() == 0:
            continue
        candidate_positions.append(i)

    if not candidate_positions:
        return []

    # Batch-mask all candidate tokens at once for efficiency
    masked_inputs = input_ids.repeat(len(candidate_positions), 1)
    masked_masks = attention_mask.repeat(len(candidate_positions), 1)
    for row_idx, pos in enumerate(candidate_positions):
        masked_inputs[row_idx, pos] = mask_id

    masked_logits = model.forward_match(
        image.repeat(len(candidate_positions), 1, 1, 1),
        masked_inputs,
        masked_masks,
    )
    masked_probs = torch.sigmoid(masked_logits).squeeze(-1).tolist()

    results: List[Tuple[str, float]] = []
    for pos, masked_prob in zip(candidate_positions, masked_probs):
        tok = pretty_token(tokens[pos])
        saliency = orig_prob - float(masked_prob)
        results.append((tok, saliency))

    return results


def update_stats(stats: Dict[str, TokenStats], token: str, saliency: float) -> None:
    if token not in stats:
        stats[token] = TokenStats()
    stats[token].add(saliency)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_top_tokens(rows: List[Dict[str, object]], title: str, out_path: Path, top_k: int = 20) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda x: x["mean_abs_saliency"], reverse=True)[:top_k]
    labels = [str(r["token"]) for r in rows][::-1]
    values = [float(r["mean_saliency"]) for r in rows][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel("Mean saliency (original_prob - masked_prob)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate token saliency over many validation samples.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--sample-start", type=int, default=0,
                        help="Start index in validation set. Useful for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/coco_correction/saliency")
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer_name = args.tokenizer_name or ckpt.get("tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = build_model_from_checkpoint(ckpt, tokenizer).to(device)

    val_ds = CaptionCorrectionDataset(
        jsonl_path=str(Path(args.data_dir) / "manifests" / "val.jsonl"),
        image_root=None,
    )

    collator = CaptionCorrectionCollator(tokenizer=tokenizer, max_length=64)

    n = min(args.num_samples, len(val_ds) - args.sample_start)
    if n <= 0:
        raise ValueError("No validation samples available in the requested range.")

    overall_stats: Dict[str, TokenStats] = defaultdict(TokenStats)
    pos_stats: Dict[str, TokenStats] = defaultdict(TokenStats)
    neg_stats: Dict[str, TokenStats] = defaultdict(TokenStats)

    match_counts = {"positive": 0, "negative": 0}

    for idx in range(args.sample_start, args.sample_start + n):
        sample = val_ds[idx]
        batch = collator([sample])

        image = batch["images"][0]
        input_ids = batch["input_ids"][0]
        attention_mask = batch["attention_mask"][0]

        token_saliency = saliency_for_one_sample(
            model=model,
            tokenizer=tokenizer,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
        )

        group = "positive" if sample["label"] == 1 else "negative"
        match_counts[group] += 1

        for token, sal in token_saliency:
            update_stats(overall_stats, token, sal)
            if group == "positive":
                update_stats(pos_stats, token, sal)
            else:
                update_stats(neg_stats, token, sal)

        if (idx - args.sample_start + 1) % 25 == 0:
            print(f"Processed {idx - args.sample_start + 1}/{n} samples", flush=True)

    def stats_to_rows(stats: Dict[str, TokenStats]) -> List[Dict[str, object]]:
        rows = []
        for token, s in stats.items():
            if s.count < args.min_count:
                continue
            rows.append({
                "token": token,
                "count": s.count,
                "mean_saliency": s.mean_saliency,
                "mean_abs_saliency": s.mean_abs_saliency,
            })
        rows.sort(key=lambda x: x["mean_abs_saliency"], reverse=True)
        return rows

    overall_rows = stats_to_rows(overall_stats)
    pos_rows = stats_to_rows(pos_stats)
    neg_rows = stats_to_rows(neg_stats)

    write_csv(out_dir / "overall_token_saliency.csv", overall_rows)
    write_csv(out_dir / "positive_token_saliency.csv", pos_rows)
    write_csv(out_dir / "negative_token_saliency.csv", neg_rows)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": n,
                "sample_start": args.sample_start,
                "min_count": args.min_count,
                "match_counts": match_counts,
                "top_overall": overall_rows[:20],
                "top_positive": pos_rows[:20],
                "top_negative": neg_rows[:20],
            },
            f,
            indent=2,
        )

    plot_top_tokens(
        overall_rows,
        title="Overall token saliency",
        out_path=out_dir / "overall_token_saliency.png",
        top_k=20,
    )
    plot_top_tokens(
        pos_rows,
        title="Positive sample token saliency",
        out_path=out_dir / "positive_token_saliency.png",
        top_k=20,
    )
    plot_top_tokens(
        neg_rows,
        title="Negative sample token saliency",
        out_path=out_dir / "negative_token_saliency.png",
        top_k=20,
    )

    print("\nDone.")
    print(f"Saved results to: {out_dir}")
    print(f"Overall CSV: {out_dir / 'overall_token_saliency.csv'}")
    print(f"Plot:        {out_dir / 'overall_token_saliency.png'}")


if __name__ == "__main__":
    main()