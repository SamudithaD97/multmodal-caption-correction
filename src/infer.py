# infer.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .model import CaptionCorrectionModel
from src.dataset import CaptionCorrectionDataset
from src.collate import CaptionCorrectionCollator


def get_special_ids(tokenizer):
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    if pad_id is None:
        raise ValueError("Tokenizer has no pad token.")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer needs BOS/CLS and EOS/SEP tokens.")

    return pad_id, bos_id, eos_id


def build_model_from_checkpoint(ckpt, tokenizer):
    cfg = ckpt.get("config", {})
    pad_id, bos_id, eos_id = get_special_ids(tokenizer)

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
        vision_backbone='resnet'
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one random validation sample.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-index", type=int, default=-1,
                        help="If set >= 0, use that validation index instead of random.")
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer_name = args.tokenizer_name or ckpt.get("tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_id, _, _ = get_special_ids(tokenizer)

    model = build_model_from_checkpoint(ckpt, tokenizer).to(device)

    val_ds = CaptionCorrectionDataset(
        jsonl_path=str(Path(args.data_dir) / "manifests" / "val.jsonl"),
        image_root=None,
    )

    if args.sample_index >= 0:
        idx = args.sample_index
    else:
        idx = random.randint(0, len(val_ds) - 1)

    sample = val_ds[idx]
    print(f"Selected validation index: {idx}")
    print(f"Sample id: {sample['sample_id']}")
    print(f"Label: {sample['label']} ({'positive' if sample['label'] == 1 else 'negative'})")
    print(f"Pair type: {sample['pair_type']}")
    print(f"Image path: {sample['image_path']}")
    print(f"Input caption: {sample['input_caption']}")
    print(f"Target caption: {sample['target_caption']}")

    collator = CaptionCorrectionCollator(tokenizer=tokenizer, max_length=args.max_length)
    batch = collator([sample])

    images = batch["images"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids[:, :-1],
            decoder_attention_mask=(target_ids[:, :-1] != pad_id).long(),
        )

        match_logits = outputs["match_logits"].squeeze(-1)
        match_prob = torch.sigmoid(match_logits).item()
        pred_label = 1 if match_prob >= 0.5 else 0

        generated_ids = model.generate(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_len=args.max_length,
        )

    generated_caption = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("\nInference result")
    print(f"Match probability: {match_prob:.4f}")
    print(f"Predicted label:   {pred_label}")
    print(f"Generated caption:  {generated_caption}")

    if sample["label"] == 0:
        print("\nThis was a negative sample, so the generated caption is the correction.")
    else:
        print("\nThis was a positive sample, so the generated caption should match the input/target caption.")


if __name__ == "__main__":
    main()