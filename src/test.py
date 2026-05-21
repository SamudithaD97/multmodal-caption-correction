# test.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import CaptionCorrectionModel
from src.dataloaders import build_dataloaders

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def get_special_ids(tokenizer):
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    if pad_id is None:
        raise ValueError("Tokenizer has no pad token.")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer needs BOS/CLS and EOS/SEP tokens.")

    return pad_id, bos_id, eos_id


def build_model_from_checkpoint(ckpt: Dict[str, Any], tokenizer) -> CaptionCorrectionModel:
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
        vision_backbone=cfg.get("vision_backbone", "vit"),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


@torch.no_grad()
def evaluate(
    model: CaptionCorrectionModel,
    loader: torch.utils.data.DataLoader,
    tokenizer,
    device: torch.device,
    pad_id: int,
    max_len: int,
    out_path: Path,
    max_print_examples: int = 10,
) -> Dict[str, float]:
    model.eval()

    match_criterion = nn.BCEWithLogitsLoss()
    caption_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_match_loss = 0.0
    total_caption_loss = 0.0

    all_labels: List[int] = []
    all_preds: List[int] = []

    bleu_refs: List[List[List[str]]] = []
    bleu_hyps: List[List[str]] = []

    prediction_rows: List[Dict[str, Any]] = []
    printed = 0

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

        probs = torch.sigmoid(match_logits)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.long().tolist())
        all_preds.extend(preds.tolist())

        generated_ids = model.generate(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_len=max_len,
        )

        batch_size = images.size(0)
        for i in range(batch_size):
            input_caption = tokenizer.decode(
                input_ids[i].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            target_caption = tokenizer.decode(
                target_ids[i].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            generated_caption = tokenizer.decode(
                generated_ids[i].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # BLEU expects: list of reference sentences, each sentence is token list
            bleu_refs.append([target_caption.split()])
            bleu_hyps.append(generated_caption.split())

            row = {
                "sample_id": batch["sample_id"][i],
                "source_image_id": batch["source_image_id"][i],
                "pair_type": batch["pair_type"][i],
                "label": int(labels[i].item()),
                "pred_label": int(preds[i].item()),
                "match_prob": float(probs[i].item()),
                "input_caption": input_caption,
                "target_caption": target_caption,
                "generated_caption": generated_caption,
                "image_path": batch["image_path"][i] if "image_path" in batch else None,
            }
            prediction_rows.append(row)

            if printed < max_print_examples:
                print("=" * 80)
                print(f"sample_id:      {row['sample_id']}")
                print(f"pair_type:      {row['pair_type']}")
                print(f"label:          {row['label']}")
                print(f"pred_label:     {row['pred_label']}")
                print(f"match_prob:     {row['match_prob']:.4f}")
                print(f"input_caption:  {row['input_caption']}")
                print(f"target_caption: {row['target_caption']}")
                print(f"generated:      {row['generated_caption']}")
                printed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(bleu_refs, bleu_hyps, smoothing_function=smoothing)

    print("\nConfusion matrix:")
    print(cm)

    return {
        "match_loss": total_match_loss / max(1, len(loader)),
        "caption_loss": total_caption_loss / max(1, len(loader)),
        "match_acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bleu": bleu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate caption correction model on the full validation set.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-print-examples", type=int, default=10)
    parser.add_argument("--out-path", type=str, default="runs/coco_correction/val_predictions.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer_name = ckpt.get("tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_id, _, _ = get_special_ids(tokenizer)

    _, val_loader, _, _ = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )

    model = build_model_from_checkpoint(ckpt, tokenizer).to(device)

    metrics = evaluate(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        pad_id=pad_id,
        max_len=args.max_length,
        out_path=Path(args.out_path),
        max_print_examples=args.max_print_examples,
    )

    print("\nValidation results")
    print(f"match_loss:   {metrics['match_loss']:.4f}")
    print(f"caption_loss: {metrics['caption_loss']:.4f}")
    print(f"match_acc:    {metrics['match_acc']:.4f}")
    print(f"precision:    {metrics['precision']:.4f}")
    print(f"recall:       {metrics['recall']:.4f}")
    print(f"f1:           {metrics['f1']:.4f}")
    print(f"bleu:         {metrics['bleu']:.4f}")


if __name__ == "__main__":
    main()