from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from pycocotools.coco import COCO


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_list(
    items: List[Any],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def build_pairs_for_split(
    split_name: str,
    source_items: List[Dict[str, Any]],
    all_captions: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for i, item in enumerate(source_items):
        image_id = item["image_id"]
        captions = item["captions"]
        image_path = item["image_path"]

        # Choose one correct caption for this image
        correct_caption = rng.choice(captions)

        # Choose a wrong caption from the global caption pool
        wrong_caption = rng.choice(all_captions)
        if wrong_caption in captions and len(all_captions) > 1:
            for _ in range(100):
                candidate = rng.choice(all_captions)
                if candidate not in captions:
                    wrong_caption = candidate
                    break

        # Positive example
        records.append({
            "sample_id": f"{split_name}-{i:06d}-pos",
            "source_image_id": image_id,
            "image_path": str(image_path),
            "input_caption": correct_caption,
            "target_caption": correct_caption,
            "label": 1,
            "pair_type": "positive",
        })

        # Negative example
        records.append({
            "sample_id": f"{split_name}-{i:06d}-neg",
            "source_image_id": image_id,
            "image_path": str(image_path),
            "input_caption": wrong_caption,
            "target_caption": correct_caption,
            "label": 0,
            "pair_type": "negative",
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a COCO caption-correction subset using existing local images.")
    parser.add_argument("--ann-file", type=str, required=True,
                        help="Path to captions_train2017.json or captions_val2017.json")
    parser.add_argument("--images-root", type=str, required=True,
                        help="Root directory containing the COCO images, e.g. coco/images/train2017")
    parser.add_argument("--out-dir", type=str, default="data/coco_correction",
                        help="Output directory for manifests")
    parser.add_argument("--num-source-images", type=int, default=10000,
                        help="Number of source COCO images to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--copy-missing", action="store_true",
                        help="If set, keep going even when some sampled images are missing; otherwise skip missing.")
    args = parser.parse_args()

    ann_file = Path(args.ann_file)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)

    manifests_dir = out_dir / "manifests"
    ensure_dir(manifests_dir)

    print("Loading COCO annotations...", flush=True)
    coco = COCO(str(ann_file))

    img_ids = list(coco.imgs.keys())
    rng = random.Random(args.seed)
    sampled_ids = rng.sample(img_ids, min(args.num_source_images, len(img_ids)))

    print(f"Sampling {len(sampled_ids)} source images from {len(img_ids)} total samples.", flush=True)

    source_items: List[Dict[str, Any]] = []
    all_captions: List[str] = []

    missing_count = 0

    for idx, img_id in enumerate(sampled_ids):
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        captions = [a["caption"].strip() for a in anns if a.get("caption", "").strip()]

        if not captions:
            continue

        file_name = img_info["file_name"]
        image_path = images_root / file_name

        if not image_path.exists():
            missing_count += 1
            if args.copy_missing:
                # keep going without error, but skip this sample
                continue
            else:
                continue

        source_items.append({
            "image_id": img_id,
            "file_name": file_name,
            "image_path": image_path,
            "coco_url": img_info.get("coco_url", ""),
            "captions": captions,
        })
        all_captions.extend(captions)

        if (idx + 1) % 100 == 0:
            print(f"Loaded metadata for {idx + 1}/{len(sampled_ids)} images", flush=True)

    print(f"Missing images skipped: {missing_count}", flush=True)

    train_items, val_items, test_items = split_list(
        source_items,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(
        f"Split sizes: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}",
        flush=True,
    )

    train_records = build_pairs_for_split("train", train_items, all_captions, seed=args.seed + 1)
    val_records = build_pairs_for_split("val", val_items, all_captions, seed=args.seed + 2)
    test_records = build_pairs_for_split("test", test_items, all_captions, seed=args.seed + 3)

    write_jsonl(manifests_dir / "train.jsonl", train_records)
    write_jsonl(manifests_dir / "val.jsonl", val_records)
    write_jsonl(manifests_dir / "test.jsonl", test_records)

    summary = {
        "ann_file": str(ann_file),
        "images_root": str(images_root),
        "num_source_images_requested": args.num_source_images,
        "num_source_images_found": len(source_items),
        "missing_images_skipped": missing_count,
        "train_source_images": len(train_items),
        "val_source_images": len(val_items),
        "test_source_images": len(test_items),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "seed": args.seed,
    }
    with (manifests_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.", flush=True)
    print(f"Manifests written to: {manifests_dir}", flush=True)
    print(f"Using existing images from: {images_root}", flush=True)


if __name__ == "__main__":
    main()