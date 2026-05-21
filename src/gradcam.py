# gradcam.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from .model import CaptionCorrectionModel
from .dataset import CaptionCorrectionDataset
from .collate import CaptionCorrectionCollator


def get_special_ids(tokenizer):
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    if pad_id is None:
        raise ValueError("Tokenizer must have a pad token.")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must have BOS/CLS and EOS/SEP tokens.")

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
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model: CaptionCorrectionModel):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hooks = []

        # With current model this is the last ResNet conv block
        self.target_layer = model.image_encoder.feature_extractor[7]  # layer4

        self.hooks.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output  # (B, C, H, W)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B, C, H, W)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        image: (1, 3, H, W)
        """
        self.model.zero_grad(set_to_none=True)

        # We need gradients through the image path
        image = image.clone().detach().requires_grad_(True)

        logits = self.model.forward_match(image, input_ids, attention_mask)
        prob = torch.sigmoid(logits)

        # We explain the match score
        prob.backward(torch.ones_like(prob))

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

        # Global average pool gradients -> weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu(), prob.item()


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """
    image: PIL image
    heatmap: (H, W) in [0,1]
    """
    image = image.resize((heatmap.shape[1], heatmap.shape[0]))
    img_np = np.array(image).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]  # (H, W, 3)

    overlay = alpha * heat_rgb + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for match branch.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-path", type=str, default="runs/coco_correction/gradcam.png")
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer_name = args.tokenizer_name or ckpt.get("tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = build_model_from_checkpoint(ckpt, tokenizer).to(device)

    val_ds = CaptionCorrectionDataset(
        jsonl_path=str(Path(args.data_dir) / "manifests" / "val.jsonl"),
        image_root=None,
    )

    if args.sample_index >= 0:
        idx = args.sample_index
    else:
        import random
        idx = random.randint(0, len(val_ds) - 1)

    sample = val_ds[idx]
    print(f"Selected sample index: {idx}")
    print(f"Sample id: {sample['sample_id']}")
    print(f"Label: {sample['label']}")
    print(f"Pair type: {sample['pair_type']}")
    print(f"Input caption:  {sample['input_caption']}")
    print(f"Target caption: {sample['target_caption']}")
    print(f"Image path:     {sample['image_path']}")

    collator = CaptionCorrectionCollator(tokenizer=tokenizer, max_length=64)
    batch = collator([sample])

    image_tensor = batch["images"].to(device)          # (1, 3, 224, 224)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    cam_extractor = GradCAM(model)
    heatmap, match_prob = cam_extractor(image_tensor, input_ids, attention_mask)
    cam_extractor.remove_hooks()

    print(f"Match probability: {match_prob:.4f}")

    # Save overlay
    original_image = sample["image"]  # PIL image
    overlay = overlay_heatmap_on_image(original_image, heatmap.numpy(), alpha=0.45)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Grad-CAM (match prob={match_prob:.3f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    main()