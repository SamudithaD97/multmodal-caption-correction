# vit_gradcam.py
from __future__ import annotations

import argparse
from pathlib import Path

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
        vision_backbone=cfg.get("vision_backbone", "vit"),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


class ViTGradCAM:
    """
    Grad-CAM style visualization for ViT patch tokens.
    Hooks the last transformer block.
    """
    def __init__(self, model: CaptionCorrectionModel):
        self.model = model
        self.activations = None
        self.gradients = None

        # Last transformer block in torchvision ViT-B/16
        self.target_layer = model.image_encoder.vit.encoder.layers[-1]

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        # output shape: (B, 197, D) for ViT-B/16 with cls token
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        image: (1, 3, 224, 224)
        """
        self.model.zero_grad(set_to_none=True)

        image = image.clone().detach().requires_grad_(True)

        logits = self.model.forward_match(image, input_ids, attention_mask)
        prob = torch.sigmoid(logits)

        prob.backward(torch.ones_like(prob))

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients for ViT Grad-CAM.")

        # activations/grads: (B, 197, D)
        # remove cls token -> patch tokens only
        acts = self.activations[:, 1:, :]   # (B, 196, D)
        grads = self.gradients[:, 1:, :]    # (B, 196, D)

        # Grad-CAM style weights over channels
        weights = grads.mean(dim=1, keepdim=True)   # (B, 1, D)

        # Token saliency
        cam_tokens = torch.relu((acts * weights).sum(dim=-1))  # (B, 196)

        cam = cam_tokens[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # reshape 196 -> 14x14
        cam = cam.reshape(14, 14)

        return cam.detach().cpu(), prob.item()


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    image = image.resize((heatmap.shape[1], heatmap.shape[0]))
    img_np = np.array(image).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]

    overlay = alpha * heat_rgb + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="ViT Grad-CAM visualization for match branch.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/coco_correction")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-path", type=str, default="runs/coco_correction/vit_gradcam.png")
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

    image_tensor = batch["images"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    cam_extractor = ViTGradCAM(model)
    heatmap, match_prob = cam_extractor(image_tensor, input_ids, attention_mask)
    cam_extractor.remove_hooks()

    print(f"Match probability: {match_prob:.4f}")

    original_image = sample["image"]
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
    plt.title(f"ViT Grad-CAM (match prob={match_prob:.3f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()