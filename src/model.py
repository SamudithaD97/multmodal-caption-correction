# model.py
from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Expects input shape: (B, T, D)
    Returns same shape.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FrozenResNetEncoder(nn.Module):
    """
    Frozen ResNet-50 image encoder.
    Input:  (B, 3, 224, 224)
    Output: (B, 2048)
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove fc

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        requires_grad = images.requires_grad or any(p.requires_grad for p in self.parameters())
        with torch.set_grad_enabled(requires_grad):
            feats = self.feature_extractor(images)  # (B, 2048, 1, 1)
            feats = feats.flatten(1)  # (B, 2048)
        return feats

class FrozenViTEncoder(nn.Module):
    """
    Frozen ViT-B/16 image encoder.
    Input:  (B, 3, 224, 224)
    Output: (B, 768)
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # ViT-B/16 hidden size is 768
        self.output_dim = getattr(self.vit, "hidden_dim", 768)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224)
        returns: (B, 768)
        """
        requires_grad = images.requires_grad or any(p.requires_grad for p in self.parameters())
        with torch.set_grad_enabled(requires_grad):
            x = self.vit._process_input(images)              # (B, num_patches, D)
            n = x.shape[0]
            cls_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)             # prepend CLS token
            x = self.vit.encoder(x)                          # transformer encoder
            x = x[:, 0]                                      # CLS token representation
        return x

class CaptionCorrectionModel(nn.Module):
    """
    Multimodal caption correction model.

    Discriminative branch:
        image + caption -> match score

    Generative branch:
        image + input_caption -> corrected caption
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
        pretrained_resnet: bool = True,
        freeze_image_encoder: bool = True,
        vision_backbone: str = "resnet",   # "resnet" or "vit"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.max_len = max_len

        # Image encoder
        if vision_backbone == "resnet":
            self.image_encoder = FrozenResNetEncoder(
                pretrained=pretrained_resnet,
                freeze=freeze_image_encoder,
            )
            image_dim = 2048
        elif vision_backbone == "vit":
            self.image_encoder = FrozenViTEncoder(
                pretrained=pretrained_resnet,
                freeze=freeze_image_encoder,
            )
            image_dim = self.image_encoder.output_dim
        else:
            raise ValueError(f"Unknown vision_backbone: {vision_backbone}")

        self.image_proj = nn.Linear(image_dim, d_model)
        # self.image_proj = nn.Linear(2048, d_model)

        # Shared token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        # Text encoder for the input caption
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Decoder for caption correction
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Discriminative matching head
        self.match_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224)
        returns: (B, d_model)
        """
        img_feat = self.image_encoder(images)      # (B, 2048)
        img_feat = self.image_proj(img_feat)       # (B, d_model)
        img_feat = self.dropout(img_feat)
        return img_feat

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: (B, T)
        attention_mask: (B, T) with 1 for real tokens, 0 for pad
        returns: (B, T, d_model)
        """
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)  # (B, T, D)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        x = x.transpose(0, 1)  # (T, B, D)
        src_key_padding_mask = attention_mask == 0  # True where padding

        x = self.text_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # (B, T, D)
        return x

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        attention_mask: (B, T), 1 for valid tokens
        returns: (B, D)
        """
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode_text_summary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns pooled text vector: (B, d_model)
        """
        text_feats = self._encode_text(input_ids, attention_mask)
        pooled = self.masked_mean_pool(text_feats, attention_mask)
        return pooled

    def _build_memory(
        self,
        img_feat: torch.Tensor,
        text_feats: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create decoder memory by concatenating image token + text encoder outputs.

        img_feat: (B, D)
        text_feats: (B, T, D)
        text_attention_mask: (B, T)

        returns:
            memory: (B, 1+T, D)
            memory_padding_mask: (B, 1+T)
        """
        img_token = img_feat.unsqueeze(1)  # (B, 1, D)
        memory = torch.cat([img_token, text_feats], dim=1)  # (B, 1+T, D)

        img_mask = torch.zeros((text_attention_mask.size(0), 1), dtype=torch.bool, device=text_attention_mask.device)
        memory_padding_mask = torch.cat([img_mask, text_attention_mask == 0], dim=1)  # (B, 1+T)

        return memory, memory_padding_mask

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask for decoder self-attention.
        Shape: (T, T)
        """
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def shift_right(self, target_ids: torch.Tensor) -> torch.Tensor:
        """
        target_ids: (B, T)
        Returns decoder input ids with BOS prepended and last token removed.
        """
        bos = torch.full(
            (target_ids.size(0), 1),
            self.bos_id,
            dtype=target_ids.dtype,
            device=target_ids.device,
        )
        return torch.cat([bos, target_ids[:, :-1]], dim=1)

    def forward_match(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminative branch.
        Returns logits: (B, 1)
        """
        img_feat = self.encode_image(images)  # (B, D)
        txt_feat = self.encode_text_summary(input_ids, attention_mask)  # (B, D)

        fused = torch.cat([img_feat, txt_feat], dim=-1)  # (B, 2D)
        logits = self.match_head(fused)  # (B, 1)
        return logits

    def forward_caption(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generative branch.

        input_ids/attention_mask = the input caption (wrong or correct caption)
        target_ids = corrected caption tokens (used for shifting if decoder_input_ids is not given)

        Returns:
            caption_logits: (B, T, vocab_size)
        """
        if decoder_input_ids is None:
            if target_ids is None:
                raise ValueError("Either decoder_input_ids or target_ids must be provided.")
            decoder_input_ids = self.shift_right(target_ids)
            decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        if decoder_attention_mask is None:
            decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        img_feat = self.encode_image(images)  # (B, D)
        text_feats = self._encode_text(input_ids, attention_mask)  # (B, T_src, D)

        memory, memory_padding_mask = self._build_memory(img_feat, text_feats, attention_mask)

        # Decoder input embeddings
        tgt = self.token_embedding(decoder_input_ids) * math.sqrt(self.d_model)  # (B, T_tgt, D)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        tgt = tgt.transpose(0, 1)  # (T_tgt, B, D)
        memory = memory.transpose(0, 1)  # (T_mem, B, D)

        tgt_mask = self._generate_square_subsequent_mask(
            sz=decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        tgt_key_padding_mask = decoder_attention_mask == 0

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )  # (T_tgt, B, D)

        out = out.transpose(0, 1)  # (B, T_tgt, D)
        logits = self.lm_head(out)  # (B, T_tgt, vocab_size)
        return logits

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns a dict with:
            match_logits: (B, 1)
            caption_logits: (B, T, vocab_size) if target_ids/decoder_input_ids are provided
        """
        outputs: Dict[str, torch.Tensor] = {}
        outputs["match_logits"] = self.forward_match(images, input_ids, attention_mask)

        if target_ids is not None or decoder_input_ids is not None:
            outputs["caption_logits"] = self.forward_caption(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int = 64,
    ) -> torch.Tensor:
        """
        Greedy decoding for caption correction.
        Returns generated token ids: (B, <= max_len)
        """
        self.eval()

        img_feat = self.encode_image(images)  # (B, D)
        text_feats = self._encode_text(input_ids, attention_mask)
        memory, memory_padding_mask = self._build_memory(img_feat, text_feats, attention_mask)

        memory = memory.transpose(0, 1)  # (T_mem, B, D)

        batch_size = images.size(0)
        generated = torch.full(
            (batch_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=images.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = self.pos_encoding(tgt)
            tgt = self.dropout(tgt)

            tgt = tgt.transpose(0, 1)  # (T, B, D)
            tgt_mask = self._generate_square_subsequent_mask(
                sz=generated.size(1),
                device=images.device,
            )
            tgt_key_padding_mask = generated == self.pad_id

            out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )
            out = out.transpose(0, 1)  # (B, T, D)
            logits = self.lm_head(out[:, -1, :])  # (B, vocab_size)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.eos_id)
            if finished.all():
                break

        return generated