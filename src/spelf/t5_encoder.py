"""The frozen text encoder: a real pretrained T5, exactly as ELF uses one.

Earlier this project pretrained its own small T5 from scratch (span
corruption on this project's own data) because it started from a
closed-vocabulary tokenizer with no pretrained checkpoint to match. Now that
`tokenizer.py` uses T5's own real tokenizer, there's a real pretrained T5
checkpoint to match it -- `transformers.T5EncoderModel.from_pretrained`, the
same call ELF's own `modules/t5_encoder.py` makes. So: no training here at
all, just load it and freeze it, matching ELF exactly.

The only thing computed locally is latent normalization (`latent_mean`,
`latent_std` -- the scalar mean/std this frozen encoder's outputs are
rescaled by before entering the diffusion model, as in ELF's
`encoder_utils.encode_text`). `scripts/prepare_encoder.py` computes these
once, over the actual training collate pipeline, and caches them plus the
encoder/tokenizer names to `config.encoder_profile_path` -- a small JSON
file, not a weights checkpoint, since the weights are re-downloaded via
`from_pretrained` on every run rather than saved locally.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn


class T5Encoder(nn.Module):
    """Wraps a pretrained `T5EncoderModel`. Forward signature matches ELF's
    `T5Encoder` (`input_ids`, `attention_mask`, `deterministic`) ->
    `last_hidden_state`."""

    def __init__(self, model: nn.Module, d_model: int):
        super().__init__()
        self.model = model
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        was_training = self.model.training
        if deterministic:
            self.model.eval()
        try:
            # HF's mask-combination utilities require a bool (or int) mask,
            # and only build a padding+causal mask themselves from a 2D
            # input; our custom pairwise mask (condition tokens attend only
            # to condition tokens, target tokens attend to everything valid
            # -- see dataset.build_self_attn_cond_masks) is 3D (B, S, S) and
            # must be handed over already-prepared as 4D (B, 1, S, S), which
            # `create_bidirectional_mask` documents as "returned as-is".
            mask = attention_mask
            if mask is not None:
                mask = mask.bool()
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
            out = self.model(input_ids=input_ids, attention_mask=mask)
        finally:
            if not deterministic and was_training:
                self.model.train()
        return out.last_hidden_state


def build_pretrained_encoder(model_name: str, device=None) -> T5Encoder:
    """Load and freeze a pretrained T5 encoder (weights downloaded/cached by
    `transformers` on first use, exactly like ELF's `get_encoder`)."""
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(model_name)
    encoder = T5Encoder(model, model.config.d_model)
    if device is not None:
        encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_text(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 encoder: T5Encoder, latent_mean: float, latent_std: float) -> torch.Tensor:
    """Frozen-encoder forward pass + latent normalization (mirrors ELF's
    `encoder_utils.encode_text`)."""
    latents = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    return (latents - latent_mean) / latent_std


def save_encoder_profile(path: str, encoder_model_name: str, tokenizer_name: str,
                          latent_mean: float, latent_std: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "encoder_model_name": encoder_model_name,
            "tokenizer_name": tokenizer_name,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        }, f, indent=2)


def load_encoder_profile(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No encoder profile at {path!r}. Run scripts/prepare_encoder.py first."
        )
    with open(path, "r") as f:
        return json.load(f)
