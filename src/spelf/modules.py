"""Shared building blocks used by GraphEncoder, DLMDecoder, and GPTDecoder.

Convention: masks follow the tokenizer's convention (True = real content, False = PAD)
everywhere in this codebase's public signatures. `nn.MultiheadAttention` wants the
opposite (`key_padding_mask`: True = ignore), so the inversion happens right at the
attention call, not in caller code.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import tokenizer as tok


class SharedEmbedding(nn.Module):
    """Token embedding table shared, by checkpoint (not object identity), across the
    graph encoder, DLM, and ARLM. The encoder pretrains it; downstream DLM/ARLM
    training loads that checkpoint and calls `freeze_pretrained_rows()`, which installs
    a gradient hook zeroing gradients for every row except `<P>`/`<EOS>` -- those two
    tokens never appear in the encoder's own input, so they're left trainable. Combine
    with zero weight-decay on this parameter (see common.build_optimizer) so frozen
    rows are truly static, not just gradient-free (AdamW would otherwise still shrink
    them via weight decay)."""

    def __init__(self, vocab_size: int = tok.VOCAB_SIZE, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Default nn.Embedding init is N(0,1) per element, so a 128-dim row has norm
        # ~sqrt(128); dotting two such rows in `unembed` gives huge, poorly-calibrated
        # logits. Use the conventional small-std init (as in BERT/GPT-2) instead.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self._freeze_hook_handle = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] -> logits [..., vocab_size], tied to the embedding table."""
        return x @ self.embedding.weight.t()

    def freeze_pretrained_rows(self, trainable_token_ids=(tok.P, tok.EOS)) -> None:
        if self._freeze_hook_handle is not None:
            self._freeze_hook_handle.remove()
        vocab_size = self.embedding.weight.shape[0]
        frozen_mask = torch.ones(vocab_size, dtype=torch.bool)
        frozen_mask[list(trainable_token_ids)] = False

        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[frozen_mask.to(grad.device)] = 0
            return grad

        self._freeze_hook_handle = self.embedding.weight.register_hook(hook)


class LearnedPosEnc(nn.Module):
    def __init__(self, max_len: int, d_model: int = 128):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        length = x.shape[1]
        positions = torch.arange(length, device=x.device)
        return x + self.pos_emb(positions)[None, :, :]


class TimeEmbedding(nn.Module):
    """Sinusoidal features of scalar diffusion time t in [0,1] -> MLP, standard
    diffusion-timestep embedding."""

    def __init__(self, d_model: int = 128, n_freq: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        device = t.device
        freqs = torch.exp(torch.linspace(0, math.log(10000.0), self.n_freq, device=device))
        args = t[:, None] * freqs[None, :]  # [B, n_freq]
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(feats)


class EncoderLayer(nn.Module):
    """Pre-LN bidirectional self-attention block (no causal mask, no cross-attention)."""

    def __init__(self, d_model: int = 128, n_heads: int = 8, d_mlp: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # dropout=0 here: nn.MultiheadAttention's SDPA fast path doesn't support
        # dropout_p>0 during training on MPS (NotImplementedError). Regularization
        # still comes from the post-attention/MLP `self.dropout` below.
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_mask: [B, L], True = real token (tokenizer convention)
        kpm = ~input_mask if input_mask is not None else None
        h = self.ln1(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN self-attention (causal or bidirectional) + cross-attention to `context` +
    MLP. Used identically by DLMDecoder (causal=False) and GPTDecoder (causal=True) so
    the conditioning mechanism is the same for both, isolating the decoder's own
    architecture as the variable under study."""

    def __init__(self, d_model: int = 128, n_heads: int = 8, d_mlp: int = 512,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(d_model)
        # dropout=0 on both attention modules here for the same MPS-SDPA reason as
        # EncoderLayer above; `self.dropout` below still regularizes each sublayer output.
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, D], context: [B, L_ctx, D], context_mask: [B, L_ctx] True=real
        length = x.shape[1]
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((length, length), float("-inf"), device=x.device), diagonal=1
            )
        h = self.ln1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        ctx_kpm = ~context_mask if context_mask is not None else None
        cross_out, _ = self.cross_attn(h, context, context, key_padding_mask=ctx_kpm, need_weights=False)
        x = x + self.dropout(cross_out)
        h = self.ln3(x)
        x = x + self.dropout(self.mlp(h))
        return x
