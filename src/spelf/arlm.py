"""GPT-style causal ARLM decoder: standard teacher-forced next-token prediction,
conditioned on the frozen graph encoder via cross-attention (same DecoderLayer /
conditioning mechanism as the DLM, for as fair a comparison as possible)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as tok
from .modules import DecoderLayer, LearnedPosEnc, SharedEmbedding


class GPTDecoder(nn.Module):
    def __init__(self, l_tgt: int = tok.TARGET_LENGTH, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 8, d_mlp: int = 512, dropout: float = 0.1,
                 embedding: SharedEmbedding | None = None):
        super().__init__()
        self.l_tgt = l_tgt
        self.d_model = d_model
        self.embedding = embedding if embedding is not None else SharedEmbedding(d_model=d_model)
        self.pos_enc = LearnedPosEnc(l_tgt, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_mlp, dropout, causal=True) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L] -> logits: [B, L, vocab]
        h = self.embedding(token_ids)
        h = self.pos_enc(h)
        for layer in self.layers:
            h = layer(h, context, context_mask)
        h = self.out_ln(h)
        return self.embedding.unembed(h)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, context_mask: torch.Tensor,
                 max_len: int | None = None, start_id: int = tok.P,
                 eos_id: int = tok.EOS, pad_id: int = tok.PAD) -> torch.Tensor:
        return sample(self, context, context_mask, max_len=max_len,
                      start_id=start_id, eos_id=eos_id, pad_id=pad_id)


def loss(model: GPTDecoder, context: torch.Tensor, context_mask: torch.Tensor,
          target_ids: torch.Tensor, pad_id: int = tok.PAD) -> dict:
    """Standard teacher-forced next-token cross-entropy; `pad_id` positions excluded
    from the loss (unlike the DLM -- the ARLM naturally stops generating via its own
    stop token, so it never needs to learn to predict trailing padding). `pad_id`
    defaults to this project's own <PAD> (0); pass T5's own `pad_token_id` when
    decoding into T5's vocabulary instead (see t5_encoder.T5GraphEncoder)."""
    inp = target_ids[:, :-1]
    labels = target_ids[:, 1:]
    logits = model(inp, context, context_mask)
    ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=pad_id)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        real = labels != pad_id
        acc = ((preds == labels) & real).sum().float() / real.sum().clamp(min=1).float()
    return {"loss": ce, "token_accuracy": acc.item()}


@torch.no_grad()
def sample(model: GPTDecoder, context: torch.Tensor, context_mask: torch.Tensor,
           max_len: int | None = None, start_id: int = tok.P,
           eos_id: int = tok.EOS, pad_id: int = tok.PAD) -> torch.Tensor:
    """Greedy autoregressive decoding starting from `start_id`, stopping at the first
    `eos_id` per example (or at max_len), padding with `pad_id` elsewhere. No KV-cache
    -- recomputes the full forward pass each step; negligible cost at this scale
    (2 layers, hidden 128-512, L_tgt<=~55).

    Defaults (`tok.P`/`tok.EOS`/`tok.PAD`) match this project's own small vocab; when
    decoding into T5's vocabulary (t5_encoder.T5GraphEncoder), pass T5's own
    `pad_token_id` for both `start_id` and `pad_id` (the standard T5/seq2seq
    convention -- T5 has no dedicated BOS token, so its own decoder_start_token_id is
    conventionally pad_token_id too) and `eos_token_id` for `eos_id`."""
    B = context.shape[0]
    device = context.device
    L = max_len or model.l_tgt
    generated = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    generated[:, 0] = start_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for pos in range(1, L):
        logits = model(generated[:, :pos], context, context_mask)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        generated[:, pos] = next_tok
        finished = finished | (next_tok == eos_id)
        if bool(finished.all()):
            break
    return generated
