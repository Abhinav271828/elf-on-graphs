"""ELF-style DLM decoder: continuous-embedding rectified-flow matching with x-prediction,
two-branch denoise/decode training, self-conditioning, and CFG-dropout conditioning.
Adapted from arXiv:2605.10938 ("Embedded Language Flows") to condition on a frozen
graph encoder instead of a frozen T5 encoder, and to generate the fixed-length PATH
target non-autoregressively (all L_TGT positions jointly) instead of free-form text.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as tok
from .modules import DecoderLayer, LearnedPosEnc, SharedEmbedding, TimeEmbedding

DENOISE_MODE = 0
DECODE_MODE = 1


class DLMDecoder(nn.Module):
    def __init__(self, l_tgt: int = tok.TARGET_LENGTH, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 8, d_mlp: int = 512, dropout: float = 0.1,
                 embedding: SharedEmbedding | None = None):
        super().__init__()
        self.l_tgt = l_tgt
        self.d_model = d_model
        self.embedding = embedding if embedding is not None else SharedEmbedding(d_model=d_model)
        self.pos_enc = LearnedPosEnc(l_tgt, d_model)
        self.time_emb = TimeEmbedding(d_model)
        self.mode_emb = nn.Embedding(2, d_model)
        self.input_proj = nn.Linear(2 * d_model, d_model)  # concat(z_t, self_cond) -> d_model
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_mlp, dropout, causal=False) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.null_context = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_context, std=0.02)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor, mode: torch.Tensor,
                self_cond: torch.Tensor | None = None) -> torch.Tensor:
        # z_t: [B, L_tgt, D]  t: [B]  context: [B, L_in, D]  context_mask: [B, L_in]
        # mode: [B] long in {DENOISE_MODE, DECODE_MODE}
        if self_cond is None:
            self_cond = torch.zeros_like(z_t)
        h = self.input_proj(torch.cat([z_t, self_cond], dim=-1))
        h = self.pos_enc(h)
        h = h + self.time_emb(t)[:, None, :]
        h = h + self.mode_emb(mode)[:, None, :]
        for layer in self.layers:
            h = layer(h, context, context_mask)
        h = self.out_ln(h)
        return self.out_proj(h)  # x_hat, in embedding space

    @torch.no_grad()
    def generate(self, context: torch.Tensor, context_mask: torch.Tensor,
                 num_steps: int = 32, guidance_scale: float = 1.0,
                 use_self_cond: bool = False) -> torch.Tensor:
        return sample(self, context, context_mask, num_steps=num_steps,
                       guidance_scale=guidance_scale, use_self_cond=use_self_cond)


def _apply_context_dropout(context: torch.Tensor, null_context: nn.Parameter,
                            drop_prob: float) -> torch.Tensor:
    B = context.shape[0]
    drop_mask = torch.rand(B, device=context.device) < drop_prob
    null = null_context.expand(B, context.shape[1], context.shape[2])
    return torch.where(drop_mask[:, None, None], null, context)


def loss(model: DLMDecoder, context: torch.Tensor, context_mask: torch.Tensor,
          target_ids: torch.Tensor, cfg_dropout_prob: float = 0.1,
          decode_branch_prob: float = 0.2, selfcond_prob: float = 0.0,
          lambda_ce: float = 1.0, eps: float = 1e-5) -> dict:
    """One training step's loss. Per-example branch assignment (denoise vs decode) is
    sampled once per batch; both branches run through a single shared forward pass
    (mode varies per example), then each example's loss term is routed to the matching
    objective (reweighted MSE for denoise, cross-entropy for decode)."""
    B, L = target_ids.shape
    device = target_ids.device

    context = _apply_context_dropout(context, model.null_context, cfg_dropout_prob)

    x = model.embedding(target_ids)  # [B, L, D] clean embeddings, the diffusion target

    is_denoise = torch.rand(B, device=device) < (1 - decode_branch_prob)
    t_denoise = torch.rand(B, device=device) * (1 - eps)
    t_decode = 0.5 + torch.rand(B, device=device) * 0.5
    t = torch.where(is_denoise, t_denoise, t_decode)
    mode = (~is_denoise).long()  # DENOISE_MODE=0, DECODE_MODE=1

    eps_noise = torch.randn_like(x)
    z_t = t[:, None, None] * x + (1 - t[:, None, None]) * eps_noise

    self_cond = torch.zeros_like(x)
    if selfcond_prob > 0:
        do_selfcond = torch.rand(B, device=device) < selfcond_prob
        with torch.no_grad():
            x_hat_prime = model(z_t, t, context, context_mask, mode, self_cond=torch.zeros_like(x))
        self_cond = torch.where(do_selfcond[:, None, None], x_hat_prime, self_cond)

    x_hat = model(z_t, t, context, context_mask, mode, self_cond=self_cond)

    sq_err = (x_hat - x) ** 2  # [B, L, D]
    weight = (1.0 / ((1 - t) ** 2 + eps))[:, None, None]
    denoise_terms = (weight * sq_err).mean(dim=(1, 2))  # [B]

    logits = model.embedding.unembed(x_hat)  # [B, L, vocab]
    ce = F.cross_entropy(logits.transpose(1, 2), target_ids, reduction="none")  # [B, L]
    decode_terms = ce.mean(dim=1)  # [B]

    denoise_mask = is_denoise.float()
    decode_mask = (~is_denoise).float()
    denoise_loss = (denoise_terms * denoise_mask).sum() / denoise_mask.sum().clamp(min=1)
    decode_loss = (decode_terms * decode_mask).sum() / decode_mask.sum().clamp(min=1)
    total = denoise_loss + lambda_ce * decode_loss

    return {
        "loss": total,
        "denoise_loss": denoise_loss.detach(),
        "decode_loss": decode_loss.detach(),
        "n_denoise": int(is_denoise.sum().item()),
        "n_decode": int((~is_denoise).sum().item()),
    }


@torch.no_grad()
def sample(model: DLMDecoder, context: torch.Tensor, context_mask: torch.Tensor,
           num_steps: int = 32, guidance_scale: float = 1.0, use_self_cond: bool = False,
           eps: float = 1e-5) -> torch.Tensor:
    """z_0 ~ N(0,I) over [L_tgt, D]; Euler-integrate dz/dt = v_theta for num_steps;
    final decode-mode forward + argmax to get tokens. guidance_scale=1.0 disables CFG
    (default, since generation here is always meant to be graph-conditioned, unlike the
    paper's unconditional-generation use case) but is wired for sweeps.

    use_self_cond should match whether the model was *trained* with selfcond_prob > 0
    (see loss()) -- a model trained with self_cond always zeroed (vanilla, the default)
    never learned to make use of a nonzero self-conditioning input, so feeding it one at
    sampling time would just be off-distribution noise through input_proj, not a genuine
    ablation-preserving no-op. When False (default), self_cond stays zero for every
    step, matching vanilla training; when True, each step's prediction is chained into
    the next step's self_cond, as in the ELF paper."""
    B = context.shape[0]
    device = context.device
    z = torch.randn(B, model.l_tgt, model.d_model, device=device)
    self_cond = torch.zeros(B, model.l_tgt, model.d_model, device=device)
    mode_denoise = torch.full((B,), DENOISE_MODE, dtype=torch.long, device=device)
    dt = 1.0 / num_steps
    null = model.null_context.expand(B, context.shape[1], context.shape[2])

    for step in range(num_steps):
        t = torch.full((B,), step * dt, device=device)
        x_hat_cond = model(z, t, context, context_mask, mode_denoise, self_cond=self_cond)
        if guidance_scale != 1.0:
            x_hat_uncond = model(z, t, null, context_mask, mode_denoise, self_cond=self_cond)
            x_hat = guidance_scale * x_hat_cond + (1 - guidance_scale) * x_hat_uncond
        else:
            x_hat = x_hat_cond
        v = (x_hat - z) / (1 - step * dt + eps)
        z = z + v * dt
        if use_self_cond:
            self_cond = x_hat_cond

    t_final = torch.full((B,), 1.0 - eps, device=device)
    mode_decode = torch.full((B,), DECODE_MODE, dtype=torch.long, device=device)
    x_hat_final = model(z, t_final, context, context_mask, mode_decode, self_cond=self_cond)
    logits = model.embedding.unembed(x_hat_final)
    return logits.argmax(dim=-1)
