"""The ELF diffusion-LM transformer: a stack of RoPE/QK-norm/SwiGLU blocks
operating in the frozen encoder's continuous embedding space, with a
flow-matching output head (`final_layer`) and a factored CE decoder head
(`proj_kernel`/`unembed_kernel`) sharing the same backbone. Ported from
ELF's `pytorch_elf` branch (`src/modules/model.py`) with two changes:

  1. Model sizes are rescaled way down (see `ELF_models` below) -- our
     vocabulary (~20 tokens) and sequences (<=~200 positions) are a small
     fraction of ELF's English/T5 setting.
  2. `patch_size` is dropped (always 1 for text; ELF only uses it for other
     modalities in the same codebase).

Everything else -- the two-branch (decoder CE / denoiser L2) forward,
prefix time/self-cond-cfg/model-mode tokens, self-conditioning input
doubling, RoPE with unrotated prefix positions -- matches the reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modules import (
    DEFAULT_BIAS_INIT, DEFAULT_KERNEL_INIT, NORMAL_INIT_002,
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder, _make_linear,
)


class ELFBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                               attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        x = x + self.attn(self.norm1(x), rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)
        return x


class ELF(nn.Module):
    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 64,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 4,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing

        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total)

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        self.final_layer = FinalLayer(hidden_size, out_channels=text_encoder_dim)

        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(self, t: torch.Tensor, self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = [self.t_emb_tokens.expand(B, -1, -1) + self.t_embedder(t).unsqueeze(1)]
        if self.num_self_cond_cfg_tokens > 0:
            # The RoPE table's `num_empty_token` prefix budget is fixed at
            # construction time assuming these tokens are always present
            # when num_self_cond_cfg_tokens > 0 -- so unlike the other
            # prefixes, this one isn't conditionally skippable per call.
            if self_cond_cfg_scale is None:
                raise ValueError(
                    "self_cond_cfg_scale must be provided on every forward call "
                    "when the model was built with num_self_cond_cfg_tokens > 0."
                )
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1))
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid."""
        B = x.shape[0]

        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)
        x = self.text_proj(x)
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                def _fwd(hidden, block=block):
                    return block(hidden, rope_fn=self.feat_rope, attention_mask=attention_mask, deterministic=deterministic)
                x = checkpoint(_fwd, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask, deterministic=deterministic)

        x = x[:, prefix_len + model_mode_offset:]

        decoder_logits = None
        if decoder_step_active is not None:
            hidden = F.gelu(x @ self.proj_kernel + self.proj_bias, approximate="tanh")
            decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias

        output = self.final_layer(x)
        return output, decoder_logits


# Model factory functions. See ARCHITECTURE.md ("Model sizing") for why
# these are so much smaller than ELF-B/M/L.
def ELF_XS(**kwargs): return ELF(depth=4, hidden_size=128, num_heads=4, **kwargs)
def ELF_S(**kwargs):  return ELF(depth=6, hidden_size=192, num_heads=6, **kwargs)
def ELF_M(**kwargs):  return ELF(depth=8, hidden_size=256, num_heads=8, **kwargs)

ELF_models = {"ELF-XS": ELF_XS, "ELF-S": ELF_S, "ELF-M": ELF_M}
