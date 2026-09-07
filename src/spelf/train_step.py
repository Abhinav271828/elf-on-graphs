"""One mini-batch forward/backward for the ELF diffusion LM.

Ported from ELF's `pytorch_elf` branch (`src/train_step.py`). As in the
reference: each example in the batch independently draws decoder (CE) vs.
denoiser (L2) mode via a per-example Bernoulli at `decoder_prob`; one forward
pass computes both heads on a mixed input, and the two losses are masked to
their respective rows and combined with a single denominator. Self-cond +
CFG guidance targets are computed from a shared no-grad "uncond" forward.

Dropped versus the reference: bf16 autocast (this project targets CPU/MPS
scale, where it's a no-op) and DDP's `no_sync()` context (single process).
Everything else -- the flow-matching v-target, the label-drop block-mask
trick, self-cond-CFG guided target construction -- matches exactly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Config
from .sampling import _randn, add_noise, net_out_to_v_x, restore_cond, sample_cfg_scale, sample_timesteps
from .t5_encoder import encode_text
from .train_state import TrainState, ema_update


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def train_step(state: TrainState, encoder: nn.Module, batch: Dict[str, torch.Tensor],
                config: Config, device: torch.device) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    dtype = next(state.model.parameters()).dtype
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale
    gen = state.generator

    input_ids = batch["input_ids"].to(device).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32)

    batch_size = input_ids.shape[0]
    label_drop_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    if config.label_drop_prob > 0:
        u = torch.rand((batch_size,), generator=gen).to(device)
        label_drop_mask = u < config.label_drop_prob

    if config.label_drop_prob > 0:
        drop = label_drop_mask.to(torch.float32).reshape(-1, 1, 1)
        block_mask = (1 - cond_seq_mask).unsqueeze(-1) * cond_seq_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=input_ids, attention_mask=encoder_attention_mask, encoder=encoder,
        latent_mean=config.latent_mean, latent_std=config.latent_std,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(batch_size, config.denoiser_p_mean, config.denoiser_p_std,
                          config.time_schedule, device=device, dtype=dtype, generator=gen)
    noise = _randn(x0.shape, dtype, device, gen)

    loss_mask = attention_mask if config.pad_token == "pad" else torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)
    cond_seq_mask_3d = cond_seq_mask.unsqueeze(-1)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask_3d)

    drop_3d = label_drop_mask.view(-1, 1, 1) & (cond_seq_mask_3d > 0)
    if config.label_drop_prob > 0:
        denoiser_z = torch.where(drop_3d, torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop_3d, torch.zeros_like(x0), x0)

    decoder_targets = input_ids

    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), decoder_prob, dtype=torch.float32), generator=gen,
    ).to(device=device, dtype=dtype)
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    decoder_z_vals = (torch.randn((batch_size * seq_length,), dtype=dtype, generator=gen).to(device)
                       * config.decoder_p_std + config.decoder_p_mean)
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = _randn(x0.shape, dtype, device, gen) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)

    use_self_cond_mask = None
    if self_cond_prob > 0:
        use_self_cond_mask = (torch.rand((batch_size,), generator=gen).to(device) < self_cond_prob).reshape(-1, 1, 1).to(dtype)

    self_cond_cfg_scale = None
    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(batch_size, config.self_cond_cfg_min, config.self_cond_cfg_max,
                                                dtype=dtype, device=device, generator=gen)

    model = state.model

    def compute_shared_uncond(z, t_input, x_tokens):
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask_3d)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad():
            return model(z_input_uncond, t_input, deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale)

    def get_sc_cond_and_uncond(z, t_input, x_tokens, shared_net_out_uncond):
        if config.self_cond_prob == 0:
            with torch.no_grad():
                net_out_uncond = model(z, t_input, deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale)
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond

        v_uncond, x_uncond = net_out_to_v_x(shared_net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_seq_mask_3d)
        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        with torch.no_grad():
            net_out_cond = model(z_input_cond, t_input, deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale)
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_v_target(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        if not (config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0):
            return base_v_target
        v_cond, v_uncond = get_sc_cond_and_uncond(z, t_input, x_tokens, shared_net_out_uncond)
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        return (base_v_target + sc_guidance).detach()

    model.train()

    denoiser_t = t
    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z

    shared_net_out_uncond = None
    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(denoiser_z, denoiser_t, x0)

    if config.self_cond_prob > 0:
        _, x_pred_init = net_out_to_v_x(shared_net_out_uncond, denoiser_z, denoiser_t, t_eps)
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask_3d)
        x_pred_cond = restore_cond(x_pred_init * use_self_cond_mask.to(dtype), x0, cond_seq_mask_3d)
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
    else:
        model_input = z_mixed

    net_out, decoder_logits = model(
        model_input, t_mixed, deterministic=False,
        self_cond_cfg_scale=self_cond_cfg_scale, decoder_step_active=decoder_step_active,
    )

    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
                                   shared_net_out_uncond=shared_net_out_uncond)
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)

    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    ce_loss_val = ((ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss_val = ((l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    accum_steps = max(config.grad_accum_steps, 1)
    state.step += 1
    is_optimizer_step = (state.step % accum_steps) == 0

    (loss / accum_steps).backward()

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        ema_update(state.ema_params, state.model, config.ema_decay1)
        state.optimizer.zero_grad(set_to_none=True)

    metrics = {"loss": loss.detach(), "l2_loss": l2_loss_val, "ce_loss": ce_loss_val}
    return state, metrics
