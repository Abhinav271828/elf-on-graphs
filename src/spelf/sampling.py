"""Flow-matching noise/time schedules, the ODE/SDE sampler, and the final
decode-to-tokens step. Ported from ELF's `pytorch_elf` branch
(`utils/sampling_utils.py` + `utils/generation_utils.py`, merged into one
module here since our task only ever needs the conditional path). The one
simplification versus the reference: a single `torch.Generator` tied to the
run's device is used everywhere for randomness, instead of branching on
`z.is_cuda` -- that branch existed to dodge a multi-host determinism corner
case this single-device project doesn't have.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .common import Config


# ============================================================
# Noise / time schedules
# ============================================================

def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, config: Config,
               cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale; condition
    positions are pinned to their clean value (never noised)."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


def _randn(shape, dtype, device, generator: Optional[torch.Generator]):
    """`torch.randn`, tolerant of `generator` living on a different device
    than `device` (a CUDA/MPS tensor call requires a same-device generator;
    we keep one CPU `torch.Generator` for the whole run for simplicity and
    checkpointability, so sample on its device then move)."""
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device)
    x = torch.randn(shape, dtype=dtype, generator=generator, device=generator.device)
    return x.to(device)


def _rand(shape, dtype, device, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.rand(shape, dtype=dtype, device=device)
    x = torch.rand(shape, dtype=dtype, generator=generator, device=generator.device)
    return x.to(device)


def sample_timesteps(batch_size: int, P_mean: float, P_std: float, time_schedule: str,
                      device=None, dtype=torch.float32, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if time_schedule == "logit_normal":
        z = _randn((batch_size,), dtype, device, generator) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return _rand((batch_size,), dtype, device, generator)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(n_steps: int, time_schedule: str, P_mean: float, P_std: float,
                        device=None, dtype=torch.float32, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Length-(n_steps+1) tensor of t in [0, 1] for one sampling rollout."""
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(n_steps - 1, P_mean, P_std, "logit_normal", device, dtype, generator)
        steps = torch.sort(steps).values
        lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([lo, steps, hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(batch_size: int, cfg_min: float, cfg_max: float,
                      dtype=torch.float32, device=None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Log-uniform CFG scale in [cfg_min, cfg_max] (used to train the
    self-cond-CFG conditioning tokens over a range of guidance strengths)."""
    u = _rand((batch_size,), dtype, device, generator)
    a, b = 1.0 + cfg_min, 1.0 + cfg_max
    log_ratio = torch.log(torch.tensor(b / a, dtype=dtype, device=u.device))
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================================
# Conditioning helpers
# ============================================================

def restore_cond(z_updated: torch.Tensor, cond_seq: torch.Tensor, cond_seq_mask: torch.Tensor) -> torch.Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor, t_eps: float = 5e-2):
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v, x


# ============================================================
# Forward passes with self-conditioning / CFG (train + sample time)
# ============================================================

def _forward_sample_self_cond(model, z, t_batch, x_pred_prev, config: Config,
                               self_cond_cfg_scale, cond_seq, cond_seq_mask):
    t_eps = config.t_eps

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        sc_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        net_out_cond = model(z_input_cond, t_batch, deterministic=True, self_cond_cfg_scale=sc_batch)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore(v_cond, x_cond)

    if config.self_cond_prob == 0:
        net_out = model(z, t_batch, deterministic=True)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore(v, x)

    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch, deterministic=True)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch, deterministic=True)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore(v_out, x_out)


def _forward_sample(model, z, t_batch, x_pred_prev, config: Config,
                     cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = None if x_pred_prev is None else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale, cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _ode_step(model, z, t, t_next, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(model, z, t_batch, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    return z + (t_next - t) * v_pred, x_pred


def _sde_step(model, z, t, t_next, x_pred_prev, config, cfg_scale, self_cond_cfg_scale,
              cond_seq, cond_seq_mask, gamma, generator):
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    eps = _randn(z.shape, z.dtype, z.device, generator) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(model, z_back, t_batch, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    return z_back + (t_next - t_back) * v_pred, x_pred


# ============================================================
# Full-rollout sampling + decode
# ============================================================

@torch.no_grad()
def generate_samples_single_batch(model: nn.Module, generator: torch.Generator, z: torch.Tensor,
                                   t_steps: torch.Tensor, cond_seq: Optional[torch.Tensor],
                                   cond_seq_mask: Optional[torch.Tensor], config: Config,
                                   cfg_scale: float, self_cond_cfg_scale: float,
                                   sampling_method: str = "ode", sde_gamma: float = 0.0) -> torch.Tensor:
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(model=model, config=config, cfg_scale=cfg_scale,
                        self_cond_cfg_scale=self_cond_cfg_scale, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    for i in range(n - 2):
        t, t_next = t_steps[i].item(), t_steps[i + 1].item()
        if sampling_method == "sde":
            z, x_pred = _sde_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, gamma=sde_gamma, generator=generator, **step_kwargs)
        elif sampling_method == "ode":
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
        else:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

    # Last step is always a plain ODE (Euler) step, matching ELF.
    t, t_next = t_steps[-2].item(), t_steps[-1].item()
    z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
    return z


@torch.no_grad()
def decode_batch(z: torch.Tensor, model: nn.Module, t_final_val: float, config: Config,
                  self_cond_cfg_scale: float) -> torch.Tensor:
    """Run the CE decoder head on the final latent -> token ids (argmax)."""
    batch_size = z.shape[0]
    t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
                if config.num_self_cond_cfg_tokens > 0 else None)
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    _, decoder_logits = model(z_input, t_final, deterministic=True, self_cond_cfg_scale=sc_batch, decoder_step_active=True)
    return decoder_logits.argmax(dim=-1)


def mask_after_eos(predicted_ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    eos_mask = predicted_ids == eos_id
    keep_mask = eos_mask.to(torch.int32).cumsum(dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0) -> torch.Tensor:
    """Shift each row left by its own amount along dim=1 (drops the
    condition prefix so only the generated target region remains)."""
    shift_per_sample = shift_per_sample.to(torch.long)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    shifted = torch.gather(x, 1, gather_idx)
    return torch.where(valid, shifted, torch.full_like(shifted, pad_value))
