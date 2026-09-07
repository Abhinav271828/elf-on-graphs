"""Shared configuration dataclass, YAML loading, and small runtime utilities.

This mirrors the two-layer config system used by ELF (`configs/config.py` +
YAML overrides), collapsed into a single flat dataclass since this project
has one model family and one task. See ARCHITECTURE.md ("Configuration")
for the rationale behind every default below.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

import numpy as np
import torch
import yaml


@dataclasses.dataclass
class Config:
    # ---- Graph generation (Erdos-Renyi) -----------------------------------
    # In-distribution (ID) graphs have `id_min_nodes`..`id_max_nodes` nodes.
    # Out-of-distribution (OOD) graphs have `ood_min_nodes`..`ood_max_nodes`
    # nodes. `ood_max_nodes` also defines the node-ID universe: every graph,
    # ID or OOD, draws its node labels from range(ood_max_nodes), so the
    # token vocabulary and every node-ID embedding is exercised during ID
    # training even though ID graphs are smaller. This isolates "OOD by
    # graph size" from "OOD by unseen vocabulary".
    id_min_nodes: int = 6
    id_max_nodes: int = 10
    ood_min_nodes: int = 11
    ood_max_nodes: int = 14
    # Edge probability p is drawn per-graph as a multiple of the connectivity
    # threshold ln(n)/n, then clipped to [edge_prob_floor, edge_prob_ceil].
    # This keeps diameters non-trivial (not always 1-2 hops) while making
    # G(n, p) connected on the first or second sample almost always.
    edge_prob_min_factor: float = 1.2
    edge_prob_max_factor: float = 2.5
    edge_prob_floor: float = 0.12
    edge_prob_ceil: float = 0.55
    max_connect_attempts: int = 200  # resample p,edges until G(n,p) is connected
    max_edges: int = 46  # reject/resample if the sampled graph exceeds this (bounds sequence length)

    n_train: int = 20000
    n_id_val: int = 500
    n_ood_test: int = 500
    data_seed: int = 0
    data_dir: str = "./data"

    # ---- Serialization / tokenizer -----------------------------------------
    # "nodes: <n0> <n1> ... edges: <u>-<v> ... find diametric path"
    # Condition = nodes + edges + prompt. Target = the path, as a
    # space-separated sequence of node-id tokens. Both live in the same
    # real T5 tokenizer (see tokenizer.py). Sized with margin above the
    # worst case at max_edges=46, max_nodes=14 (~222 condition tokens,
    # ~16 target tokens under T5's tokenizer -- see ARCHITECTURE.md).
    max_input_length: int = 240   # token budget for the condition (prompt side)
    max_length: int = 264         # token budget for condition + target combined
    pad_token: str = "pad"        # "pad" (dedicated PAD id) or "eos"

    # ---- Encoder (real pretrained T5, frozen -- exactly as ELF uses one) ---
    encoder_model_name: str = "t5-small"
    tokenizer_name: Optional[str] = None   # defaults to encoder_model_name if unset
    encoder_profile_path: str = "./runs/t5_profile.json"
    latent_stats_sample_size: int = 4000   # examples used to compute latent_mean/std

    # latent_mean / latent_std normalize frozen-encoder outputs before they
    # enter the diffusion model (as in ELF's `encode_text`). They are
    # computed once, over the actual training collate pipeline, by
    # scripts/prepare_encoder.py, and cached to encoder_profile_path;
    # 0.0 / 1.0 here are placeholders overwritten at load time.
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # ---- ELF diffusion transformer ------------------------------------------
    # "ELF-XS": far smaller than ELF-B (105M) -- our sequences are <=~260
    # tokens vs. document-length OWT sequences (the vocabulary is the same
    # T5 vocab either way, ~32k tokens, since the encoder is real T5).
    # Depth/width are cut roughly 6-8x while keeping the exact same block
    # design (RoPE + QK-norm attention, SwiGLU, RMSNorm, prefix
    # time/self-cond-cfg/model-mode tokens); the CE decoder head keeps ELF's
    # full-vocab unembedding, so it (and the bottleneck text projection,
    # sized off the encoder's real d_model=512) dominate ELF-XS's parameter
    # count -- the backbone itself is tiny.
    model: str = "ELF-XS"
    bottleneck_dim: int = 64
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    gradient_checkpointing: bool = False

    # ---- Denoiser (flow-matching) objective ---------------------------------
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # "logit_normal" or "uniform"

    # ---- Decoder (CE) objective ----------------------------------------------
    decoder_prob: float = 0.5
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    # ---- Conditioning / CFG --------------------------------------------------
    label_drop_prob: float = 0.1   # >0 so classifier-free guidance is meaningful at sampling
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # ---- Optimization ---------------------------------------------------------
    epochs: int = 60
    warmup_steps: int = 300
    batch_size: int = 128
    lr: Optional[float] = None
    blr: float = 1e-3          # base lr; effective lr = blr * batch_size / 256 (matches ELF)
    min_lr: float = 0.0
    lr_schedule: str = "cosine"
    weight_decay: float = 0.0
    optimizer: str = "muon"    # "muon" or "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1
    ema_decay1: float = 0.999
    use_bf16: bool = True      # only takes effect on CUDA

    # ---- Sampling (used for periodic + standalone eval) -----------------------
    sampling_method: str = "ode"       # "ode" or "sde"
    num_sampling_steps: int = 32
    cfg_scale: float = 2.0
    self_cond_cfg_scale: float = 1.0
    sampling_time_schedule: str = "logit_normal"
    sde_gamma: float = 0.0

    # ---- Evaluation -------------------------------------------------------------
    eval_num_examples: int = 200   # examples per split (ID / OOD) used for metrics; -1 = full split
    eval_num_viz: int = 5          # graph visualizations logged to wandb per eval, per split

    # ---- Logging & checkpointing --------------------------------------------
    log_freq: int = 50
    eval_freq: int = 2     # epochs
    save_freq: float = 1   # epochs; may be fractional (e.g. 0.5)
    output_dir: str = "./runs/dlm"
    resume: Optional[str] = None

    # ---- Weights & Biases -----------------------------------------------------
    use_wandb: bool = True
    wandb_project: str = "spelf-diametric-path"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_mode: str = "online"   # "online", "offline", or "disabled"

    # ---- Misc -------------------------------------------------------------------
    seed: int = 0
    device: Optional[str] = None
    num_workers: int = 0


def load_config(path: Optional[str]) -> Config:
    """Load a YAML file and overlay it onto `Config` defaults."""
    config = Config()
    if not path:
        return config
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    for key, value in raw.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key!r}")
        setattr(config, key, value)
    return config


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply `field=value` CLI overrides, using the field's current type to coerce `value`."""
    field_types = {f.name: f.type for f in dataclasses.fields(config)}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected field=value")
        key, raw_value = item.split("=", 1)
        key, raw_value = key.strip(), raw_value.strip()
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key!r}")
        if raw_value.lower() == "none":
            setattr(config, key, None)
            continue
        current = getattr(config, key)
        py_type = type(current) if current is not None else field_types.get(key)
        if py_type is bool:
            value: object = raw_value.lower() in ("1", "true", "yes")
        elif py_type is int:
            value = int(raw_value)
        elif py_type is float:
            value = float(raw_value)
        else:
            value = raw_value
        setattr(config, key, value)
    return config


def save_config(config: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(config: Config) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
