"""Training state container, optimizer/LR-schedule construction, and EMA.

Ported from ELF's `pytorch_elf` branch (`utils/train_utils.py`), with the
DDP/`torch.compile` unwrap-chain kept (harmless if unused) and the
prefetch-thread helper dropped (our datasets are small enough that a plain
`DataLoader` iterator is not a bottleneck).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .common import Config
from .muon import muon_with_aux_adam


@dataclass
class TrainState:
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: Any = None
    ema_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    step: int = 0
    epoch: float = 0.0
    wandb_run_id: Optional[str] = None
    generator: Optional[torch.Generator] = None

    @staticmethod
    def init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.named_parameters()}


def unwrap_model(model: nn.Module) -> nn.Module:
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            break
    return model


@torch.no_grad()
def ema_update(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    inner = unwrap_model(model)
    for name, param in inner.named_parameters():
        if name in ema_state:
            ema_state[name].lerp_(param.detach(), 1.0 - decay)


def get_optimizer(model: nn.Module, config: Config, lr: float) -> Optimizer:
    if config.optimizer == "muon":
        return muon_with_aux_adam(model, lr=lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay,
                                  betas=(config.adam_b1, config.adam_b2))
    raise ValueError(f"Unknown optimizer: {config.optimizer!r}. Choose 'muon' or 'adamw'.")


def create_learning_rate_fn(num_train_steps: int, num_warmup_steps: int, learning_rate: float,
                             schedule: str = "constant", min_lr: float = 0.0):
    alpha = (min_lr / learning_rate) if learning_rate > 0 else 0.0

    def fn(step: int) -> float:
        step = int(step)
        if num_warmup_steps > 0 and step < num_warmup_steps:
            return learning_rate * step / max(1, num_warmup_steps)
        if schedule == "cosine":
            progress = (step - num_warmup_steps) / max(1, num_train_steps - num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return learning_rate * (alpha + (1.0 - alpha) * cosine)
        return learning_rate

    return fn


def attach_lr_scheduler(optimizer: Optimizer, lr_fn) -> LambdaLR:
    base_lr = optimizer.param_groups[0]["lr"]
    return LambdaLR(optimizer, lr_lambda=lambda step: lr_fn(step) / max(base_lr, 1e-12))
