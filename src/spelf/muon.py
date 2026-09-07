"""A self-contained Muon optimizer (Newton-Schulz orthogonalized momentum for
2D parameters, bias-corrected Nesterov-Adam for everything else), following
the recipe ELF uses (`optax.contrib.muon`, ported to PyTorch on the
`pytorch_elf` branch as `utils/muon_utils.py`).

This is a from-scratch, single-device reimplementation rather than a vendored
copy of ELF's `muon_utils.py`, because that file wraps and monkey-patches an
external `muon` PyPI package plus `torch.distributed` all-gather logic for
multi-host training -- machinery this project has no use for at its scale.
The algorithmic core (5-step Newton-Schulz orthogonalization in fp32,
Nesterov momentum with bias correction, `sqrt(max(1, fan_out/fan_in))` shape
scaling, non-2D params routed to Nesterov-Adam) is preserved exactly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G via Newton-Schulz iteration (quintic, fp32)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _nesterov_adam_update(grad, mu, nu, step, betas, eps):
    b1, b2 = betas
    mu.lerp_(grad, 1 - b1)
    nu.lerp_(grad.square(), 1 - b2)
    mu_hat = b1 * (mu / (1 - b1 ** (step + 1))) + (1 - b1) * (grad / (1 - b1 ** step))
    nu_hat = nu / (1 - b2 ** step)
    return mu_hat / (nu_hat.sqrt() + eps)


def _muon_update(grad, momentum, step, beta=0.95, ns_steps=5, in_out_layout=False):
    momentum.lerp_(grad, 1 - beta)
    mu_hat = momentum / (1 - beta ** (step + 1))
    g_hat = grad / (1 - beta ** step)
    update = beta * mu_hat + (1 - beta) * g_hat
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    m, n = grad.size(-2), grad.size(-1)
    # nn.Linear.weight is stored (out, in) -> fan_out = m; a bare
    # (in, out)-convention Parameter (e.g. dlm.ELF.proj_kernel) has
    # fan_out = n. Flip the ratio accordingly.
    if in_out_layout:
        update = update * max(1, n / m) ** 0.5
    else:
        update = update * max(1, m / n) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Muon for 2D params + Nesterov-Adam for everything else, single-device.

    Construct via `muon_with_aux_adam(model, lr)` rather than directly, so
    the (out,in) vs (in,out) layout of every 2D parameter is detected
    automatically.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _muon_update(
                        p.grad, state["momentum_buffer"], state["step"],
                        beta=group["momentum"], in_out_layout=group["in_out_layout"].get(id(p), False),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _nesterov_adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


def muon_with_aux_adam(model: nn.Module, lr: float, weight_decay: float = 0.0,
                        muon_momentum: float = 0.95,
                        adam_betas=(0.9, 0.999), adam_eps: float = 1e-8) -> Muon:
    """Partition `model`'s trainable parameters: 2D -> Muon, else -> Adam.

    Hyperparameters default to `optax.contrib.muon`'s (matching ELF).
    """
    linear_weight_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear) and m.weight is not None}

    muon_params, adam_params = [], []
    in_out_layout: Dict[int, bool] = {}
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
            in_out_layout[id(p)] = id(p) not in linear_weight_ids
        else:
            adam_params.append(p)

    param_groups = [
        dict(params=muon_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay,
             use_muon=True, in_out_layout=in_out_layout),
        dict(params=adam_params, lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay,
             use_muon=False, in_out_layout={}),
    ]
    return Muon(param_groups, dict(lr=lr))
