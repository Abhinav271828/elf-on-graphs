"""Muon (MomentUm Orthogonalized by Newton-Schulz), the optimizer the canonical ELF
implementation uses for its decoder (arXiv:2605.10938, Section 4: "Muon optimizer with
learning rate 0.002"). Verified against the paper before wiring in -- see the
conversation that added this file for the exact check.

Muon is designed to optimize only a transformer's own >=2D "hidden" weight matrices
(attention projections, MLP linears); embeddings, unembedding/classifier heads, and any
1D parameter (biases, LayerNorm weight/bias) are conventionally left to a plain AdamW
group instead -- Muon's orthogonalized-update semantics assume the parameter is a linear
map between two continuous spaces, which doesn't fit an embedding table's per-row
lookup semantics. `common.build_dlm_optimizer` is what actually splits a DLM's
parameters between this optimizer and AdamW; this module only implements the
optimizer itself, with no knowledge of any particular model's parameter names.
"""
from __future__ import annotations

import torch


def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2D matrix via a quintic Newton-Schulz iteration:
    starting from G (rescaled to unit Frobenius norm), repeatedly apply a fixed
    quintic polynomial map (coefficients a,b,c tuned by Jordan et al. so the iteration
    converges to an orthogonal matrix sharing G's singular vectors within a handful of
    steps -- 5 is the standard choice, trading a closer-to-exact orthogonalization
    against per-optimizer-step cost). Runs in bfloat16 purely for speed; the result is
    cast back to G's own dtype before being returned. Operates on whichever of
    (rows, cols) is smaller (transposing back afterward if needed) since the
    per-iteration matmuls scale with the larger dimension otherwise."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """One Muon step: Nesterov momentum accumulation on the raw gradient, then the
    momentum-adjusted update is orthogonalized via `_newton_schulz5` before being
    applied -- rather than taking a step in the raw (co-)gradient direction, this takes
    a step of the same shape but with all singular values pushed toward 1, so no single
    direction in weight-space is over- or under-updated relative to the others. The
    orthogonalized update is then rescaled by `max(1, rows/cols)**0.5`, an empirical
    calibration (Jordan et al.) so one `lr` behaves reasonably across weight matrices of
    different shapes -- without it, a wide matrix's orthogonalized update would have a
    different effective RMS size than a tall one's. Weight decay is decoupled (applied
    to the parameter directly, as in AdamW), not folded into the gradient.

    Every parameter passed to this optimizer must be >=2D -- see this class's module
    docstring for why 1D/embedding parameters belong in a separate (AdamW) group
    instead; passing one here raises immediately at construction rather than failing
    obscurely inside `_newton_schulz5` partway through training.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                         nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon only supports >=2D parameters, got shape {tuple(p.shape)} "
                        f"-- route 1D/embedding parameters to a separate AdamW group instead "
                        f"(see common.build_dlm_optimizer)"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:
                    g = g.reshape(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = _newton_schulz5(update, steps=group["ns_steps"])
                update = update * max(1.0, update.size(0) / update.size(1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update.view_as(p), alpha=-lr)
        return loss
