"""Shared training infrastructure: seeding, device selection, optimizer/LR-schedule
construction, checkpoint save/load/resume, early stopping, and a wandb init helper.
Reused by pretrain_encoder.py / train_dlm.py / train_arlm.py -- each script's training
loop stays explicit and readable; only this low-level plumbing is shared."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .muon import Muon


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def trainable_parameters(modules: torch.nn.Module | list[torch.nn.Module]) -> list[tuple[str, torch.nn.Parameter]]:
    """(name, param) pairs with requires_grad=True across one or more modules, deduped
    by identity -- e.g. decoder.embedding and encoder.embedding are literally the same
    shared object in both the custom-encoder and T5 setups (see t5_encoder.py), so
    naively concatenating named_parameters() across [decoder, encoder] would double-list
    (and, worse, double-clip/double-step) it."""
    if isinstance(modules, nn.Module):
        modules = [modules]
    seen: set[int] = set()
    out = []
    for module in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            out.append((name, p))
    return out


def build_optimizer(modules: torch.nn.Module | list[torch.nn.Module], lr: float,
                     weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """AdamW with the conventional exclusion of embeddings/LayerNorm/bias/1-D params
    from weight decay. This also makes SharedEmbedding.freeze_pretrained_rows() (see
    modules.py) behave as a true freeze: without this, AdamW's weight decay would keep
    shrinking the "frozen" rows even though their gradient is zeroed.

    `modules` may be a single module (the usual case) or a list -- e.g. [decoder,
    t5_encoder] when the conditioning encoder itself has trainable parameters. Use the
    same `modules` argument with `trainable_parameters()` when grad-clipping, so the
    clipped set matches exactly what's being optimized."""
    decay, no_decay = [], []
    for name, p in trainable_parameters(modules):
        if p.ndim <= 1 or "embedding" in name.lower() or "norm" in name.lower() or "null_context" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def _is_muon_eligible(name: str, p: torch.nn.Parameter) -> bool:
    """True for a transformer's own >=2D "hidden" weight matrices -- the only
    parameters Muon (muon.py) is designed to optimize. Everything else is routed to a
    conventional AdamW group instead: any 1D parameter (biases, LayerNorm weight/bias),
    any embedding table (`SharedEmbedding`'s tied token embedding, `LearnedPosEnc`'s
    positional embedding, the DLM's per-branch `mode_emb`), and the DLM's
    `null_context` vector (shape (1,1,d_model) -- a single learned bias-like vector,
    not a weight matrix, even though its ndim happens to be >=2)."""
    if p.ndim < 2:
        return False
    lname = name.lower()
    if "embedding" in lname or "pos_emb" in lname or "mode_emb" in lname or "null_context" in lname:
        return False
    return True


class MultiOptimizer:
    """Duck-types just enough of torch.optim.Optimizer's interface (step, zero_grad,
    state_dict, load_state_dict) to drop into save_checkpoint/load_checkpoint
    unchanged, while actually driving more than one underlying optimizer -- e.g. Muon
    over a DLM's hidden weight matrices plus AdamW over everything else, see
    build_dlm_optimizer -- as a single unit."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    def step(self, closure=None) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state: dict) -> None:
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state[name])


def build_dlm_optimizer(
    modules: torch.nn.Module | list[torch.nn.Module],
    adamw_lr: float,
    adamw_weight_decay: float = 0.01,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
) -> MultiOptimizer:
    """DLM-only optimizer construction matching the canonical ELF implementation
    (arXiv:2605.10938, Section 4): Muon over the decoder's (and, if --encoder_kind t5,
    the T5 projection's) own >=2D hidden weight matrices, AdamW over everything else --
    embeddings, positional/mode embeddings, null_context, LayerNorm, biases -- see
    _is_muon_eligible for the exact split and build_optimizer's docstring for why
    embeddings/norms/biases are excluded from AdamW weight decay too. Returns a
    MultiOptimizer wrapping both underlying optimizers.

    Deduplicates by parameter identity via trainable_parameters, same as
    build_optimizer, so a parameter shared across modules (e.g. the embedding table,
    shared between encoder and decoder) is never assigned to two groups.

    pretrain_encoder.py and train_arlm.py are untouched by this function -- they
    still call the plain AdamW-only build_optimizer above; only train_dlm.py's
    --optimizer muon path (the default) uses this."""
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for name, p in trainable_parameters(modules):
        if _is_muon_eligible(name, p):
            muon_params.append(p)
        elif p.ndim <= 1 or "embedding" in name.lower() or "norm" in name.lower() or "null_context" in name.lower():
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    optimizers: dict[str, torch.optim.Optimizer] = {}
    if muon_params:
        optimizers["muon"] = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                                   weight_decay=muon_weight_decay)
    adamw_groups = [
        {"params": adamw_decay, "weight_decay": adamw_weight_decay},
        {"params": adamw_no_decay, "weight_decay": 0.0},
    ]
    optimizers["adamw"] = torch.optim.AdamW(adamw_groups, lr=adamw_lr)
    return MultiOptimizer(optimizers)


def encode_context(encoder: torch.nn.Module, input_ids: torch.Tensor,
                    input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform (context, context_mask) call across conditioning-encoder backends.

    GraphEncoder returns a single tensor whose sequence is length-aligned with
    input_ids, so context_mask is just input_mask -- and per GraphEncoder.freeze()'s own
    docstring, it must always be called under no_grad here even though its embedding
    table's <P>/<EOS> rows keep requires_grad=True for use elsewhere (they should never
    accumulate gradient via this conditioning path, only via the decoder's own target
    embedding lookups).

    T5GraphEncoder/T5DiffusionEncoder re-tokenize the graph as text, so their output has
    its own length/mask and returns both directly as a tuple. Both leave
    `context_requires_grad` at its default (False, wrapped in no_grad here) since
    neither has any trainable parameter upstream of its own returned context anymore
    (T5GraphEncoder's only trainable piece, `embedding`, is downstream -- consumed by
    the decoder's own target-embedding lookups, not produced by this forward pass;
    T5DiffusionEncoder has no trainable parameters at all). `T5GraphEncoder.forward`
    still manages its own internal no_grad scope around the frozen T5 stack regardless,
    for clarity at the call site rather than relying solely on this wrapper."""
    if getattr(encoder, "context_requires_grad", False):
        out = encoder(input_ids, input_mask)
    else:
        with torch.no_grad():
            out = encoder(input_ids, input_mask)
    if isinstance(out, tuple):
        return out
    return out, input_mask


def build_lr_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup -> cosine decay to 0, as a LambdaLR multiplier on the base lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class MultiScheduler:
    """Same duck-typing idea as MultiOptimizer (step, state_dict, load_state_dict,
    get_last_lr), for a matching set of per-optimizer LR schedulers -- see
    build_multi_lr_schedule."""

    def __init__(self, schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler]):
        self.schedulers = schedulers

    def step(self) -> None:
        for s in self.schedulers.values():
            s.step()

    def state_dict(self) -> dict:
        return {name: s.state_dict() for name, s in self.schedulers.items()}

    def load_state_dict(self, state: dict) -> None:
        for name, s in self.schedulers.items():
            s.load_state_dict(state[name])

    def get_last_lr(self) -> dict[str, list[float]]:
        return {name: s.get_last_lr() for name, s in self.schedulers.items()}


def build_multi_lr_schedule(optimizer: MultiOptimizer, warmup_steps: int, total_steps: int) -> MultiScheduler:
    """One independent linear-warmup -> cosine-decay LambdaLR per underlying optimizer
    in `optimizer` (see build_dlm_optimizer) -- same warmup/total-step shape for both,
    each keyed to its own optimizer's own peak lr (Muon's lr is typically an order of
    magnitude higher than AdamW's, see train_dlm.py's --muon_lr default)."""
    return MultiScheduler({name: build_lr_schedule(opt, warmup_steps, total_steps)
                            for name, opt in optimizer.optimizers.items()})


class EarlyStopper:
    """Tracks a metric (default: higher-is-better) across eval rounds; `step()` returns
    True once it has failed to improve by more than `tolerance` for `patience`
    consecutive rounds."""

    def __init__(self, tolerance: float = 0.005, patience: int = 5, mode: str = "max"):
        self.tolerance = tolerance
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad = 0

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            self.num_bad = 0
            return False
        improved = (metric > self.best + self.tolerance) if self.mode == "max" else (metric < self.best - self.tolerance)
        if improved:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "num_bad": self.num_bad}

    def load_state_dict(self, d: dict) -> None:
        self.best = d["best"]
        self.num_bad = d["num_bad"]


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor; torch.load's map_location can
    # relocate it onto the model's device (e.g. "mps") along with everything else, so
    # force it back to CPU here regardless of where the checkpoint was loaded onto.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_mps") is not None and torch.backends.mps.is_available():
        try:
            torch.mps.set_rng_state(state["torch_mps"])
        except Exception:
            pass


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: "Optional[torch.optim.Optimizer | MultiOptimizer]",
    scheduler=None,
    extra_state: Optional[dict] = None,
    config: Optional[dict] = None,
    tag: str = "latest",
    also_snapshot: bool = False,
    keep_last_k: int = 3,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "extra_state": extra_state or {},
        "config": config or {},
        "rng_state": _rng_state(),
    }
    path = run_dir / f"checkpoint_{tag}.pt"
    torch.save(state, path)
    if also_snapshot:
        torch.save(state, run_dir / f"checkpoint_step_{step}.pt")
        _prune_old_snapshots(run_dir, keep_last_k)
    return path


def _prune_old_snapshots(run_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    snaps = sorted(run_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    for p in snaps[:-keep_last_k]:
        p.unlink()


def check_checkpoint_config(path: str | Path, expected: dict) -> dict:
    """Peek at a checkpoint's saved config before fully resuming from it, and assert
    that the given keys match. Without this, e.g. resuming a --run_dir whose checkpoint
    was trained with a different --encoder_kind surfaces later as a cryptic
    optimizer.load_state_dict "parameter group doesn't match" error, since the two
    encoder kinds have different trainable-parameter sets."""
    cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
    for key, want in expected.items():
        got = cfg.get(key)
        assert got == want, (
            f"checkpoint {path} was saved with {key}={got!r}, but {key}={want!r} was "
            f"requested -- use a different --run_dir or --resume none to start fresh there"
        )
    return cfg


def resolve_checkpoint_path(run_dir: str | Path, resume: Optional[str]) -> Optional[Path]:
    if not resume or resume.lower() == "none":
        return None
    run_dir = Path(run_dir)
    if resume == "latest":
        p = run_dir / "checkpoint_latest.pt"
    elif resume == "best":
        p = run_dir / "checkpoint_best.pt"
    else:
        p = Path(resume)
    return p if p.exists() else None


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: "Optional[torch.optim.Optimizer | MultiOptimizer]" = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if restore_rng and state.get("rng_state"):
        _restore_rng_state(state["rng_state"])
    return state


def wandb_init(project: str, run_name: str, group: str, config: dict, mode: str = "online",
                run_id: Optional[str] = None):
    """`run_id` should be the previous run's `run.id` (see save_checkpoint's
    `extra_state["wandb_run_id"]` convention in train_dlm.py/train_arlm.py/
    pretrain_encoder.py), read back from a resumed checkpoint -- this reattaches to that
    same wandb run instead of starting a new one, so a locally-resumed run continues its
    existing wandb history rather than forking a fresh, empty-looking run at the same
    step. `resume="allow"` reattaches if that run id still exists, or falls back to
    creating a fresh run under that id if it doesn't (e.g. deleted on the wandb side)."""
    import wandb
    kwargs = dict(project=project, name=run_name, group=group, config=config, mode=mode)
    if run_id is not None:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    return wandb.init(**kwargs)
