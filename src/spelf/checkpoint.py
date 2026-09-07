"""Checkpoint save/load/discovery, and the wandb-run-id <-> checkpoint sync.

Design: the wandb run id is stored *inside* every checkpoint. On resume, we
read that id back out of the checkpoint before calling `wandb.init`, and
pass it as `id=..., resume="allow"` -- so resuming a run from its checkpoint
directory automatically reattaches to the same wandb run, with no separate
"run name" bookkeeping for the user to keep in sync by hand (contrast with
ELF's PyTorch reference, which requires the user to pass the same
`wandb_run_name` on every resume). A fresh run (no checkpoint found) mints a
new id via `wandb.util.generate_id()` and it rides along in every checkpoint
from then on.

Checkpoints are named `checkpoint_<step>.pt` under `config.output_dir`; only
the `keep_last` most recent are retained.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import torch

from .train_state import TrainState, unwrap_model


def find_all_checkpoints(output_dir: str) -> list:
    if not os.path.isdir(output_dir):
        return []
    paths = glob.glob(os.path.join(output_dir, "checkpoint_*.pt"))

    def step_of(p: str) -> int:
        m = re.search(r"checkpoint_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=step_of)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    all_ckpts = find_all_checkpoints(output_dir)
    return all_ckpts[-1] if all_ckpts else None


def peek_wandb_run_id(checkpoint_path: str) -> Optional[str]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return payload.get("wandb_run_id")


def save_checkpoint(state: TrainState, output_dir: str, step: int, keep_last: int = 3) -> str:
    os.makedirs(output_dir, exist_ok=True)
    inner_model = unwrap_model(state.model)
    payload = {
        "model": inner_model.state_dict(),
        "ema_params": state.ema_params,
        "optimizer": state.optimizer.state_dict(),
        "lr_scheduler": state.lr_scheduler.state_dict() if state.lr_scheduler is not None else None,
        "step": int(state.step),
        "epoch": float(state.epoch),
        "generator_state": state.generator.get_state() if state.generator is not None else None,
        "wandb_run_id": state.wandb_run_id,
    }
    path = os.path.join(output_dir, f"checkpoint_{step}.pt")
    torch.save(payload, path)

    all_ckpts = find_all_checkpoints(output_dir)
    for stale in all_ckpts[:-keep_last]:
        try:
            os.remove(stale)
        except OSError:
            pass
    return path


def load_checkpoint(path: str, state: TrainState, device: Optional[torch.device] = None) -> Tuple[TrainState, int]:
    payload = torch.load(path, map_location=device or "cpu")
    inner_model = unwrap_model(state.model)
    inner_model.load_state_dict(payload["model"])
    state.ema_params = {k: v.to(device or "cpu") for k, v in payload["ema_params"].items()}
    state.optimizer.load_state_dict(payload["optimizer"])
    if state.lr_scheduler is not None and payload.get("lr_scheduler") is not None:
        state.lr_scheduler.load_state_dict(payload["lr_scheduler"])
    state.step = int(payload["step"])
    state.epoch = float(payload["epoch"])
    if payload.get("generator_state") is not None and state.generator is not None:
        # torch.Generator() is CPU-only; map_location may have moved this
        # tensor to `device` along with everything else in the payload, but
        # set_state() requires a CPU ByteTensor specifically.
        state.generator.set_state(payload["generator_state"].cpu())
    state.wandb_run_id = payload.get("wandb_run_id")
    return state, state.step


def generate_wandb_run_id() -> str:
    """`wandb.util.generate_id` moved across wandb versions (also lives at
    `wandb.sdk.lib.runid.generate_id`); try both, then fall back to
    hand-rolling an id in the same format (8 lowercase base-36 chars) so a
    fresh run always gets an id even if wandb's internals move again."""
    try:
        import wandb
        return wandb.util.generate_id()
    except AttributeError:
        pass
    try:
        from wandb.sdk.lib.runid import generate_id
        return generate_id()
    except ImportError:
        pass
    import random
    import string
    return "".join(random.choices(string.digits + string.ascii_lowercase, k=8))


def resolve_run(output_dir: str, explicit_resume: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Decide what to resume from. Returns (checkpoint_path_or_None, wandb_run_id_or_None).

    `explicit_resume` may be a checkpoint file, a run directory (its latest
    checkpoint is used), or None (auto-detect the latest checkpoint already
    in `output_dir`, mirroring ELF's auto-resume).
    """
    candidate = explicit_resume
    if candidate and os.path.isdir(candidate):
        candidate = find_latest_checkpoint(candidate)
    if not candidate:
        candidate = find_latest_checkpoint(output_dir)
    if not candidate or not os.path.isfile(candidate):
        return None, None
    return candidate, peek_wandb_run_id(candidate)
