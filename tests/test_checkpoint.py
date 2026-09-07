import pytest
import torch
import torch.nn as nn

from spelf.checkpoint import (
    find_all_checkpoints, find_latest_checkpoint, generate_wandb_run_id, load_checkpoint,
    peek_wandb_run_id, resolve_run, save_checkpoint,
)
from spelf.train_state import TrainState


def _make_state(wandb_run_id="run-abc"):
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return TrainState(model=model, optimizer=optimizer, ema_params=TrainState.init_ema(model),
                       step=0, epoch=0.0, wandb_run_id=wandb_run_id, generator=torch.Generator())


def test_save_creates_a_checkpoint_named_by_step(tmp_path):
    state = _make_state()
    path = save_checkpoint(state, str(tmp_path), step=10)
    assert path == str(tmp_path / "checkpoint_10.pt")
    assert find_latest_checkpoint(str(tmp_path)) == path


def test_find_latest_checkpoint_orders_by_step_not_lexicographically(tmp_path):
    state = _make_state()
    save_checkpoint(state, str(tmp_path), step=9)
    save_checkpoint(state, str(tmp_path), step=10)  # "10" < "9" lexicographically, but not numerically
    save_checkpoint(state, str(tmp_path), step=100)
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint_100.pt")
    assert [p.split("_")[-1] for p in find_all_checkpoints(str(tmp_path))] == ["9.pt", "10.pt", "100.pt"]


def test_save_prunes_old_checkpoints(tmp_path):
    state = _make_state()
    for step in [1, 2, 3, 4, 5]:
        save_checkpoint(state, str(tmp_path), step=step, keep_last=2)
    remaining = find_all_checkpoints(str(tmp_path))
    assert len(remaining) == 2
    assert remaining[-1] == str(tmp_path / "checkpoint_5.pt")


def test_load_checkpoint_restores_step_epoch_and_wandb_id(tmp_path):
    state = _make_state(wandb_run_id="my-run-id")
    state.step = 42
    state.epoch = 3.5
    path = save_checkpoint(state, str(tmp_path), step=42)

    fresh = _make_state(wandb_run_id=None)
    fresh, resumed_step = load_checkpoint(path, fresh)
    assert resumed_step == 42
    assert fresh.epoch == 3.5
    assert fresh.wandb_run_id == "my-run-id"


def test_peek_wandb_run_id(tmp_path):
    state = _make_state(wandb_run_id="peek-me")
    path = save_checkpoint(state, str(tmp_path), step=1)
    assert peek_wandb_run_id(path) == "peek-me"


def test_generate_wandb_run_id_returns_distinct_strings():
    ids = {generate_wandb_run_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(isinstance(i, str) and len(i) > 0 for i in ids)


def test_resolve_run_fresh_start_returns_none(tmp_path):
    ckpt_path, run_id = resolve_run(str(tmp_path / "does_not_exist"), explicit_resume=None)
    assert ckpt_path is None and run_id is None


def test_resolve_run_auto_detects_latest_and_its_wandb_id(tmp_path):
    state = _make_state(wandb_run_id="auto-detected")
    save_checkpoint(state, str(tmp_path), step=1)
    save_checkpoint(state, str(tmp_path), step=2)

    ckpt_path, run_id = resolve_run(str(tmp_path), explicit_resume=None)
    assert ckpt_path == str(tmp_path / "checkpoint_2.pt")
    assert run_id == "auto-detected"


def test_load_checkpoint_generator_state_survives_non_cpu_map_location(tmp_path):
    # torch.Generator() is CPU-only, but `load_checkpoint(..., device=...)`
    # passes that same device as `map_location` for everything else in the
    # payload; the generator's saved state must be excluded from that move
    # (regression test for the mps/cuda resume crash this caught).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        pytest.skip("no non-cpu device available")

    state = _make_state()
    state.step, state.epoch = 5, 1.0
    path = save_checkpoint(state, str(tmp_path), step=5)

    fresh = _make_state()
    fresh.model.to(device)
    fresh, _ = load_checkpoint(path, fresh, device=device)
    # Must not raise, and the restored generator must still be usable.
    fresh.generator.manual_seed(0)
    torch.randn(3, generator=fresh.generator)


def test_resolve_run_explicit_dir(tmp_path):
    run_dir = tmp_path / "some_run"
    run_dir.mkdir()
    state = _make_state(wandb_run_id="explicit-run")
    save_checkpoint(state, str(run_dir), step=7)

    ckpt_path, run_id = resolve_run(str(tmp_path / "other_output_dir"), explicit_resume=str(run_dir))
    assert ckpt_path == str(run_dir / "checkpoint_7.pt")
    assert run_id == "explicit-run"
