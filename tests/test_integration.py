"""An end-to-end smoke test: the real frozen pretrained T5 encoder + a tiny
ELF model, a few real `train_step` updates with the Muon optimizer, then a
full conditional generation + decode + metrics pass through `run_eval`.
This is the test most likely to catch a wiring bug between modules (shapes,
mask semantics, dtypes) that a narrower unit test would miss.
"""

import random

import torch

from spelf.data_cache import example_to_dict
from spelf.dataset import PathDataset, get_dataloader
from spelf.dlm import ELF
from spelf.graphgen import sample_id_example
from spelf.metrics import aggregate_metrics, run_eval
from spelf.train_state import TrainState, get_optimizer
from spelf.train_step import train_step


def test_end_to_end_train_step_and_eval(tiny_config, tokenizer, encoder):
    rng = random.Random(0)
    raw = [example_to_dict(sample_id_example(rng, tiny_config)) for _ in range(16)]

    device = torch.device("cpu")
    model = ELF(
        text_encoder_dim=encoder.d_model, max_length=tiny_config.max_length,
        hidden_size=32, depth=2, num_heads=2, bottleneck_dim=8,
        num_time_tokens=tiny_config.num_time_tokens,
        num_self_cond_cfg_tokens=tiny_config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=tiny_config.num_model_mode_tokens,
        vocab_size=len(tokenizer),
    ).to(device)

    optimizer = get_optimizer(model, tiny_config, lr=1e-2)
    generator = torch.Generator().manual_seed(0)
    state = TrainState(model=model, optimizer=optimizer, ema_params=TrainState.init_ema(model),
                        step=0, epoch=0.0, generator=generator)

    dataset = PathDataset(raw, tokenizer)
    loader = get_dataloader(dataset, tiny_config, tokenizer, batch_size=tiny_config.batch_size,
                             shuffle=True, drop_last=True)

    losses = []
    for i, batch in enumerate(loader):
        state, metrics = train_step(state, encoder, batch, tiny_config, device)
        assert torch.isfinite(metrics["loss"])
        losses.append(metrics["loss"].item())
        if i >= 3:
            break
    assert len(losses) > 0

    tiny_config.num_sampling_steps = 4
    results = run_eval(model, encoder, tokenizer, raw, tiny_config, device, generator,
                        num_examples=4, batch_size=4)
    assert len(results) == 4
    agg = aggregate_metrics(results)
    assert agg["num_examples"] == 4
    for key in ("valid_path_rate", "shortest_path_rate", "correct_length_rate", "optimal_path_rate"):
        assert 0.0 <= agg[key] <= 1.0
