#!/usr/bin/env python
"""Train the ELF diffusion LM on the diametric-path task.

Wandb logging is on by default (`config.use_wandb=True`); checkpointing and
wandb resume are synced via the run id stored in every checkpoint (see
`spelf/checkpoint.py`). Every `eval_freq` epochs, both the ID (held-out) and
OOD splits are scored (valid_path / shortest_path / correct_length /
optimal_path rates) and `eval_num_viz` sample graphs per split are rendered
and logged as captioned `wandb.Image`s.

Usage:
    python scripts/generate_data.py   --config configs/default.yml   # once
    python scripts/prepare_encoder.py --config configs/default.yml   # once
    python scripts/train.py --config configs/default.yml
    python scripts/train.py --config configs/default.yml --config_override output_dir=./runs/dlm_run2
"""

import argparse
import copy
import dataclasses
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch  # noqa: E402

from spelf.checkpoint import generate_wandb_run_id, load_checkpoint, resolve_run, save_checkpoint  # noqa: E402
from spelf.common import Config, apply_overrides, load_config, resolve_device, save_config, set_seed  # noqa: E402
from spelf.dataset import PathDataset, get_dataloader, load_examples_jsonl  # noqa: E402
from spelf.dlm import ELF_models  # noqa: E402
from spelf.metrics import aggregate_metrics, pick_viz_examples, run_eval  # noqa: E402
from spelf.t5_encoder import build_pretrained_encoder, load_encoder_profile  # noqa: E402
from spelf.tokenizer import load_tokenizer  # noqa: E402
from spelf.train_state import (  # noqa: E402
    TrainState, attach_lr_scheduler, create_learning_rate_fn, get_optimizer, unwrap_model,
)
from spelf.train_step import train_step  # noqa: E402
from spelf.viz import wandb_images_for_examples  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def _build_eval_model(state: TrainState) -> torch.nn.Module:
    """A frozen, eval-mode copy of the model with EMA weights loaded."""
    model = copy.deepcopy(unwrap_model(state.model))
    if state.ema_params:
        model.load_state_dict(state.ema_params)
    model.eval()
    return model


def run_periodic_eval(state: TrainState, encoder, tokenizer, id_val_raw, ood_raw, config, device, global_step: int):
    eval_model = _build_eval_model(state)
    generator = torch.Generator().manual_seed(config.seed)

    for split_name, raw in (("id", id_val_raw), ("ood", ood_raw)):
        batch_size = min(config.batch_size, max(1, config.eval_num_examples if config.eval_num_examples > 0 else len(raw)))
        results = run_eval(eval_model, encoder, tokenizer, raw, config, device, generator,
                            num_examples=config.eval_num_examples, batch_size=batch_size)
        agg = aggregate_metrics(results)
        print(f"  [eval/{split_name}] n={agg['num_examples']} "
              f"valid={agg['valid_path_rate']:.3f} shortest={agg['shortest_path_rate']:.3f} "
              f"correct_len={agg['correct_length_rate']:.3f} optimal={agg['optimal_path_rate']:.3f}")

        if config.use_wandb and wandb is not None:
            log_dict = {f"eval_{split_name}/{k}": v for k, v in agg.items() if k != "num_examples"}
            log_dict[f"eval_{split_name}/num_examples"] = agg["num_examples"]
            viz_examples = pick_viz_examples(results, config.eval_num_viz, seed=config.seed)
            if viz_examples:
                log_dict[f"eval_{split_name}/samples"] = wandb_images_for_examples(viz_examples)
            wandb.log(log_dict, step=global_step)


def run_training(config: Config) -> None:
    set_seed(config.seed)
    device = resolve_device(config)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Device: {device}  |  output_dir: {config.output_dir}")

    resume_ckpt_path, wandb_run_id = resolve_run(config.output_dir, config.resume)
    if wandb_run_id is None and config.use_wandb and wandb is not None:
        wandb_run_id = generate_wandb_run_id()

    if config.use_wandb and wandb is not None:
        tags = config.wandb_tags.split(",") if config.wandb_tags else None
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, id=wandb_run_id,
                   resume="allow", name=config.wandb_run_name, tags=tags, mode=config.wandb_mode,
                   config=dataclasses.asdict(config), dir=config.output_dir)
        if wandb.run is not None:
            print(f"wandb run: {wandb.run.url or '(offline)'} (id={wandb_run_id})")
    elif config.use_wandb and wandb is None:
        print("WARNING: use_wandb=True but the `wandb` package is not installed; continuing without it.")

    print(f"Loading encoder profile from {config.encoder_profile_path}...")
    profile = load_encoder_profile(config.encoder_profile_path)
    if profile["encoder_model_name"] != config.encoder_model_name:
        raise ValueError(
            f"Encoder profile at {config.encoder_profile_path!r} was built for "
            f"encoder_model_name={profile['encoder_model_name']!r}, but the current config says "
            f"{config.encoder_model_name!r}. Re-run prepare_encoder.py, or fix the config."
        )
    config.latent_mean, config.latent_std = profile["latent_mean"], profile["latent_std"]

    tokenizer_name = profile["tokenizer_name"]
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

    print(f"Loading pretrained encoder: {config.encoder_model_name}")
    encoder = build_pretrained_encoder(config.encoder_model_name, device=device)
    print(f"Encoder: d_model={encoder.d_model}  latent_mean={config.latent_mean:.4f}  latent_std={config.latent_std:.4f}")

    model = ELF_models[config.model](
        text_encoder_dim=encoder.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens, vocab_size=len(tokenizer),
        bottleneck_dim=config.bottleneck_dim, gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ELF model ({config.model}): {n_params:,} params")

    train_raw = load_examples_jsonl(os.path.join(config.data_dir, "train.jsonl"))
    id_val_raw = load_examples_jsonl(os.path.join(config.data_dir, "id_val.jsonl"))
    ood_raw = load_examples_jsonl(os.path.join(config.data_dir, "ood_test.jsonl"))
    train_dataset = PathDataset(train_raw, tokenizer)
    train_loader = get_dataloader(train_dataset, config, tokenizer, batch_size=config.batch_size,
                                   shuffle=True, drop_last=True)
    print(f"Train: {len(train_dataset)}  ID-val: {len(id_val_raw)}  OOD-test: {len(ood_raw)}")

    steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
    num_train_steps = steps_per_epoch * config.epochs
    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * config.batch_size / 256
    lr_fn = create_learning_rate_fn(num_train_steps, config.warmup_steps, config.lr, config.lr_schedule, config.min_lr)
    optimizer = get_optimizer(model, config, config.lr)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)
    print(f"steps/epoch={steps_per_epoch}  total_steps={num_train_steps}  lr={config.lr:.2e}  optimizer={config.optimizer}")

    generator = torch.Generator().manual_seed(config.seed)
    state = TrainState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        ema_params=TrainState.init_ema(model), step=0, epoch=0.0,
                        wandb_run_id=wandb_run_id, generator=generator)

    if resume_ckpt_path:
        state, resumed_step = load_checkpoint(resume_ckpt_path, state, device=device)
        print(f"Resumed from {resume_ckpt_path}: step={resumed_step} epoch={state.epoch:.2f}")

    save_config(config, os.path.join(config.output_dir, "config.yml"))

    start_epoch = int(state.epoch)
    last_save_frac_epoch = state.epoch
    global_step = state.step
    t0 = time.time()

    for epoch in range(start_epoch, config.epochs):
        for step_in_epoch, batch in enumerate(train_loader):
            state, metrics = train_step(state, encoder, batch, config, device)
            global_step = state.step

            if global_step % config.log_freq == 0:
                lr_now = state.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(f"step {global_step} epoch {epoch + 1}/{config.epochs} "
                      f"loss={metrics['loss'].item():.4f} l2={metrics['l2_loss'].item():.4f} "
                      f"ce={metrics['ce_loss'].item():.4f} lr={lr_now:.2e} ({elapsed:.1f}s)")
                if config.use_wandb and wandb is not None:
                    wandb.log({
                        "train/loss": metrics["loss"].item(), "train/l2_loss": metrics["l2_loss"].item(),
                        "train/ce_loss": metrics["ce_loss"].item(), "train/lr": lr_now,
                        "epoch": epoch + (step_in_epoch + 1) / steps_per_epoch,
                    }, step=global_step)

            if 0 < config.save_freq < 1:
                progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                if progress - last_save_frac_epoch >= config.save_freq:
                    state.epoch = progress
                    save_checkpoint(state, config.output_dir, global_step)
                    last_save_frac_epoch = progress

        state.epoch = float(epoch + 1)
        current_epoch = epoch + 1

        if config.save_freq >= 1 and current_epoch % int(config.save_freq) == 0:
            save_checkpoint(state, config.output_dir, global_step)
            print(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            print(f"Running eval at epoch {current_epoch}...")
            run_periodic_eval(state, encoder, tokenizer, id_val_raw, ood_raw, config, device, global_step)

    save_checkpoint(state, config.output_dir, global_step)
    print(f"Final checkpoint saved (step {global_step}).")
    print("Running final eval...")
    run_periodic_eval(state, encoder, tokenizer, id_val_raw, ood_raw, config, device, global_step)

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train the ELF diffusion LM.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.config_override)
    run_training(config)


if __name__ == "__main__":
    main()
