#!/usr/bin/env python
"""Standalone evaluation of a trained checkpoint on the ID (held-out) and
OOD splits, logging the same four metrics + captioned sample images that
periodic in-training eval logs. Reattaches to the checkpoint's wandb run
(via its stored run id) so results land in the same run's history as a
"final" evaluation, rather than opening a disconnected new run.

Usage:
    python scripts/eval.py --config configs/default.yml --checkpoint runs/dlm/checkpoint_12000.pt
    python scripts/eval.py --config configs/default.yml   # auto-picks the latest checkpoint in output_dir
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch  # noqa: E402

from spelf.checkpoint import find_latest_checkpoint, peek_wandb_run_id  # noqa: E402
from spelf.common import apply_overrides, load_config, resolve_device, set_seed  # noqa: E402
from spelf.dataset import load_examples_jsonl  # noqa: E402
from spelf.dlm import ELF_models  # noqa: E402
from spelf.metrics import aggregate_metrics, pick_viz_examples, run_eval  # noqa: E402
from spelf.t5_encoder import build_pretrained_encoder, load_encoder_profile  # noqa: E402
from spelf.tokenizer import load_tokenizer  # noqa: E402
from spelf.viz import wandb_images_for_examples  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def main():
    parser = argparse.ArgumentParser(description="Standalone ID + OOD evaluation of a trained checkpoint.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file; defaults to the latest in output_dir.")
    parser.add_argument("--num_examples", type=int, default=None, help="Override eval_num_examples (-1 = full split).")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.config_override)
    if args.num_examples is not None:
        config.eval_num_examples = args.num_examples

    set_seed(config.seed)
    device = resolve_device(config)

    ckpt_path = args.checkpoint or find_latest_checkpoint(config.output_dir)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found (looked in {config.output_dir!r}); pass --checkpoint.")
    print(f"Evaluating checkpoint: {ckpt_path}")

    profile = load_encoder_profile(config.encoder_profile_path)
    if profile["encoder_model_name"] != config.encoder_model_name:
        raise ValueError("Encoder profile does not match the current config's encoder_model_name.")
    config.latent_mean, config.latent_std = profile["latent_mean"], profile["latent_std"]

    tokenizer = load_tokenizer(profile["tokenizer_name"])
    encoder = build_pretrained_encoder(config.encoder_model_name, device=device)

    model = ELF_models[config.model](
        text_encoder_dim=encoder.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens, vocab_size=len(tokenizer),
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    weights = payload["ema_params"] if payload.get("ema_params") else payload["model"]
    model.load_state_dict(weights)
    model.eval()
    print(f"Loaded weights from {ckpt_path} (step={payload['step']}, epoch={payload['epoch']:.2f}, "
          f"using {'EMA' if payload.get('ema_params') else 'raw'} weights)")

    if config.use_wandb and wandb is not None:
        wandb_run_id = peek_wandb_run_id(ckpt_path)
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, id=wandb_run_id,
                   resume="allow", name=config.wandb_run_name, mode=config.wandb_mode)
        if wandb.run is not None:
            print(f"wandb run: {wandb.run.url}")

    id_val_raw = load_examples_jsonl(os.path.join(config.data_dir, "id_val.jsonl"))
    ood_raw = load_examples_jsonl(os.path.join(config.data_dir, "ood_test.jsonl"))
    generator = torch.Generator().manual_seed(config.seed)

    for split_name, raw in (("id", id_val_raw), ("ood", ood_raw)):
        batch_size = min(config.batch_size, max(1, len(raw)))
        results = run_eval(model, encoder, tokenizer, raw, config, device, generator,
                            num_examples=config.eval_num_examples, batch_size=batch_size)
        agg = aggregate_metrics(results)
        print(f"[{split_name}] n={agg['num_examples']} "
              f"valid_path={agg['valid_path_rate']:.3f}  shortest_path={agg['shortest_path_rate']:.3f}  "
              f"correct_length={agg['correct_length_rate']:.3f}  optimal_path={agg['optimal_path_rate']:.3f}")

        if config.use_wandb and wandb is not None:
            log_dict = {f"final_eval_{split_name}/{k}": v for k, v in agg.items()}
            viz_examples = pick_viz_examples(results, config.eval_num_viz, seed=config.seed)
            if viz_examples:
                log_dict[f"final_eval_{split_name}/samples"] = wandb_images_for_examples(viz_examples)
            wandb.log(log_dict)
            for k, v in agg.items():
                wandb.summary[f"final_eval_{split_name}/{k}"] = v

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
