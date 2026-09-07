#!/usr/bin/env python
"""Load the pretrained, frozen T5 encoder and compute the latent
normalization stats it needs before entering the diffusion LM.

No training happens here -- the encoder is real pretrained T5 weights
(`transformers.T5EncoderModel.from_pretrained`), used exactly as ELF uses
one. The only thing computed locally is `latent_mean`/`latent_std`: the
scalar mean/std this frozen encoder's outputs are rescaled by (as in ELF's
`encode_text`), measured over the *actual* training collate pipeline
(condition + target concatenated, run through the encoder with the same
`encoder_attention_mask` train_step.py will use) rather than condition text
alone, so the stats match what training will really see. Cached to
`config.encoder_profile_path` as a small JSON file (model/tokenizer names +
the two stats) -- there's no encoder checkpoint to save since the weights
are just re-downloaded via `from_pretrained` on every run.

Usage:
    python scripts/generate_data.py   --config configs/default.yml   # once, first
    python scripts/prepare_encoder.py --config configs/default.yml
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch  # noqa: E402

from spelf.common import Config, apply_overrides, load_config, resolve_device, set_seed  # noqa: E402
from spelf.dataset import PathDataset, get_dataloader, load_examples_jsonl  # noqa: E402
from spelf.t5_encoder import build_pretrained_encoder, save_encoder_profile  # noqa: E402
from spelf.tokenizer import load_tokenizer  # noqa: E402

STATS_BATCH_SIZE = 64


@torch.no_grad()
def compute_latent_stats(encoder, dataloader, device, sample_size: int):
    total_sum = total_sq = 0.0
    total_count = 0
    seen = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device).long()
        encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.float32)

        out = encoder(input_ids=input_ids, attention_mask=encoder_attention_mask, deterministic=True)
        valid = attention_mask.bool().unsqueeze(-1).expand_as(out)
        vals = out[valid]
        total_sum += vals.sum().item()
        total_sq += (vals ** 2).sum().item()
        total_count += vals.numel()

        seen += input_ids.shape[0]
        if seen >= sample_size:
            break

    mean = total_sum / total_count
    var = max(total_sq / total_count - mean ** 2, 1e-8)
    return mean, var ** 0.5


def prepare_encoder(config: Config) -> None:
    set_seed(config.seed)
    device = resolve_device(config)
    print(f"Device: {device}")

    tokenizer_name = config.tokenizer_name or config.encoder_model_name
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

    print(f"Loading pretrained encoder: {config.encoder_model_name}")
    encoder = build_pretrained_encoder(config.encoder_model_name, device=device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder (frozen): {n_params:,} params, d_model={encoder.d_model}")

    raw = load_examples_jsonl(os.path.join(config.data_dir, "train.jsonl"))
    dataset = PathDataset(raw, tokenizer)
    dataloader = get_dataloader(dataset, config, tokenizer, batch_size=STATS_BATCH_SIZE,
                                 shuffle=True, drop_last=False)

    print(f"Computing latent stats over up to {config.latent_stats_sample_size} training examples...")
    latent_mean, latent_std = compute_latent_stats(encoder, dataloader, device, config.latent_stats_sample_size)
    print(f"Latent stats: mean={latent_mean:.4f} std={latent_std:.4f}")

    save_encoder_profile(config.encoder_profile_path, config.encoder_model_name, tokenizer_name,
                          latent_mean, latent_std)
    print(f"Saved encoder profile to {config.encoder_profile_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the frozen pretrained T5 encoder (compute latent stats).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.config_override)
    prepare_encoder(config)


if __name__ == "__main__":
    main()
