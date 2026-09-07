#!/usr/bin/env python3
"""Standalone evaluation of a trained DLM or ARLM checkpoint that visualizes the overlap
between three of metrics.evaluate_generation's per-example correctness conditions as a
nested Venn diagram, for both ID and OOD splits in one figure.

`shortest` and `correct_length` (see metrics.py) are both subsets of `valid` by
construction (metrics.evaluate_generation only ever sets them True when `valid` is also
True), so this draws two uniform-size overlapping circles -- "shortest" and "correct
length" -- enclosed in a larger dashed "valid path" boundary; circle/overlap sizes are
purely schematic (not area- or count-proportional), with the actual counts and
percentages reported as text in each region. That gives four regions inside "valid path":
valid-only (neither), shortest-only, correct-length-only, and the shortest &
correct_length intersection, which is exactly `optimal` (a genuine diametric path).
Generations that aren't even a valid path fall outside the valid boundary entirely and are
reported as a count/percentage alongside it, rather than drawn as their own region --
there's nothing further to subdivide there.

Usage:
  python scripts/eval_venn.py --checkpoint runs/dlm/checkpoint_best.pt --model_kind dlm --data_dir data/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from torch.utils.data import DataLoader

from spelf import arlm as arlm_module, common, dataset as ds, dlm as dlm_module, metrics
from spelf.encoder import load_frozen_encoder
from spelf.t5_encoder import load_t5_encoder


def load_encoder_from_checkpoint(checkpoint: str, encoder_ckpt_override, device):
    """Same encoder-reconstruction logic as eval_only.py: the conditioning encoder isn't
    saved inside the DLM/ARLM checkpoint itself (it's a separately frozen/pretrained
    module), so it's rebuilt from either an explicit --encoder_ckpt or the path the
    checkpoint's own config recorded at training time."""
    ckpt_state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt_state["config"]
    encoder_kind = cfg.get("encoder_kind", "custom")
    if encoder_kind == "custom":
        encoder_ckpt = encoder_ckpt_override or cfg.get("encoder_ckpt")
        assert encoder_ckpt, "encoder_ckpt not found in checkpoint config -- pass --encoder_ckpt explicitly"
        encoder = load_frozen_encoder(encoder_ckpt, device)
    else:
        encoder = load_t5_encoder(cfg.get("t5_model_name", "t5-small"), cfg.get("d_model", 128), device)
        encoder_trainable = ckpt_state["extra_state"].get("encoder_trainable")
        assert encoder_trainable, "checkpoint's config says encoder_kind=t5 but has no saved encoder_trainable state"
        encoder.load_trainable_state_dict(encoder_trainable)
    return ckpt_state, cfg, encoder


def build_decoder(model_kind, cfg, l_tgt, encoder, device, num_sample_steps, guidance_scale):
    if model_kind == "dlm":
        model = dlm_module.DLMDecoder(
            l_tgt=l_tgt, d_model=encoder.d_model, n_layers=cfg.get("n_layers", 2), n_heads=cfg.get("n_heads", 8),
            d_mlp=cfg.get("d_mlp", 512), dropout=cfg.get("dropout", 0.1), embedding=encoder.embedding,
        ).to(device)
        # use_self_cond must match how this checkpoint was *trained* -- read back from
        # the checkpoint's own config, not a fresh CLI default (see dlm.sample's
        # docstring and eval_only.py's matching comment).
        sample_kwargs = {"num_steps": num_sample_steps, "guidance_scale": guidance_scale,
                          "use_self_cond": cfg.get("selfcond_prob", 0.0) > 0}
    else:
        model = arlm_module.GPTDecoder(
            l_tgt=l_tgt, d_model=encoder.d_model, n_layers=cfg.get("n_layers", 2), n_heads=cfg.get("n_heads", 8),
            d_mlp=cfg.get("d_mlp", 512), dropout=cfg.get("dropout", 0.1), embedding=encoder.embedding,
        ).to(device)
        sample_kwargs = {}
    return model, sample_kwargs


def _counts_from_examples(examples: list[dict], n_total: int) -> dict:
    return {
        "n": n_total,
        "valid": sum(e["valid"] for e in examples),
        "shortest": sum(e["shortest"] for e in examples),
        "correct_length": sum(e["correct_length"] for e in examples),
        "optimal": sum(e["optimal"] for e in examples),
    }


def _draw_venn(ax, counts: dict, title: str) -> None:
    """counts: {"n", "valid", "shortest", "correct_length", "optimal"} -- all *counts*,
    not rates. Draws shortest/correct_length as two fixed-size overlapping circles (sizes
    and positions are a plain schematic layout, not proportional to the counts) enclosed
    in a larger dashed "valid path" circle, with the actual count and percentage of the
    total (`n`) reported as text in each of the four regions."""
    n = max(counts["n"], 1)
    shortest_only = counts["shortest"] - counts["optimal"]
    correct_only = counts["correct_length"] - counts["optimal"]
    valid_only = counts["valid"] - counts["shortest"] - counts["correct_length"] + counts["optimal"]
    invalid = counts["n"] - counts["valid"]

    r = 1.4
    d = 1.4  # center-to-center distance -- fixed moderate overlap, same for every panel
    cx_short, cx_correct, cy = -d / 2, d / 2, 0.0
    centroid_x = (cx_short + cx_correct) / 2

    ax.add_patch(Circle((cx_short, cy), r, facecolor="tab:blue", alpha=0.35,
                         edgecolor="tab:blue", linewidth=1.5))
    ax.add_patch(Circle((cx_correct, cy), r, facecolor="tab:orange", alpha=0.35,
                         edgecolor="tab:orange", linewidth=1.5))

    r_valid = d / 2 + r + 0.6
    ax.add_patch(Circle((centroid_x, cy), r_valid, facecolor="none", edgecolor="tab:green",
                         linewidth=2.0, linestyle="--"))

    def pct(x: float) -> str:
        return f"{100 * x / n:.1f}%"

    ax.text(cx_short - r * 0.55, cy, f"shortest only\n{shortest_only} ({pct(shortest_only)})",
            ha="center", va="center", fontsize=8)
    ax.text(cx_correct + r * 0.55, cy, f"correct length only\n{correct_only} ({pct(correct_only)})",
            ha="center", va="center", fontsize=8)
    ax.text(centroid_x, cy, f"optimal\n{counts['optimal']} ({pct(counts['optimal'])})",
            ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(centroid_x, cy - r_valid * 0.75, f"valid only (neither)\n{valid_only} ({pct(valid_only)})",
            ha="center", va="center", fontsize=7.5, color="tab:green")

    ax.text(centroid_x, r_valid + 0.35, f"valid path: {counts['valid']} ({pct(counts['valid'])})",
            ha="center", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
    ax.text(centroid_x, -(r_valid + 0.55), f"invalid (outside): {invalid} ({pct(invalid)})",
            ha="center", va="top", fontsize=8, color="tab:red")

    lim = r_valid + 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 0.6, lim + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\nN={n}", fontsize=10)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_kind", type=str, required=True, choices=["dlm", "arlm"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--encoder_ckpt", type=str, default=None,
                    help="defaults to the value stored in the checkpoint's own config")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_sample_steps", type=int, default=32, help="DLM only")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="DLM only")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                    help="output PNG path; defaults to venn_{model_kind}_{checkpoint_stem}.png")
    args = p.parse_args()

    device = common.get_device()
    print(f"device={device}")

    ckpt_state, cfg, encoder = load_encoder_from_checkpoint(args.checkpoint, args.encoder_ckpt, device)
    meta = ds.load_meta(args.data_dir)
    model, sample_kwargs = build_decoder(args.model_kind, cfg, meta["l_tgt"], encoder, device,
                                          args.num_sample_steps, args.guidance_scale)
    model.load_state_dict(ckpt_state["model_state"])
    model.eval()
    print(f"loaded {args.model_kind} checkpoint from {args.checkpoint} (step {ckpt_state.get('step')})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, split_name, filename in zip(axes, ["id", "ood"], ["val_id.pt", "val_ood.pt"]):
        val_ds = ds.PathDataset(Path(args.data_dir) / filename)
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=ds.collate_fn)
        res = metrics.run_eval(model, encoder, loader, device, sample_kwargs)
        counts = _counts_from_examples(res["examples"], res["n_examples"])
        print(f"[{split_name}] {counts}")
        _draw_venn(ax, counts, title=f"{split_name.upper()} split")

    fig.suptitle(f"{args.model_kind.upper()} -- {Path(args.checkpoint).name}", fontsize=12)
    fig.tight_layout()

    out = args.out or f"venn_{args.model_kind}_{Path(args.checkpoint).stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
