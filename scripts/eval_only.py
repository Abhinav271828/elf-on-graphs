#!/usr/bin/env python3
"""Standalone evaluation of a trained DLM or ARLM checkpoint: reports (and optionally
logs to wandb / saves as PNGs) exact-match/valid/optimal rates on ID and/or OOD splits.

Usage:
  python scripts/eval_only.py --checkpoint runs/dlm/checkpoint_best.pt --model_kind dlm --data_dir data/ --split both
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from spelf import arlm as arlm_module, common, dataset as ds, dlm as dlm_module, metrics, viz
from spelf.encoder import load_frozen_encoder
from spelf.t5_encoder import load_t5_encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_kind", type=str, required=True, choices=["dlm", "arlm"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--encoder_ckpt", type=str, default=None,
                    help="defaults to the value stored in the checkpoint's own config")
    p.add_argument("--split", type=str, default="both", choices=["id", "ood", "both"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_sample_steps", type=int, default=32, help="DLM only")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="DLM only")
    p.add_argument("--n_viz_examples", type=int, default=8)
    p.add_argument("--save_viz_dir", type=str, default=None,
                    help="if set, save example PNGs here (in addition to any wandb logging)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="shortest-path-elf")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="graph-shortest-path")
    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    device = common.get_device()
    print(f"device={device}")

    ckpt_state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt_state["config"]
    encoder_kind = cfg.get("encoder_kind", "custom")
    if encoder_kind == "custom":
        encoder_ckpt = args.encoder_ckpt or cfg.get("encoder_ckpt")
        assert encoder_ckpt, "encoder_ckpt not found in checkpoint config -- pass --encoder_ckpt explicitly"
        encoder = load_frozen_encoder(encoder_ckpt, device)
    else:
        encoder = load_t5_encoder(cfg.get("t5_model_name", "t5-small"), cfg.get("d_model", 128), device)
        encoder_trainable = ckpt_state["extra_state"].get("encoder_trainable")
        assert encoder_trainable, "checkpoint's config says encoder_kind=t5 but has no saved encoder_trainable state"
        encoder.load_trainable_state_dict(encoder_trainable)

    meta = ds.load_meta(args.data_dir)
    l_tgt = meta["l_tgt"]

    if args.model_kind == "dlm":
        model = dlm_module.DLMDecoder(
            l_tgt=l_tgt, d_model=encoder.d_model, n_layers=cfg.get("n_layers", 2), n_heads=cfg.get("n_heads", 8),
            d_mlp=cfg.get("d_mlp", 512), dropout=cfg.get("dropout", 0.1), embedding=encoder.embedding,
        ).to(device)
        # use_self_cond must match how this checkpoint was *trained* (see dlm.sample's
        # docstring), not a fresh CLI default -- read it back from the checkpoint's own
        # config, defaulting to off if an older checkpoint predates this flag.
        sample_kwargs = {"num_steps": args.num_sample_steps, "guidance_scale": args.guidance_scale,
                          "use_self_cond": cfg.get("selfcond_prob", 0.0) > 0}
    else:
        model = arlm_module.GPTDecoder(
            l_tgt=l_tgt, d_model=encoder.d_model, n_layers=cfg.get("n_layers", 2), n_heads=cfg.get("n_heads", 8),
            d_mlp=cfg.get("d_mlp", 512), dropout=cfg.get("dropout", 0.1), embedding=encoder.embedding,
        ).to(device)
        sample_kwargs = {}
    model.load_state_dict(ckpt_state["model_state"])
    model.eval()
    print(f"loaded {args.model_kind} checkpoint from {args.checkpoint} (step {ckpt_state.get('step')})")

    splits = {"id": "val_id.pt", "ood": "val_ood.pt"} if args.split == "both" else {args.split: f"val_{args.split}.pt"}

    run_name = args.wandb_run_name or f"eval-{args.model_kind}-{Path(args.checkpoint).stem}"
    run = common.wandb_init(args.wandb_project, run_name, args.wandb_group, vars(args), mode=args.wandb_mode)

    if args.save_viz_dir:
        Path(args.save_viz_dir).mkdir(parents=True, exist_ok=True)

    for split_name, filename in splits.items():
        val_ds = ds.PathDataset(Path(args.data_dir) / filename)
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=ds.collate_fn)
        res = metrics.run_eval(model, encoder, loader, device, sample_kwargs)
        print(f"[{split_name}] n={res['n_examples']} token_acc={res['token_accuracy']:.4f} "
              f"token_acc_nopad={res['token_accuracy_nopad']:.4f} "
              f"exact_match={res['exact_match_rate']:.4f} valid={res['valid_rate']:.4f} "
              f"shortest={res['shortest_rate']:.4f} correct_len={res['correct_length_rate']:.4f} "
              f"optimal={res['optimal_rate']:.4f}")
        run.log({
            f"eval/{split_name}/token_accuracy": res["token_accuracy"],
            f"eval/{split_name}/token_accuracy_nopad": res["token_accuracy_nopad"],
            f"eval/{split_name}/exact_match_rate": res["exact_match_rate"],
            f"eval/{split_name}/valid_rate": res["valid_rate"],
            f"eval/{split_name}/shortest_rate": res["shortest_rate"],
            f"eval/{split_name}/correct_length_rate": res["correct_length_rate"],
            f"eval/{split_name}/optimal_rate": res["optimal_rate"],
        })

        viz_examples = res["examples"][: args.n_viz_examples]
        if args.wandb_mode != "disabled":
            viz.log_examples_to_wandb(run, viz_examples, split_name, step=0)
        if args.save_viz_dir:
            import matplotlib.pyplot as plt
            for i, ex in enumerate(viz_examples):
                fig = viz.plot_example(ex)
                fig.savefig(Path(args.save_viz_dir) / f"{split_name}_{i}.png", dpi=120)
                plt.close(fig)

    run.finish()


if __name__ == "__main__":
    main()
