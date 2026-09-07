#!/usr/bin/env python3
"""Pretrain the bidirectional graph encoder via masked node-token prediction (no path
labels). Trains on train.pt's 6-10 node inputs concatenated with
encoder_pretrain_extra.pt's 11-14 node inputs (see dataset.EncoderPretrainDataset).

Usage:
  python scripts/pretrain_encoder.py --data_dir data/ --run_dir runs/encoder/ --steps 30000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, random_split

from spelf import common, dataset as ds
from spelf.encoder import GraphEncoder, mlm_forward
from spelf.modules import SharedEmbedding


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--run_dir", type=str, default="runs/encoder")
    p.add_argument("--resume", type=str, default="latest", help="'latest', 'best', explicit path, or 'none'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--n_eval_holdout", type=int, default=2000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="shortest-path-elf")
    p.add_argument("--wandb_run_name", type=str, default="encoder-pretrain")
    p.add_argument("--wandb_group", type=str, default="graph-shortest-path")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    common.set_seed(args.seed)
    device = common.get_device()
    print(f"device={device}")

    meta = ds.load_meta(args.data_dir)
    l_in = meta["l_in"]

    full_ds = ds.EncoderPretrainDataset(args.data_dir)
    n_eval = min(args.n_eval_holdout, len(full_ds) // 10)
    n_train = len(full_ds) - n_eval
    train_ds, eval_ds = random_split(full_ds, [n_train, n_eval], generator=torch.Generator().manual_seed(args.seed))
    print(f"encoder pretrain corpus: {len(full_ds)} examples ({n_train} train / {n_eval} held-out eval)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=ds.collate_fn, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=ds.collate_fn)

    embedding = SharedEmbedding(vocab_size=meta["vocab_size"], d_model=args.d_model)
    model = GraphEncoder(l_in=l_in, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                          d_mlp=args.d_mlp, dropout=args.dropout, embedding=embedding).to(device)

    optimizer = common.build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = common.build_lr_schedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps)

    config = vars(args) | {"l_in": l_in, "l_tgt": meta["l_tgt"], "vocab_size": meta["vocab_size"]}

    start_step = 0
    best_eval_loss = None
    resumed_wandb_run_id = None
    ckpt_path = common.resolve_checkpoint_path(args.run_dir, args.resume)
    if ckpt_path is not None:
        state = common.load_checkpoint(ckpt_path, model, optimizer, scheduler, map_location=device)
        start_step = state["step"] + 1
        best_eval_loss = state["extra_state"].get("best_eval_loss")
        resumed_wandb_run_id = state["extra_state"].get("wandb_run_id")
        print(f"resumed from {ckpt_path} at step {start_step}")

    run = common.wandb_init(args.wandb_project, args.wandb_run_name, args.wandb_group, config,
                             mode=args.wandb_mode, run_id=resumed_wandb_run_id)

    data_iter = infinite_loader(train_loader)
    t0 = time.time()
    for step in range(start_step, args.steps):
        model.train()
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)

        optimizer.zero_grad()
        out = mlm_forward(model, input_ids, input_mask, mlm_prob=args.mlm_prob)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step} loss {out['loss'].item():.4f} acc {out['accuracy']:.4f} lr {scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")
            run.log({"train/loss": out["loss"].item(), "train/accuracy": out["accuracy"],
                      "train/lr": scheduler.get_last_lr()[0]}, step=step)

        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            eval_losses, eval_accs, n_eval_masked = [], [], 0
            with torch.no_grad():
                for eb in eval_loader:
                    eb_ids = eb["input_ids"].to(device)
                    eb_mask = eb["input_mask"].to(device)
                    eo = mlm_forward(model, eb_ids, eb_mask, mlm_prob=args.mlm_prob)
                    eval_losses.append(eo["loss"].item())
                    eval_accs.append(eo["accuracy"])
                    n_eval_masked += eo["n_masked"]
            eval_loss = sum(eval_losses) / max(1, len(eval_losses))
            eval_acc = sum(eval_accs) / max(1, len(eval_accs))
            print(f"  eval @ step {step}: loss {eval_loss:.4f} acc {eval_acc:.4f}")
            run.log({"eval/loss": eval_loss, "eval/accuracy": eval_acc}, step=step)

            extra_state = {"best_eval_loss": best_eval_loss, "wandb_run_id": run.id}
            improved = best_eval_loss is None or eval_loss < best_eval_loss
            if improved:
                best_eval_loss = eval_loss
                extra_state["best_eval_loss"] = best_eval_loss
                common.save_checkpoint(args.run_dir, step, model, optimizer, scheduler,
                                        extra_state=extra_state, config=config, tag="best")
            common.save_checkpoint(args.run_dir, step, model, optimizer, scheduler,
                                    extra_state=extra_state, config=config,
                                    tag="latest", also_snapshot=True)

    common.save_checkpoint(args.run_dir, args.steps - 1, model, optimizer, scheduler,
                            extra_state={"best_eval_loss": best_eval_loss, "wandb_run_id": run.id},
                            config=config, tag="latest")
    print(f"done. best_eval_loss={best_eval_loss}")
    run.finish()


if __name__ == "__main__":
    main()
