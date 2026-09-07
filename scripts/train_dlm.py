#!/usr/bin/env python3
"""Train the ELF-style DLM decoder, conditioned on a frozen pretrained graph encoder.

Usage:
  python scripts/train_dlm.py --data_dir data/ --encoder_ckpt runs/encoder/checkpoint_best.pt --run_dir runs/dlm/
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from spelf import common, dataset as ds, dlm as dlm_module, metrics, tokenizer as tok, viz
from spelf.encoder import load_frozen_encoder
from spelf import t5_encoder


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--encoder_kind", type=str, default="custom", choices=["custom", "t5"],
                    help="'custom' conditions on this project's own pretrained GraphEncoder "
                         "(--encoder_ckpt required), diffusing in that encoder's own "
                         "(from-scratch-pretrained) embedding space. 't5' conditions on a "
                         "frozen pretrained HuggingFace T5 encoder over a text serialization "
                         "of the graph AND diffuses directly in T5's own frozen embedding "
                         "space -- matching the canonical ELF implementation's actual use of "
                         "T5 (arXiv:2605.10938) -- rather than merely using T5 as a "
                         "conditioning source (--encoder_ckpt ignored; see --t5_model_name, "
                         "and t5_encoder.T5DiffusionEncoder's docstring).")
    p.add_argument("--encoder_ckpt", type=str, default=None,
                    help="required when --encoder_kind=custom")
    p.add_argument("--t5_model_name", type=str, default="t5-small",
                    help="HuggingFace T5 checkpoint name, used when --encoder_kind=t5. "
                         "--encoder_kind=t5 diffuses directly in this model's own frozen "
                         "embedding space (see t5_encoder.T5DiffusionEncoder) -- d_model "
                         "is therefore always this T5 model's own hidden size, not a free "
                         "choice; there is no --d_model flag here (custom mode infers its "
                         "d_model from --encoder_ckpt the same way).")
    p.add_argument("--run_dir", type=str, default="runs/dlm")
    p.add_argument("--resume", type=str, default="latest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=150_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"],
                    help="'muon' (default) matches the canonical ELF implementation "
                         "(arXiv:2605.10938, Section 4): Muon over the decoder's own "
                         ">=2D hidden weight matrices, AdamW (--lr/--weight_decay above) "
                         "over embeddings/norms/biases/null_context -- see "
                         "common.build_dlm_optimizer. 'adamw' uses a single AdamW "
                         "optimizer over everything, as in train_arlm.py/pretrain_encoder.py.")
    p.add_argument("--muon_lr", type=float, default=0.02,
                    help="peak LR for the Muon param group; only used when --optimizer muon. "
                         "The ELF paper reports 0.002 -- pass that explicitly to match it; "
                         "this default instead follows Muon's own more common convention "
                         "(Jordan et al.), an order of magnitude above typical AdamW LRs.")
    p.add_argument("--muon_momentum", type=float, default=0.95,
                    help="only used when --optimizer muon.")
    p.add_argument("--muon_weight_decay", type=float, default=0.0,
                    help="only used when --optimizer muon; decoupled weight decay on the "
                         "Muon param group, applied separately from --weight_decay.")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--decode_branch_prob", type=float, default=0.2)
    p.add_argument("--selfcond_prob", type=float, default=0.0,
                    help="probability a training example's self_cond input comes from a "
                         "real (detached) first-pass prediction instead of zeros. Default 0.0 "
                         "runs the DLM in vanilla mode (no self-conditioning at train OR sample "
                         "time -- see dlm.sample's use_self_cond, threaded from this below).")
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--num_sample_steps", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--full_eval_every", type=int, default=5000)
    p.add_argument("--n_id_subsample", type=int, default=500)
    p.add_argument("--n_ood_subsample", type=int, default=300)
    p.add_argument("--n_viz_examples", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="shortest-path-elf")
    p.add_argument("--wandb_run_name", type=str, default="dlm-run1")
    p.add_argument("--wandb_group", type=str, default="graph-shortest-path")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    common.set_seed(args.seed)
    device = common.get_device()
    print(f"device={device}")

    meta = ds.load_meta(args.data_dir)
    l_in, l_tgt, vocab_size = meta["l_in"], meta["l_tgt"], meta["vocab_size"]

    if args.encoder_kind == "custom":
        assert args.encoder_ckpt, "--encoder_ckpt is required when --encoder_kind=custom"
        encoder = load_frozen_encoder(args.encoder_ckpt, device)
        d_model = encoder.d_model
        print(f"loaded frozen custom encoder from {args.encoder_ckpt} (d_model={d_model})")
    else:
        # Diffuses directly in T5's own frozen embedding space (see
        # t5_encoder.T5DiffusionEncoder) -- both d_model and l_tgt are therefore derived
        # from the T5 model itself, not free CLI choices; l_tgt_t5 also replaces
        # meta["l_tgt"] as this run's diffusion-canvas length.
        encoder = t5_encoder.load_t5_diffusion_encoder(args.t5_model_name, device)
        d_model = encoder.d_model
        l_tgt = encoder.l_tgt_t5
        print(f"loaded frozen T5 diffusion encoder '{args.t5_model_name}' -- diffusing "
              f"directly in T5's own embedding space: d_model={d_model}, l_tgt_t5={l_tgt} "
              f"(vs. this project's own vocab_size={vocab_size}, l_tgt={meta['l_tgt']})")

    model = dlm_module.DLMDecoder(
        l_tgt=l_tgt, d_model=d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        d_mlp=args.d_mlp, dropout=args.dropout, embedding=encoder.embedding,
    ).to(device)

    train_ds = ds.PathDataset(Path(args.data_dir) / "train.pt")
    val_id_ds = ds.PathDataset(Path(args.data_dir) / "val_id.pt")
    val_ood_ds = ds.PathDataset(Path(args.data_dir) / "val_ood.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=ds.collate_fn, drop_last=True)
    val_id_loader = DataLoader(val_id_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=ds.collate_fn)
    val_ood_loader = DataLoader(val_ood_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=ds.collate_fn)
    n_id_batches = math.ceil(args.n_id_subsample / args.batch_size)
    n_ood_batches = math.ceil(args.n_ood_subsample / args.batch_size)

    if args.optimizer == "muon":
        optimizer = common.build_dlm_optimizer(
            [model, encoder], adamw_lr=args.lr, adamw_weight_decay=args.weight_decay,
            muon_lr=args.muon_lr, muon_momentum=args.muon_momentum, muon_weight_decay=args.muon_weight_decay,
        )
        scheduler = common.build_multi_lr_schedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    else:
        optimizer = common.build_optimizer([model, encoder], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = common.build_lr_schedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    early_stopper = common.EarlyStopper(tolerance=args.tolerance, patience=args.patience, mode="max")
    sample_kwargs = {"num_steps": args.num_sample_steps, "guidance_scale": args.guidance_scale,
                      "use_self_cond": args.selfcond_prob > 0}

    config = vars(args) | {"l_in": l_in, "l_tgt": l_tgt, "vocab_size": vocab_size, "d_model": d_model}
    # eval_decoder is what metrics.run_eval actually calls .generate() on: in T5-diffusion
    # mode this is the raw model wrapped so it still speaks the project's own vocab to
    # every downstream consumer (metrics.py, viz.py) -- see T5SpaceDecoderAdapter's docstring.
    eval_decoder = (
        t5_encoder.T5SpaceDecoderAdapter(model, encoder.t5_tokenizer) if args.encoder_kind == "t5" else model
    )

    start_step = 0
    resumed_wandb_run_id = None
    ckpt_path = common.resolve_checkpoint_path(args.run_dir, args.resume)
    if ckpt_path is not None:
        common.check_checkpoint_config(
            ckpt_path, {"encoder_kind": args.encoder_kind, "optimizer": args.optimizer, "t5_model_name": args.t5_model_name}
        )
        state = common.load_checkpoint(ckpt_path, model, optimizer, scheduler, map_location=device)
        start_step = state["step"] + 1
        if "early_stopper" in state["extra_state"]:
            early_stopper.load_state_dict(state["extra_state"]["early_stopper"])
        if args.encoder_kind == "t5" and "encoder_trainable" in state["extra_state"]:
            encoder.load_trainable_state_dict(state["extra_state"]["encoder_trainable"])
        resumed_wandb_run_id = state["extra_state"].get("wandb_run_id")
        print(f"resumed from {ckpt_path} at step {start_step}")

    run = common.wandb_init(args.wandb_project, args.wandb_run_name, args.wandb_group, config,
                             mode=args.wandb_mode, run_id=resumed_wandb_run_id)

    def do_eval(step: int, max_batches, prefix: str, log_images: bool):
        id_res = metrics.run_eval(eval_decoder, encoder, val_id_loader, device, sample_kwargs, max_batches=max_batches)
        ood_res = metrics.run_eval(eval_decoder, encoder, val_ood_loader, device, sample_kwargs, max_batches=max_batches)
        run.log({
            f"{prefix}/id/token_accuracy": id_res["token_accuracy"],
            f"{prefix}/id/token_accuracy_nopad": id_res["token_accuracy_nopad"],
            f"{prefix}/id/exact_match_rate": id_res["exact_match_rate"],
            f"{prefix}/id/valid_rate": id_res["valid_rate"],
            f"{prefix}/id/shortest_rate": id_res["shortest_rate"],
            f"{prefix}/id/correct_length_rate": id_res["correct_length_rate"],
            f"{prefix}/id/optimal_rate": id_res["optimal_rate"],
            f"{prefix}/ood/token_accuracy": ood_res["token_accuracy"],
            f"{prefix}/ood/token_accuracy_nopad": ood_res["token_accuracy_nopad"],
            f"{prefix}/ood/exact_match_rate": ood_res["exact_match_rate"],
            f"{prefix}/ood/valid_rate": ood_res["valid_rate"],
            f"{prefix}/ood/shortest_rate": ood_res["shortest_rate"],
            f"{prefix}/ood/correct_length_rate": ood_res["correct_length_rate"],
            f"{prefix}/ood/optimal_rate": ood_res["optimal_rate"],
        }, step=step)
        print(f"  [{prefix}] step {step} ID exact={id_res['exact_match_rate']:.3f} valid={id_res['valid_rate']:.3f} "
              f"shortest={id_res['shortest_rate']:.3f} correct_len={id_res['correct_length_rate']:.3f} opt={id_res['optimal_rate']:.3f} "
              f"| OOD exact={ood_res['exact_match_rate']:.3f} valid={ood_res['valid_rate']:.3f} "
              f"shortest={ood_res['shortest_rate']:.3f} correct_len={ood_res['correct_length_rate']:.3f} opt={ood_res['optimal_rate']:.3f}")
        if log_images:
            viz.log_examples_to_wandb(run, metrics.sample_examples(id_res["examples"], args.n_viz_examples), "id", step)
            viz.log_examples_to_wandb(run, metrics.sample_examples(ood_res["examples"], args.n_viz_examples), "ood", step)
        return id_res

    data_iter = infinite_loader(train_loader)
    t0 = time.time()
    stopped = False
    for step in range(start_step, args.max_steps):
        model.train()
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        # In T5-diffusion mode, target_ids must be T5's own tokenization of the
        # serialized path text (what the DLM actually diffuses into), not the project's
        # own vocab -- see T5DiffusionEncoder.tokenize_path_targets.
        if args.encoder_kind == "t5":
            target_ids = encoder.tokenize_path_targets(batch["target_ids"], device)
            pad_id = encoder.t5_tokenizer.pad_token_id
        else:
            target_ids = batch["target_ids"].to(device)
            pad_id = tok.PAD

        context, context_mask = common.encode_context(encoder, input_ids, input_mask)

        optimizer.zero_grad()
        out = dlm_module.loss(
            model, context, context_mask, target_ids, cfg_dropout_prob=args.cfg_dropout,
            decode_branch_prob=args.decode_branch_prob, selfcond_prob=args.selfcond_prob, pad_id=pad_id,
            lambda_ce=args.lambda_ce,
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in common.trainable_parameters([model, encoder])], args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            log_dict = {"train/loss": out["loss"].item(), "train/denoise_loss": out["denoise_loss"].item(),
                        "train/decode_loss": out["decode_loss"].item()}
            if args.optimizer == "muon":
                lrs = scheduler.get_last_lr()  # {"muon": [...], "adamw": [...]}
                lr_str = " ".join(f"lr_{name}={vals[0]:.2e}" for name, vals in lrs.items())
                log_dict.update({f"train/lr_{name}": vals[0] for name, vals in lrs.items()})
            else:
                lr_str = f"lr {scheduler.get_last_lr()[0]:.2e}"
                log_dict["train/lr"] = scheduler.get_last_lr()[0]
            print(f"step {step} loss {out['loss'].item():.4f} denoise {out['denoise_loss'].item():.4f} "
                  f"decode {out['decode_loss'].item():.4f} {lr_str} ({elapsed:.1f}s)")
            run.log(log_dict, step=step)

        do_full = (step % args.full_eval_every == 0) or (step == args.max_steps - 1)
        do_cheap = (step % args.eval_every == 0) or do_full or (step == args.max_steps - 1)

        if do_cheap:
            id_res = do_eval(step, n_id_batches, "eval", log_images=True)
            should_stop = early_stopper.step(id_res["optimal_rate"])
            is_best = early_stopper.best == id_res["optimal_rate"]
            extra_state = {"early_stopper": early_stopper.state_dict(), "wandb_run_id": run.id}
            if args.encoder_kind == "t5":
                extra_state["encoder_trainable"] = encoder.trainable_state_dict()
            if is_best:
                common.save_checkpoint(args.run_dir, step, model, optimizer, scheduler, extra_state, config, tag="best")
            common.save_checkpoint(args.run_dir, step, model, optimizer, scheduler, extra_state, config,
                                    tag="latest", also_snapshot=True)
            if should_stop:
                print(f"early stopping at step {step}: ID optimal-rate plateaued (best={early_stopper.best:.4f})")
                stopped = True

        if do_full:
            do_eval(step, None, "eval_full", log_images=False)

        if stopped:
            break

    print("final full evaluation...")
    do_eval(min(step, args.max_steps - 1), None, "eval_full", log_images=True)
    run.finish()


if __name__ == "__main__":
    main()
