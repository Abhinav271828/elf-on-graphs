#!/usr/bin/env python3
"""Generate and cache all data splits for the shortest-path DLM-vs-ARLM comparison.

Usage:
  python scripts/generate_data.py --seed 42 --out_dir data/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spelf import data_cache  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--n_train_graphs", type=int, default=50_000)
    p.add_argument("--max_paths_per_graph", type=int, default=5,
                    help="cap on how many distinct diametric-pair paths each graph "
                         "contributes as training examples (all of them, if fewer than "
                         "this exist); also used as the number of edge-order-shuffled "
                         "copies per graph in the unlabeled encoder-pretraining corpus.")
    p.add_argument("--n_val_id_graphs", type=int, default=500)
    p.add_argument("--n_val_ood_graphs", type=int, default=300)
    p.add_argument("--n_encoder_pretrain_ood_graphs", type=int, default=10_000,
                    help="Unlabeled 11-14 node graphs added to the encoder's MLM "
                         "pretraining corpus, to avoid an OOD positional-encoding gap.")
    p.add_argument("--avg_degree", type=float, default=3.0)
    args = p.parse_args()

    data_cache.generate_all(
        out_dir=Path(args.out_dir),
        seed=args.seed,
        n_train_graphs=args.n_train_graphs,
        n_val_id_graphs=args.n_val_id_graphs,
        n_val_ood_graphs=args.n_val_ood_graphs,
        n_encoder_pretrain_ood_graphs=args.n_encoder_pretrain_ood_graphs,
        max_paths_per_graph=args.max_paths_per_graph,
        avg_degree=args.avg_degree,
    )


if __name__ == "__main__":
    main()
