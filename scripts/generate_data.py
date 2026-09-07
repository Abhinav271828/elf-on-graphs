#!/usr/bin/env python
"""Pre-generate the train / id_val / ood_test splits and cache them to disk
as JSONL under `config.data_dir`. Run this once before pretraining the
encoder or training the diffusion LM -- both only ever read cached data.

Usage:
    python scripts/generate_data.py --config configs/default.yml
    python scripts/generate_data.py --config_override n_train=2000 data_dir=./data_small
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from spelf.common import apply_overrides, load_config  # noqa: E402
from spelf.data_cache import generate_and_cache_all  # noqa: E402
from spelf.dataset import load_examples_jsonl  # noqa: E402


def _summarize(path: str, name: str) -> None:
    examples = load_examples_jsonl(path)
    diameters = [e["diameter"] for e in examples]
    n_nodes = [len(e["nodes"]) for e in examples]
    n_edges = [len(e["edges"]) for e in examples]
    print(f"  {name:10s}: n={len(examples):6d}  "
          f"nodes[min/mean/max]={min(n_nodes)}/{sum(n_nodes)/len(n_nodes):.1f}/{max(n_nodes)}  "
          f"edges[min/mean/max]={min(n_edges)}/{sum(n_edges)/len(n_edges):.1f}/{max(n_edges)}  "
          f"diameter[min/mean/max]={min(diameters)}/{sum(diameters)/len(diameters):.2f}/{max(diameters)}")


def main():
    parser = argparse.ArgumentParser(description="Generate ID/OOD Erdos-Renyi diametric-path data.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.config_override)

    print(f"Generating data under {config.data_dir!r} "
          f"(train={config.n_train}, id_val={config.n_id_val}, ood_test={config.n_ood_test})")
    print(f"ID nodes: [{config.id_min_nodes}, {config.id_max_nodes}]  "
          f"OOD nodes: [{config.ood_min_nodes}, {config.ood_max_nodes}]  "
          f"node-id universe size: {config.ood_max_nodes}")

    paths = generate_and_cache_all(config)
    print("Done. Split summary:")
    for name, path in paths.items():
        _summarize(path, name)


if __name__ == "__main__":
    main()
