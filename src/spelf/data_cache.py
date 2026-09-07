"""Generate the three data splits (train / id_val / ood_test) once, offline,
and cache them to disk as JSONL. Training and evaluation only ever read from
these files -- graphs are never generated on the fly during training, per
the task spec ("data should be generated beforehand").

Splits:
  - train.jsonl:    ID graphs (id_min_nodes..id_max_nodes), used for training.
  - id_val.jsonl:    ID graphs, disjoint RNG stream from train -- the
                       in-distribution held-out eval set.
  - ood_test.jsonl:  OOD graphs (ood_min_nodes..ood_max_nodes) -- the
                       out-of-distribution eval set.
"""

from __future__ import annotations

import json
import os
import random
from typing import Callable, List

from .common import Config
from .graphgen import DiametricExample, sample_id_example, sample_ood_example


def example_to_dict(example: DiametricExample) -> dict:
    return {
        "nodes": list(example.graph.nodes),
        "edges": [list(e) for e in example.graph.edges],
        "source": example.source,
        "target": example.target,
        "path": list(example.path),
        "diameter": example.diameter,
    }


def generate_split(seed: int, n_examples: int, sampler: Callable, config: Config) -> List[dict]:
    rng = random.Random(seed)
    return [example_to_dict(sampler(rng, config)) for _ in range(n_examples)]


def write_jsonl(examples: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_and_cache_all(config: Config) -> dict:
    """Generate all three splits and write them under `config.data_dir`.

    Returns the dict of {split_name: path}. Distinct seeds (derived from
    `config.data_seed`) keep the three RNG streams independent, so id_val is
    not merely "more of train" by construction accident.
    """
    os.makedirs(config.data_dir, exist_ok=True)

    splits = {
        "train": (config.data_seed, config.n_train, sample_id_example),
        "id_val": (config.data_seed + 1, config.n_id_val, sample_id_example),
        "ood_test": (config.data_seed + 2, config.n_ood_test, sample_ood_example),
    }
    paths = {}
    for name, (seed, n, sampler) in splits.items():
        examples = generate_split(seed, n, sampler, config)
        path = os.path.join(config.data_dir, f"{name}.jsonl")
        write_jsonl(examples, path)
        paths[name] = path
    return paths
