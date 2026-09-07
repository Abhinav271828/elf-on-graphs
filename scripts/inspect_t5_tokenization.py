#!/usr/bin/env python
"""Print how the real T5 tokenizer splits a handful of serialized examples,
with `|` between tokens, and write them to `data_sample.txt` at the repo
root. Exists to make node-id tokenization boundaries directly inspectable --
see tokenizer.py's docstring for why spacing is chosen the way it is.

Uses `data/train.jsonl` if it exists (from scripts/generate_data.py);
otherwise generates a few fresh examples on the spot, so this works as a
quick sanity check even before the full data-generation step.

Usage:
    python scripts/inspect_t5_tokenization.py --config configs/default.yml
    python scripts/inspect_t5_tokenization.py --config configs/default.yml -n 10
"""

import argparse
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from spelf.common import apply_overrides, load_config  # noqa: E402
from spelf.data_cache import example_to_dict  # noqa: E402
from spelf.dataset import load_examples_jsonl, serialize_condition_text, serialize_target_text  # noqa: E402
from spelf.graphgen import Graph, sample_id_example, sample_ood_example  # noqa: E402
from spelf.tokenizer import load_tokenizer  # noqa: E402


def _load_or_generate_examples(config, n: int) -> list:
    train_path = os.path.join(config.data_dir, "train.jsonl")
    if os.path.isfile(train_path):
        examples = load_examples_jsonl(train_path)
        rng = random.Random(0)
        return rng.sample(examples, min(n, len(examples)))

    print(f"No cached data at {train_path!r}; generating {n} fresh examples instead "
          f"(run scripts/generate_data.py to inspect real cached data).")
    rng = random.Random(0)
    out = []
    for i in range(n):
        sampler = sample_id_example if i % 2 == 0 else sample_ood_example
        out.append(example_to_dict(sampler(rng, config)))
    return out


def render_example(idx: int, raw: dict, tokenizer) -> str:
    graph = Graph(nodes=tuple(raw["nodes"]), edges=tuple(tuple(e) for e in raw["edges"]))
    cond_text = serialize_condition_text(graph)
    target_text = serialize_target_text(raw["path"])

    cond_ids = tokenizer(cond_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    cond_tokens = tokenizer.convert_ids_to_tokens(cond_ids)
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

    lines = [
        f"=== example {idx} ({len(graph.nodes)} nodes, {len(graph.edges)} edges, diameter={raw['diameter']}) ===",
        f"condition text:   {cond_text}",
        f"condition tokens: {'|'.join(cond_tokens)}",
        f"condition ids:    {len(cond_ids)} tokens",
        f"target text:      {target_text}",
        f"target tokens:    {'|'.join(target_tokens)}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect T5 tokenization of serialized examples.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("-n", "--num_examples", type=int, default=6)
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "data_sample.txt"))
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.config_override)

    tokenizer_name = config.tokenizer_name or config.encoder_model_name
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

    examples = _load_or_generate_examples(config, args.num_examples)
    rendered = [render_example(i, raw, tokenizer) for i, raw in enumerate(examples)]

    text = "\n".join(rendered)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote {len(examples)} example(s) to {args.out}")


if __name__ == "__main__":
    main()
