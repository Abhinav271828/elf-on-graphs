#!/usr/bin/env python3
"""Compute descriptive statistics over a generated dataset (see generate_data.py /
data_cache.py) and write them to a human-readable text report: per-split example/graph
counts, token-count statistics (input and target sequence lengths, unpadded), node-count
statistics, and graph-connectivity statistics (edge count, average degree, density,
diameter, average shortest-path length).

Connectivity/node-count stats are computed once per *unique* graph in each split, not per
example -- a graph typically contributes several examples (one per diametric path in the
labeled splits, or several independently edge-order-shuffled copies in the unlabeled
encoder-pretraining corpus), which would otherwise skew those stats towards graphs with
more paths/copies. Token-count stats are computed per example instead, since that's what
actually varies example-to-example and is what a model/dataloader sees.

Usage:
  python scripts/dataset_stats.py --data_dir data/ --out data/dataset_stats.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import networkx as nx
import torch

from spelf import data_cache, dataset as ds, tokenizer as tok


def _stat(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": t.mean().item(), "std": t.std(unbiased=False).item(),
            "min": min(values), "max": max(values)}


def _unique_labeled_graphs(input_ids: torch.Tensor, graph_id: torch.Tensor) -> list[dict]:
    """One decoded graph per unique graph_id, in first-occurrence order."""
    seen: set[int] = set()
    graphs = []
    for i in range(input_ids.shape[0]):
        gid = int(graph_id[i])
        if gid in seen:
            continue
        seen.add(gid)
        decoded = tok.decode_input(input_ids[i].tolist())
        assert decoded is not None, "ground-truth input should always be well-formed"
        graphs.append(decoded)
    return graphs


def _unique_unlabeled_graphs(input_ids: torch.Tensor) -> list[dict]:
    """One decoded graph per unique (n, edge-set), in first-occurrence order -- the
    unlabeled corpus has no graph_id, so identity is the graph's own content (same
    convention as data_cache.graph_key)."""
    seen: set = set()
    graphs = []
    for i in range(input_ids.shape[0]):
        decoded = tok.decode_input(input_ids[i].tolist())
        assert decoded is not None, "ground-truth input should always be well-formed"
        key = data_cache.graph_key(decoded["n"], decoded["edges"])
        if key in seen:
            continue
        seen.add(key)
        graphs.append(decoded)
    return graphs


def _graph_connectivity_stats(graphs: list[dict]) -> dict:
    n_list, m_list, deg_list, density_list, diam_list, spl_list = [], [], [], [], [], []
    for decoded in graphs:
        G = nx.Graph()
        G.add_nodes_from(decoded["node_list"])
        G.add_edges_from(decoded["edges"])
        n, m = G.number_of_nodes(), G.number_of_edges()
        n_list.append(n)
        m_list.append(m)
        deg_list.append(2 * m / n if n else 0.0)
        density_list.append(2 * m / (n * (n - 1)) if n > 1 else 0.0)
        diam_list.append(nx.diameter(G))
        spl_list.append(nx.average_shortest_path_length(G))
    return {
        "n_unique_graphs": len(graphs),
        "nodes": _stat(n_list),
        "edges": _stat(m_list),
        "avg_degree": _stat(deg_list),
        "density": _stat(density_list),
        "diameter": _stat(diam_list),
        "avg_shortest_path_length": _stat(spl_list),
    }


def _token_count_stats(input_mask: torch.Tensor, target_ids: "torch.Tensor | None") -> dict:
    out = {"input_tokens": _stat(input_mask.sum(dim=1).tolist())}
    if target_ids is not None:
        target_lens = (target_ids != tok.PAD).sum(dim=1)
        out["target_tokens"] = _stat(target_lens.tolist())
        # <P> + path_nodes + <EOS>, and path_nodes - 1 == diameter for every labeled
        # example by construction (see graphgen.sample_diametric_paths) -- a cheap,
        # decode-free cross-check against the diameter computed from the decoded graphs
        # in _graph_connectivity_stats (the two "diameter" means below should match).
        out["implied_diameter_from_target"] = _stat((target_lens - 3).tolist())
    return out


def compute_split_stats(data_dir: Path, filename: str, labeled: bool) -> dict:
    d = torch.load(data_dir / filename)
    if labeled:
        graphs = _unique_labeled_graphs(d["input_ids"], d["graph_id"])
    else:
        graphs = _unique_unlabeled_graphs(d["input_ids"])
    n_examples = d["input_ids"].shape[0]
    return {
        "filename": filename,
        "n_examples": n_examples,
        "examples_per_graph": n_examples / max(1, len(graphs)),
        "tokens": _token_count_stats(d["input_mask"], d.get("target_ids")),
        "connectivity": _graph_connectivity_stats(graphs),
    }


def _fmt(label: str, s: dict) -> str:
    return f"    {label:<28} mean={s['mean']:8.3f}  std={s['std']:7.3f}  min={s['min']:6.3g}  max={s['max']:6.3g}"


def format_split(name: str, stats: dict) -> str:
    lines = [
        f"== {name} ({stats['filename']}) ==",
        f"  n_examples: {stats['n_examples']}",
        f"  n_unique_graphs: {stats['connectivity']['n_unique_graphs']}",
        f"  examples_per_graph: {stats['examples_per_graph']:.3f}",
        "",
        "  token counts (unpadded, per example):",
        _fmt("input_tokens", stats["tokens"]["input_tokens"]),
    ]
    if "target_tokens" in stats["tokens"]:
        lines.append(_fmt("target_tokens", stats["tokens"]["target_tokens"]))
        lines.append(_fmt("implied_diameter_from_target", stats["tokens"]["implied_diameter_from_target"]))
    lines += [
        "",
        "  graph connectivity (per unique graph):",
        _fmt("n_nodes", stats["connectivity"]["nodes"]),
        _fmt("n_edges", stats["connectivity"]["edges"]),
        _fmt("avg_degree", stats["connectivity"]["avg_degree"]),
        _fmt("density", stats["connectivity"]["density"]),
        _fmt("diameter", stats["connectivity"]["diameter"]),
        _fmt("avg_shortest_path_length", stats["connectivity"]["avg_shortest_path_length"]),
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, default=None, help="defaults to {data_dir}/dataset_stats.txt")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    meta = ds.load_meta(data_dir)

    splits = [
        ("train", "train.pt", True),
        ("val_id", "val_id.pt", True),
        ("val_ood", "val_ood.pt", True),
        ("encoder_pretrain_extra", "encoder_pretrain_extra.pt", False),
    ]

    lines = [
        f"Dataset stats for {data_dir}",
        f"seed={meta['seed']}  vocab_size={meta['vocab_size']}  l_in={meta['l_in']}  l_tgt={meta['l_tgt']}",
        f"id_node_range={meta['id_node_range']}  ood_node_range={meta['ood_node_range']}  "
        f"avg_degree(target)={meta['avg_degree']}",
        "",
    ]
    for name, filename, labeled in splits:
        path = data_dir / filename
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        print(f"computing stats for {name}...")
        stats = compute_split_stats(data_dir, filename, labeled)
        lines.append(format_split(name, stats))

    out_path = Path(args.out) if args.out else data_dir / "dataset_stats.txt"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
