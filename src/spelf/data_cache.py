"""Orchestrates graph sampling -> diametric-path selection -> tokenization -> cached
tensors + meta.json.

Produces, under an output directory:
  meta.json                 vocab/shape/generation config, shared by all splits
  train.pt                  labeled diametric-path examples (6-10 node graphs)
  val_id.pt                 labeled examples, held-out 6-10 node graphs (in-distribution)
  val_ood.pt                labeled examples, held-out 11-14 node graphs (out-of-distribution)
  encoder_pretrain_extra.pt unlabeled 11-14 node graph inputs (no path), for encoder MLM
                             pretraining only -- concatenated with train.pt's inputs at
                             pretraining time so the encoder sees both node-count ranges
                             without train.pt duplicating any data on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import graphgen, tokenizer as tok

# Seed offsets keep each split's random stream independent and reproducible.
SEED_OFFSETS = {
    "train": 0,
    "val_id": 1_000_000,
    "val_ood": 2_000_000,
    "encoder_pretrain_extra": 3_000_000,
}

ID_NODE_RANGE = (6, 10)
OOD_NODE_RANGE = (11, 14)
AVG_DEGREE = 3.0


def graph_key(n: int, edges: list[tuple[int, int]]) -> tuple:
    """Canonical, order/direction-independent identity for a graph, used to check that
    val splits never reuse a training graph."""
    return (n, frozenset(tuple(sorted(e)) for e in edges))


def _shuffled_edges(edges: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    idx = rng.permutation(len(edges))
    return [edges[i] for i in idx]


def generate_labeled_split(
    num_graphs: int,
    max_paths_per_graph: int,
    node_range: tuple[int, int],
    rng: np.random.Generator,
    avg_degree: float = AVG_DEGREE,
    graph_id_offset: int = 0,
    forbidden_keys: Optional[set] = None,
    max_resample_tries: int = 200,
) -> list[dict]:
    """Each graph contributes one example per distinct diametric pair (a pair of nodes
    whose shortest-path distance equals the graph's diameter), up to
    `max_paths_per_graph` -- so a graph with several equally-long "longest shortest
    paths" trains the model on all of them (subsampled if there are more than the cap),
    while a graph with a unique diametric pair contributes just one example. Edge order
    is independently shuffled per example.

    `forbidden_keys` (canonical graph_key()s, typically the held-out val splits) are
    reject-resampled around: for small n (e.g. 6 nodes has only 15 possible edges), the
    space of likely Erdos-Renyi outcomes is small enough that with tens of thousands of
    draws, exact-graph collisions between train and a few-hundred-graph val split are
    near-guaranteed by the birthday paradox otherwise. Resampling around a reserved set
    this small (a few hundred keys) succeeds within a handful of tries even when the
    underlying space is small; harmless *within-train* duplicates are not addressed
    (and don't need to be -- they don't violate the held-out guarantee)."""
    forbidden_keys = forbidden_keys or set()
    examples = []
    for gi in range(num_graphs):
        n = int(rng.integers(node_range[0], node_range[1] + 1))
        for _ in range(max_resample_tries):
            G = graphgen.sample_graph(n, avg_degree, rng, label_pool_size=tok.MAX_NODES)
            if graph_key(n, list(G.edges())) not in forbidden_keys:
                break
        else:
            raise RuntimeError(
                f"could not sample an n={n} graph outside forbidden_keys after {max_resample_tries} tries"
            )
        edges = list(G.edges())
        node_list = sorted(G.nodes())
        diameter = graphgen.graph_diameter(G)
        paths = graphgen.sample_diametric_paths(G, max_paths_per_graph, rng)
        graph_id = graph_id_offset + gi
        for path in paths:
            examples.append({
                "n": n,
                "edges": _shuffled_edges(edges, rng),
                "node_list": node_list,
                "path": path,
                "diameter": diameter,
                "graph_id": graph_id,
            })
    return examples


def generate_unlabeled_inputs(
    num_graphs: int,
    examples_per_graph: int,
    node_range: tuple[int, int],
    rng: np.random.Generator,
    avg_degree: float = AVG_DEGREE,
    forbidden_keys: Optional[set] = None,
    max_resample_tries: int = 200,
) -> list[dict]:
    """Same shape as generate_labeled_split but without paths -- for encoder MLM
    pretraining data that must never see path/answer supervision. Each graph
    contributes `examples_per_graph` copies with independently shuffled edge order, so
    the encoder sees the same structure described in different token orders. See
    generate_labeled_split's docstring for why `forbidden_keys` matters."""
    forbidden_keys = forbidden_keys or set()
    examples = []
    for gi in range(num_graphs):
        n = int(rng.integers(node_range[0], node_range[1] + 1))
        for _ in range(max_resample_tries):
            G = graphgen.sample_graph(n, avg_degree, rng, label_pool_size=tok.MAX_NODES)
            if graph_key(n, list(G.edges())) not in forbidden_keys:
                break
        else:
            raise RuntimeError(
                f"could not sample an n={n} graph outside forbidden_keys after {max_resample_tries} tries"
            )
        edges = list(G.edges())
        node_list = sorted(G.nodes())
        for _ in range(examples_per_graph):
            examples.append({
                "n": n,
                "edges": _shuffled_edges(edges, rng),
                "node_list": node_list,
            })
    return examples


def max_input_length(*example_lists: list[dict]) -> int:
    best = 0
    for examples in example_lists:
        for ex in examples:
            best = max(best, tok.input_length(ex["n"], len(ex["edges"])))
    return best


def build_labeled_tensors(examples: list[dict], l_in: int) -> dict[str, torch.Tensor]:
    n_ex = len(examples)
    input_ids = torch.zeros(n_ex, l_in, dtype=torch.long)
    input_mask = torch.zeros(n_ex, l_in, dtype=torch.bool)
    target_ids = torch.zeros(n_ex, tok.TARGET_LENGTH, dtype=torch.long)
    n_nodes = torch.zeros(n_ex, dtype=torch.long)
    graph_id = torch.zeros(n_ex, dtype=torch.long)
    for i, ex in enumerate(examples):
        raw = tok.encode_input(ex["n"], ex["edges"], node_list=ex["node_list"])
        padded, mask = tok.pad_input(raw, l_in)
        input_ids[i] = torch.tensor(padded, dtype=torch.long)
        input_mask[i] = torch.tensor(mask, dtype=torch.bool)
        target_ids[i] = torch.tensor(tok.encode_target(ex["path"]), dtype=torch.long)
        n_nodes[i] = ex["n"]
        graph_id[i] = ex["graph_id"]
    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "target_ids": target_ids,
        "n_nodes": n_nodes,
        "graph_id": graph_id,
    }


def build_unlabeled_tensors(examples: list[dict], l_in: int) -> dict[str, torch.Tensor]:
    n_ex = len(examples)
    input_ids = torch.zeros(n_ex, l_in, dtype=torch.long)
    input_mask = torch.zeros(n_ex, l_in, dtype=torch.bool)
    n_nodes = torch.zeros(n_ex, dtype=torch.long)
    for i, ex in enumerate(examples):
        raw = tok.encode_input(ex["n"], ex["edges"], node_list=ex["node_list"])
        padded, mask = tok.pad_input(raw, l_in)
        input_ids[i] = torch.tensor(padded, dtype=torch.long)
        input_mask[i] = torch.tensor(mask, dtype=torch.bool)
        n_nodes[i] = ex["n"]
    return {"input_ids": input_ids, "input_mask": input_mask, "n_nodes": n_nodes}


def assert_disjoint_graphs(name_a: str, examples_a: list[dict], name_b: str, examples_b: list[dict]) -> None:
    keys_a = {graph_key(ex["n"], ex["edges"]) for ex in examples_a}
    keys_b = {graph_key(ex["n"], ex["edges"]) for ex in examples_b}
    overlap = keys_a & keys_b
    assert not overlap, f"{len(overlap)} graph(s) shared between {name_a} and {name_b} -- held-out split contaminated"


def assert_full_node_id_coverage(name: str, examples: list[dict], max_nodes: int) -> None:
    """Every node-id token 0..max_nodes-1 must appear in at least one example's graph --
    otherwise the model would have to generate a token at inference (in OOD graphs) that
    it never saw during training, confounding "generalizes to larger graphs" with
    "generalizes to unseen token values". With `sample_graph(..., label_pool_size=...)`
    drawing labels from the full pool for every split, missing a token here would
    indicate an actual bug rather than expected sampling variance (see this function's
    call site for the back-of-envelope probability)."""
    seen: set[int] = set()
    for ex in examples:
        seen.update(ex["node_list"])
    missing = set(range(max_nodes)) - seen
    assert not missing, f"{name}: node id(s) {sorted(missing)} never appear in any example -- coverage gap"


def generate_all(
    out_dir: Path,
    seed: int,
    n_train_graphs: int,
    n_val_id_graphs: int,
    n_val_ood_graphs: int,
    n_encoder_pretrain_ood_graphs: int,
    max_paths_per_graph: int = 5,
    avg_degree: float = AVG_DEGREE,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rng = np.random.default_rng(seed + SEED_OFFSETS["train"])
    val_id_rng = np.random.default_rng(seed + SEED_OFFSETS["val_id"])
    val_ood_rng = np.random.default_rng(seed + SEED_OFFSETS["val_ood"])
    pretrain_rng = np.random.default_rng(seed + SEED_OFFSETS["encoder_pretrain_extra"])

    # Held-out splits are generated FIRST and their graph identities reserved, so train
    # (and the encoder's unlabeled OOD pretraining corpus) can reject-resample around
    # them. For small n the space of likely graphs is small enough that generating
    # train first and merely *checking* disjointness afterward fails in practice (e.g.
    # n=6 has only 15 possible edges) -- see generate_labeled_split's docstring.
    print(f"Generating val_id split: {n_val_id_graphs} graphs, up to {max_paths_per_graph} diametric paths each...")
    val_id_examples = generate_labeled_split(n_val_id_graphs, max_paths_per_graph, ID_NODE_RANGE, val_id_rng, avg_degree, graph_id_offset=10**9)

    print(f"Generating val_ood split: {n_val_ood_graphs} graphs, up to {max_paths_per_graph} diametric paths each...")
    val_ood_examples = generate_labeled_split(n_val_ood_graphs, max_paths_per_graph, OOD_NODE_RANGE, val_ood_rng, avg_degree, graph_id_offset=2 * 10**9)

    held_out_keys = {graph_key(ex["n"], ex["edges"]) for ex in val_id_examples + val_ood_examples}

    print(f"Generating train split: {n_train_graphs} graphs, up to {max_paths_per_graph} diametric paths each...")
    train_examples = generate_labeled_split(n_train_graphs, max_paths_per_graph, ID_NODE_RANGE, train_rng, avg_degree,
                                              graph_id_offset=0, forbidden_keys=held_out_keys)

    print(f"Generating encoder_pretrain_extra: {n_encoder_pretrain_ood_graphs} unlabeled OOD-sized graphs...")
    pretrain_extra_examples = generate_unlabeled_inputs(n_encoder_pretrain_ood_graphs, max_paths_per_graph, OOD_NODE_RANGE, pretrain_rng,
                                                          avg_degree, forbidden_keys=held_out_keys)

    print("Checking held-out splits don't share graphs with train...")
    assert_disjoint_graphs("train", train_examples, "val_id", val_id_examples)
    assert_disjoint_graphs("train", train_examples, "val_ood", val_ood_examples)
    assert_disjoint_graphs("val_id", val_id_examples, "val_ood", val_ood_examples)
    assert_disjoint_graphs("encoder_pretrain_extra", pretrain_extra_examples, "val_ood", val_ood_examples)

    print("Checking every node-id token appears in the training data...")
    assert_full_node_id_coverage("train", train_examples, tok.MAX_NODES)

    l_in = max_input_length(train_examples, val_id_examples, val_ood_examples, pretrain_extra_examples)
    print(f"Global L_IN = {l_in}, L_TGT = {tok.TARGET_LENGTH}")

    print("Tokenizing and saving tensors...")
    torch.save(build_labeled_tensors(train_examples, l_in), out_dir / "train.pt")
    torch.save(build_labeled_tensors(val_id_examples, l_in), out_dir / "val_id.pt")
    torch.save(build_labeled_tensors(val_ood_examples, l_in), out_dir / "val_ood.pt")
    torch.save(build_unlabeled_tensors(pretrain_extra_examples, l_in), out_dir / "encoder_pretrain_extra.pt")

    meta = {
        "seed": seed,
        "vocab_size": tok.VOCAB_SIZE,
        "max_nodes": tok.MAX_NODES,
        "l_in": l_in,
        "l_tgt": tok.TARGET_LENGTH,
        "id_node_range": list(ID_NODE_RANGE),
        "ood_node_range": list(OOD_NODE_RANGE),
        "avg_degree": avg_degree,
        "max_paths_per_graph": max_paths_per_graph,
        "n_train_graphs": n_train_graphs,
        "n_train_examples": len(train_examples),
        "avg_train_paths_per_graph": len(train_examples) / max(1, n_train_graphs),
        "n_val_id_graphs": n_val_id_graphs,
        "n_val_id_examples": len(val_id_examples),
        "n_val_ood_graphs": n_val_ood_graphs,
        "n_val_ood_examples": len(val_ood_examples),
        "n_encoder_pretrain_ood_graphs": n_encoder_pretrain_ood_graphs,
        "n_encoder_pretrain_extra_examples": len(pretrain_extra_examples),
        "special_tokens": tok.SPECIAL_NAMES,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Wrote {out_dir}")
    return meta
