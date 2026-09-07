"""Shared evaluation: decode raw generations back to graphs/paths (via tokenizer, the
single source of truth) and score token accuracy, exact-match, valid-path, and
diametric-optimality rates. Used identically for DLM and ARLM since both expose the
same `model.generate(context, context_mask, **kwargs) -> [B, L_tgt]` interface -- the
only per-model difference is what's in `sample_kwargs`.
"""
from __future__ import annotations

import random
from typing import Optional

import networkx as nx
import torch

from . import common, tokenizer as tok


def _rebuild_graph(decoded_input: dict) -> nx.Graph:
    G = nx.Graph()
    # Node labels are drawn from a pool wider than [0, n) (see graphgen.sample_graph),
    # so the real node set is node_list, not range(n) -- see viz.plot_example's matching
    # comment. Doesn't change any current metric (has_edge/diameter over the real node
    # set are unaffected by extra phantom nodes), but keeping this accurate avoids
    # surprises for anything added later that iterates G's full node set.
    G.add_nodes_from(decoded_input["node_list"])
    G.add_edges_from(decoded_input["edges"])
    return G


def evaluate_generation(input_ids_row: list[int], gen_ids_row: list[int], target_ids_row: list[int]) -> dict:
    """Per-example metrics for one generated sequence vs ground truth."""
    decoded_input = tok.decode_input(input_ids_row)
    assert decoded_input is not None, "ground-truth input should always be well-formed"
    G = _rebuild_graph(decoded_input)
    diameter = nx.diameter(G)

    gen_path = tok.decode_target(gen_ids_row)
    exact_match = gen_ids_row == target_ids_row

    valid = False
    shortest = False
    correct_length = False
    optimal = False
    if gen_path:
        valid = (
            len(set(gen_path)) == len(gen_path)
            and all(node in G for node in gen_path)
            and all(G.has_edge(a, b) for a, b in zip(gen_path, gen_path[1:]))
        )
        if valid:
            # Four increasingly strict conditions, each requiring `valid`:
            #  - shortest: the path is actually the shortest path between its own two
            #    endpoints (no shortcut exists) -- a necessary but not sufficient
            #    condition for being a genuine diametric path, since a path can be
            #    shortest between its endpoints while those endpoints aren't a
            #    diametric pair (their distance is less than the graph's diameter).
            #  - correct_length: the path's edge-length equals the graph's diameter --
            #    also not sufficient alone, since a *non-shortest* walk between two
            #    close-together nodes could coincidentally have `diameter` edges.
            #  - optimal: both at once, i.e. the path is a genuine shortest path
            #    between its endpoints AND that shared length equals the diameter --
            #    equivalent to "the endpoints are a diametric pair and this is a
            #    shortest path between them", which is exactly what a diametric path
            #    is by definition.
            shortest = (len(gen_path) - 1 == nx.shortest_path_length(G, gen_path[0], gen_path[-1]))
            correct_length = (len(gen_path) - 1 == diameter)
            optimal = shortest and correct_length

    token_accuracy = sum(a == b for a, b in zip(gen_ids_row, target_ids_row)) / len(target_ids_row)

    # Same idea as arlm.loss's train-time token_accuracy: mask out positions the target
    # itself pads (<PAD> is trivial to predict once <EOS> has been emitted, and for OOD's
    # longer L_TGT canvas a large fraction of positions are padding -- diluting the plain
    # token_accuracy above well above valid_rate/optimal_rate even for near-miss
    # generations). Masks by the TARGET's <PAD> positions, not the generation's own, so a
    # generation that mispredicts where <EOS>/<PAD> starts is still penalized correctly.
    real_positions = [i for i, t in enumerate(target_ids_row) if t != tok.PAD]
    token_accuracy_nopad = (
        sum(gen_ids_row[i] == target_ids_row[i] for i in real_positions) / len(real_positions)
        if real_positions else 1.0
    )

    return {
        "token_accuracy": token_accuracy,
        "token_accuracy_nopad": token_accuracy_nopad,
        "exact_match": exact_match,
        "valid": valid,
        "shortest": shortest,
        "correct_length": correct_length,
        "optimal": optimal,
        "diameter": diameter,
        "gen_path": gen_path,
        "decoded_input": decoded_input,
    }


def sample_examples(examples: list[dict], k: int) -> list[dict]:
    """A different random k-subset of `examples` on every call -- used to pick which
    examples get logged as wandb images each eval round, instead of always slicing the
    same examples[:k]. Once those first few examples are solved, a fixed slice stops
    showing whether the model is still improving on the rest of the (sub-sampled) val
    set; a fresh random draw each round keeps the visualizations informative. Uses the
    process's global `random` state (already seeded once via common.set_seed at
    startup), so a run's full sequence of picks is still reproducible run-to-run under a
    fixed --seed, it just varies step-to-step within a run -- see viz.py."""
    return random.sample(examples, min(k, len(examples)))


@torch.no_grad()
def run_eval(
    decoder,
    encoder,
    loader,
    device: torch.device,
    sample_kwargs: Optional[dict] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """decoder: DLMDecoder or GPTDecoder (anything with .generate(context, context_mask,
    **sample_kwargs)). encoder: frozen GraphEncoder or T5GraphEncoder (see
    common.encode_context). Returns aggregate rates plus a list of per-example dicts (for
    the caller to pick a few for visualization). Four nested-strictness rates, each
    computed only over generations decoding to a well-formed path (see
    evaluate_generation for exact definitions):
      - valid_rate: decodes to a genuine simple path using real edges of the graph.
      - shortest_rate: is actually the shortest path between its own two endpoints.
      - correct_length_rate: its edge-length equals the graph's diameter.
      - optimal_rate: both at once, i.e. a genuine diametric path -- this is the
        "id_optimal rate" when called on the ID loader (the fraction of graphs for which
        the model found any true diametric path), the primary early-stopping signal."""
    decoder.eval()
    encoder.eval()
    sample_kwargs = sample_kwargs or {}

    totals = {
        "token_accuracy": 0.0, "token_accuracy_nopad": 0.0, "exact_match": 0,
        "valid": 0, "shortest": 0, "correct_length": 0, "optimal": 0, "n": 0,
    }
    examples: list[dict] = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        target_ids = batch["target_ids"].to(device)

        context, context_mask = common.encode_context(encoder, input_ids, input_mask)
        gen = decoder.generate(context, context_mask, **sample_kwargs)

        gen_cpu = gen.cpu().tolist()
        tgt_cpu = target_ids.cpu().tolist()
        inp_cpu = input_ids.cpu().tolist()

        for i in range(len(gen_cpu)):
            m = evaluate_generation(inp_cpu[i], gen_cpu[i], tgt_cpu[i])
            totals["token_accuracy"] += m["token_accuracy"]
            totals["token_accuracy_nopad"] += m["token_accuracy_nopad"]
            totals["exact_match"] += int(m["exact_match"])
            totals["valid"] += int(m["valid"])
            totals["shortest"] += int(m["shortest"])
            totals["correct_length"] += int(m["correct_length"])
            totals["optimal"] += int(m["optimal"])
            totals["n"] += 1
            examples.append({**m, "input_ids": inp_cpu[i], "gen_ids": gen_cpu[i], "target_ids": tgt_cpu[i]})

    n = max(totals["n"], 1)
    return {
        "token_accuracy": totals["token_accuracy"] / n,
        "token_accuracy_nopad": totals["token_accuracy_nopad"] / n,
        "exact_match_rate": totals["exact_match"] / n,
        "valid_rate": totals["valid"] / n,
        "shortest_rate": totals["shortest"] / n,
        "correct_length_rate": totals["correct_length"] / n,
        "optimal_rate": totals["optimal"] / n,
        "n_examples": totals["n"],
        "examples": examples,
    }
