"""Random connected graph sampling, diametric-pair enumeration, and deterministic
shortest-path tie-breaking for the diametric-path task (find a path whose length
equals the graph's diameter).

Graphs are always connected (by construction), so the diameter and every pairwise
shortest path are always well-defined. Node labels are optionally drawn from a wider
pool than the graph's own size (see `sample_graph`'s `label_pool_size`) so that ID and
OOD graphs differ only in *size*, never in which node-id token values can appear.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np


def _sample_graph_contiguous(n: int, avg_degree: float, rng: np.random.Generator, max_tries: int) -> nx.Graph:
    """Erdos-Renyi graph on nodes 0..n-1, with p tuned for the target average degree,
    reject-sampled until connected. Falls back to a random spanning tree + extra edges
    (which can never fail to be connected) if the retry budget is exhausted -- this
    keeps generation from ever hanging, though in practice p is comfortably above the
    connectivity threshold for n in [6, 14] and the fallback is rarely hit."""
    p = min(1.0, avg_degree / (n - 1))
    for _ in range(max_tries):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        if nx.is_connected(G):
            return G
    return _random_spanning_tree_fallback(n, avg_degree, rng)


def sample_graph(n: int, avg_degree: float, rng: np.random.Generator, max_tries: int = 200,
                  label_pool_size: Optional[int] = None) -> nx.Graph:
    """Sample a connected graph with `n` nodes. If `label_pool_size` is given (e.g. the
    tokenizer's MAX_NODES), the graph's nodes are labeled with a uniformly random
    n-subset of {0, ..., label_pool_size-1} instead of contiguous 0..n-1 -- so every
    node-id *token value* the model can be asked to emit gets exercised by graphs of
    every size, and graph *size* (not label identity) is the only axis that
    distinguishes ID from OOD. If `label_pool_size` is None (default), labels are
    contiguous 0..n-1, matching the graph's own node count."""
    G = _sample_graph_contiguous(n, avg_degree, rng, max_tries)
    if label_pool_size is not None:
        if label_pool_size < n:
            raise ValueError(f"label_pool_size={label_pool_size} < n={n}")
        chosen = rng.choice(label_pool_size, size=n, replace=False)
        G = nx.relabel_nodes(G, {i: int(chosen[i]) for i in range(n)})
    return G


def _random_spanning_tree_fallback(n: int, avg_degree: float, rng: np.random.Generator) -> nx.Graph:
    edges = set()
    for i in range(1, n):
        parent = int(rng.integers(0, i))
        edges.add((parent, i))
    target_edges = round(avg_degree * n / 2)
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
    rng.shuffle(all_possible)
    for e in all_possible:
        if len(edges) >= target_edges:
            break
        edges.add(e)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def shortest_path_lexsmallest(G: nx.Graph, start: int, end: int) -> list[int]:
    """Deterministic shortest path: BFS distances *from end*, then greedily step from
    `start` to the smallest-id neighbor whose distance-to-end is one less than the
    current node's. This yields the lexicographically-smallest shortest path (every
    candidate at each step lies on some shortest path, by the distance invariant, and
    picking the smallest never forecloses a smaller full path later)."""
    if start == end:
        return [start]
    dist = nx.shortest_path_length(G, source=end)
    path = [start]
    current = start
    while current != end:
        candidates = sorted(
            w for w in G.neighbors(current) if dist.get(w) == dist[current] - 1
        )
        assert candidates, f"no progress from {current} toward {end} -- graph disconnected?"
        current = candidates[0]
        path.append(current)
    return path


def graph_diameter(G: nx.Graph) -> int:
    """The graph's diameter (max shortest-path distance over all pairs). Graphs here
    are always connected by construction, so this is always well-defined."""
    return nx.diameter(G)


def diametric_pairs(G: nx.Graph) -> list[tuple[int, int]]:
    """All unordered {u, v} pairs (returned as (u, v) with u < v) whose shortest-path
    distance equals the graph's diameter -- i.e. every pair a "diametric path" could
    legitimately connect. A graph's diameter is always achieved by at least one pair,
    and often by several (e.g. every antipodal pair on an even cycle)."""
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    diam = max(max(d.values()) for d in lengths.values())
    nodes = sorted(G.nodes())
    pairs = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if lengths[u].get(v) == diam:
                pairs.append((u, v))
    return pairs


def sample_diametric_paths(G: nx.Graph, k: int, rng: np.random.Generator) -> list[list[int]]:
    """Up to `k` canonical diametric paths -- one per distinct unordered diametric
    pair, each computed via the deterministic `shortest_path_lexsmallest` tie-break. If
    more than `k` diametric pairs exist, a random k-subset is used; if fewer, all of
    them are returned (so a graph with a unique diametric pair contributes exactly one
    path, not k duplicates). This is how "use multiple paths per graph if they exist"
    is implemented for training data generation."""
    pairs = diametric_pairs(G)
    k = min(k, len(pairs))
    idx = rng.choice(len(pairs), size=k, replace=False)
    chosen = [pairs[i] for i in idx]
    return [shortest_path_lexsmallest(G, u, v) for u, v in chosen]
