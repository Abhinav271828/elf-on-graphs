"""Erdos-Renyi graph generation, BFS shortest paths, and diameter search.

All graph math here is deterministic given a `random.Random` instance, and
every "pick one among several equally valid answers" choice (which shortest
path to report as ground truth, which diametric pair to use when several
achieve the diameter) is resolved by an explicit, testable tie-break rule
rather than left to iteration-order accident. See ARCHITECTURE.md
("Graph generation & ground truth") for the full rationale.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from .common import Config


@dataclasses.dataclass(frozen=True)
class Graph:
    nodes: Tuple[int, ...]                 # sorted node ids, drawn from the full node universe
    edges: Tuple[Tuple[int, int], ...]      # sorted (u, v) with u < v

    def adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        for n in adj:
            adj[n].sort()
        return adj


@dataclasses.dataclass(frozen=True)
class DiametricExample:
    graph: Graph
    source: int
    target: int
    path: Tuple[int, ...]   # canonical (lexicographically-smallest) diametric path
    diameter: int            # == len(path) - 1, edges


def bfs_distances(adjacency: Dict[int, List[int]], source: int) -> Dict[int, int]:
    """Unweighted single-source shortest-path distances."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def shortest_path(adjacency: Dict[int, List[int]], source: int, target: int) -> Optional[List[int]]:
    """The lexicographically-smallest shortest path from source to target.

    Among all min-length source-target paths, greedily walk from `source`
    always choosing the smallest-id neighbor whose distance-to-target is one
    less than the current node's -- this is a well-defined canonical choice
    (independent of adjacency build order) so ground-truth data is
    reproducible. Returns None if target is unreachable from source.
    """
    dist_to_target = bfs_distances(adjacency, target)
    if source not in dist_to_target:
        return None
    path = [source]
    cur = source
    while cur != target:
        remaining = dist_to_target[cur]
        candidates = [v for v in adjacency[cur] if dist_to_target.get(v) == remaining - 1]
        nxt = min(candidates)
        path.append(nxt)
        cur = nxt
    return path


def is_connected(graph: Graph) -> bool:
    if not graph.nodes:
        return True
    adj = graph.adjacency()
    seen = bfs_distances(adj, graph.nodes[0])
    return len(seen) == len(graph.nodes)


def find_diametric_example(graph: Graph) -> DiametricExample:
    """Find the diameter and a canonical diametric (source, target, path).

    Tie-break across all (s, t) pairs achieving the diameter: smallest s,
    then smallest t (both scanned in sorted node order), so the choice is
    deterministic and independent of adjacency/BFS traversal order.
    """
    adj = graph.adjacency()
    nodes = graph.nodes
    all_dist = {s: bfs_distances(adj, s) for s in nodes}

    best_len, best_s, best_t = -1, None, None
    for s in nodes:
        for t in nodes:
            if t == s:
                continue
            d = all_dist[s][t]
            if d > best_len:
                best_len, best_s, best_t = d, s, t

    assert best_s is not None and best_t is not None, "graph must have >=2 nodes"
    path = shortest_path(adj, best_s, best_t)
    assert path is not None and len(path) - 1 == best_len
    return DiametricExample(graph=graph, source=best_s, target=best_t,
                             path=tuple(path), diameter=best_len)


def _edge_probability(rng, n: int, config: Config) -> float:
    if n <= 1:
        return 0.0
    threshold = math.log(n) / n
    factor = rng.uniform(config.edge_prob_min_factor, config.edge_prob_max_factor)
    p = factor * threshold
    return min(max(p, config.edge_prob_floor), config.edge_prob_ceil)


def sample_graph(rng, n_min: int, n_max: int, universe_size: int, config: Config) -> Graph:
    """Sample a connected Erdos-Renyi graph G(n, p) with node ids drawn from
    `range(universe_size)`.

    Standard G(n, p) sampling can produce a disconnected graph; since a path
    task requires connectivity (undefined diameter otherwise), we use
    rejection sampling: redraw p and re-sample edges until connected, up to
    `config.max_connect_attempts`. p is drawn per-attempt from a band above
    the Erdos-Renyi connectivity threshold ln(n)/n, so rejections are rare in
    practice. Graphs whose edge count exceeds `config.max_edges` are also
    rejected, to bound the serialized sequence length.
    """
    n = rng.randint(n_min, n_max)
    nodes = tuple(sorted(rng.sample(range(universe_size), n)))

    for _ in range(config.max_connect_attempts):
        p = _edge_probability(rng, n, config)
        edges = [(nodes[i], nodes[j])
                 for i in range(n) for j in range(i + 1, n)
                 if rng.random() < p]
        if len(edges) > config.max_edges:
            continue
        graph = Graph(nodes=nodes, edges=tuple(sorted(edges)))
        if is_connected(graph):
            return graph

    raise RuntimeError(
        f"Failed to sample a connected graph with n={n} in "
        f"{config.max_connect_attempts} attempts; widen edge_prob_* bounds."
    )


def sample_id_example(rng, config: Config) -> DiametricExample:
    graph = sample_graph(rng, config.id_min_nodes, config.id_max_nodes, config.ood_max_nodes, config)
    return find_diametric_example(graph)


def sample_ood_example(rng, config: Config) -> DiametricExample:
    graph = sample_graph(rng, config.ood_min_nodes, config.ood_max_nodes, config.ood_max_nodes, config)
    return find_diametric_example(graph)
