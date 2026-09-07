import networkx as nx
import numpy as np

from spelf import graphgen as gg


def test_sample_graph_always_connected_id_range():
    rng = np.random.default_rng(123)
    for n in range(6, 11):
        for _ in range(30):
            G = gg.sample_graph(n, 3.0, rng)
            assert nx.is_connected(G)
            assert G.number_of_nodes() == n
            assert set(G.nodes()) == set(range(n))


def test_sample_graph_always_connected_ood_range():
    rng = np.random.default_rng(456)
    for n in range(11, 15):
        for _ in range(30):
            G = gg.sample_graph(n, 3.0, rng)
            assert nx.is_connected(G)
            assert G.number_of_nodes() == n


def test_avg_degree_roughly_three():
    rng = np.random.default_rng(789)
    degrees = []
    for _ in range(200):
        n = int(rng.integers(6, 11))
        G = gg.sample_graph(n, 3.0, rng)
        degrees.extend(d for _, d in G.degree())
    avg = sum(degrees) / len(degrees)
    assert 2.0 < avg < 4.5, avg


def test_graph_diameter_path_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert gg.graph_diameter(G) == 4
    assert gg.diametric_pairs(G) == [(0, 4)]


def test_diametric_pairs_multiple_when_tied():
    # cycle of 6 nodes: diameter 3, achieved by 3 antipodal pairs
    G = nx.cycle_graph(6)
    assert gg.graph_diameter(G) == 3
    pairs = gg.diametric_pairs(G)
    assert len(pairs) == 3
    for u, v in pairs:
        assert u < v
        assert nx.shortest_path_length(G, u, v) == 3


def test_sample_diametric_paths_respects_cap():
    rng = np.random.default_rng(0)
    G = nx.cycle_graph(6)
    paths = gg.sample_diametric_paths(G, k=2, rng=rng)
    assert len(paths) == 2
    for p in paths:
        assert len(p) - 1 == 3
        assert len(set(p)) == len(p)  # simple path, no repeats
        for a, b in zip(p, p[1:]):
            assert G.has_edge(a, b)


def test_sample_diametric_paths_returns_all_when_fewer_than_cap():
    rng = np.random.default_rng(0)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    paths = gg.sample_diametric_paths(G, k=5, rng=rng)
    assert len(paths) == 1
    assert paths[0] == [0, 1, 2, 3, 4]


def test_fallback_spanning_tree_is_connected():
    rng = np.random.default_rng(42)
    G = gg._random_spanning_tree_fallback(10, 3.0, rng)
    assert nx.is_connected(G)
    assert G.number_of_nodes() == 10
