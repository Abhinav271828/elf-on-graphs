import networkx as nx

from spelf import graphgen as gg


def test_matches_data_sample():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(3, 7), (1, 4), (7, 4), (4, 5), (0, 1), (2, 6), (6, 0)])
    assert gg.shortest_path_lexsmallest(G, 3, 5) == [3, 7, 4, 5]


def test_lexicographically_smallest_among_ties():
    # 0-1-3 and 0-2-3 are both shortest (length 2); 1 < 2 so 0-1-3 must win.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    assert gg.shortest_path_lexsmallest(G, 0, 3) == [0, 1, 3]


def test_lexicographic_tiebreak_looks_ahead_correctly():
    # Two length-3 shortest paths from 0 to 4: 0-2-3-4 and 0-1-3-4.
    # Greedy-smallest-first-step must pick 0-1-3-4 (1 < 2), not just any valid one.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    assert gg.shortest_path_lexsmallest(G, 0, 4) == [0, 1, 3, 4]


def test_trivial_self_path():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    assert gg.shortest_path_lexsmallest(G, 0, 0) == [0]


def test_single_edge():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    assert gg.shortest_path_lexsmallest(G, 0, 1) == [0, 1]
