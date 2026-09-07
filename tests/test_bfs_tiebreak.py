from spelf.graphgen import Graph, bfs_distances, find_diametric_example, shortest_path


def test_shortest_path_basic():
    # 0 - 1 - 2 - 3 (a path graph); shortest 0->3 is unique.
    g = Graph(nodes=(0, 1, 2, 3), edges=((0, 1), (1, 2), (2, 3)))
    assert shortest_path(g.adjacency(), 0, 3) == [0, 1, 2, 3]


def test_shortest_path_tiebreak_picks_lexicographically_smallest():
    # Diamond: 0 connects to 1 and 2, both connect to 3. Two equal-length
    # paths (0-1-3 and 0-2-3) exist; the canonical choice must be the
    # lexicographically smaller one, deterministically.
    g = Graph(nodes=(0, 1, 2, 3), edges=((0, 1), (0, 2), (1, 3), (2, 3)))
    assert shortest_path(g.adjacency(), 0, 3) == [0, 1, 3]


def test_shortest_path_tiebreak_is_deterministic_regardless_of_edge_order():
    edges_a = ((0, 1), (0, 2), (1, 3), (2, 3))
    edges_b = tuple(sorted(edges_a, reverse=True))
    ga = Graph(nodes=(0, 1, 2, 3), edges=edges_a)
    gb = Graph(nodes=(0, 1, 2, 3), edges=edges_b)
    assert shortest_path(ga.adjacency(), 0, 3) == shortest_path(gb.adjacency(), 0, 3)


def test_shortest_path_unreachable_returns_none():
    g = Graph(nodes=(0, 1, 2), edges=((0, 1),))
    assert shortest_path(g.adjacency(), 0, 2) is None


def test_bfs_distances_single_source():
    g = Graph(nodes=(0, 1, 2, 3), edges=((0, 1), (1, 2), (2, 3)))
    dist = bfs_distances(g.adjacency(), 0)
    assert dist == {0: 0, 1: 1, 2: 2, 3: 3}


def test_find_diametric_example_star_graph():
    # Star: center 0, leaves 1,2,3. Diameter is 2, between any two leaves;
    # tie-break picks smallest (s, t) = (1, 2).
    g = Graph(nodes=(0, 1, 2, 3), edges=((0, 1), (0, 2), (0, 3)))
    ex = find_diametric_example(g)
    assert ex.diameter == 2
    assert (ex.source, ex.target) == (1, 2)
    assert ex.path == (1, 0, 2)


def test_find_diametric_example_path_graph():
    g = Graph(nodes=(0, 1, 2, 3, 4), edges=((0, 1), (1, 2), (2, 3), (3, 4)))
    ex = find_diametric_example(g)
    assert ex.diameter == 4
    assert ex.source == 0 and ex.target == 4
    assert ex.path == (0, 1, 2, 3, 4)
