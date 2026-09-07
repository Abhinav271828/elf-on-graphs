from spelf.graphgen import Graph
from spelf.metrics import evaluate_path


def _diamond():
    # 0-1, 0-2, 1-3, 2-3: diameter 2 (e.g. 1-0-2), also 0-3 has two
    # length-2 shortest paths.
    return Graph(nodes=(0, 1, 2, 3), edges=((0, 1), (0, 2), (1, 3), (2, 3)))


def test_optimal_path_is_true_for_a_genuine_diametric_path():
    g = _diamond()
    m = evaluate_path(g, [1, 0, 2], diameter=2)
    assert m == {"valid_path": True, "shortest_path": True, "correct_length": True, "optimal_path": True}


def test_invalid_path_missing_edge():
    g = _diamond()
    m = evaluate_path(g, [1, 2], diameter=2)  # 1-2 is not an edge
    assert m["valid_path"] is False
    assert m["shortest_path"] is False
    assert m["optimal_path"] is False


def test_invalid_path_unknown_node():
    g = _diamond()
    m = evaluate_path(g, [1, 0, 99], diameter=2)
    assert m["valid_path"] is False


def test_invalid_path_repeated_node():
    g = _diamond()
    m = evaluate_path(g, [0, 1, 0], diameter=2)
    assert m["valid_path"] is False


def test_valid_and_shortest_but_wrong_length_fails_correct_length():
    # 6-cycle: 0->2 via 0-1-2 (length 2) is a genuine shortest path, but the
    # graph's diameter is 3, so this output should fail correct_length/
    # optimal_path while still passing valid_path/shortest_path.
    g = Graph(nodes=(0, 1, 2, 3, 4, 5), edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)))
    m = evaluate_path(g, [0, 1, 2], diameter=3)
    assert m["valid_path"] is True
    assert m["shortest_path"] is True   # 0->2 in 2 hops is indeed shortest
    assert m["correct_length"] is False  # length 2 != diameter 3
    assert m["optimal_path"] is False


def test_correct_length_can_be_true_even_when_path_is_invalid():
    g = _diamond()
    m = evaluate_path(g, [1, 2, 3], diameter=2)  # length 2 == diameter, but 1-2 is not an edge
    assert m["valid_path"] is False
    assert m["correct_length"] is True  # length 2 == diameter, checked independently


def test_none_or_short_path_is_all_false():
    g = _diamond()
    assert evaluate_path(g, None, diameter=2) == {
        "valid_path": False, "shortest_path": False, "correct_length": False, "optimal_path": False,
    }
    assert evaluate_path(g, [0], diameter=2)["valid_path"] is False
