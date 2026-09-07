import random

from spelf.graphgen import is_connected, sample_graph, sample_id_example, sample_ood_example


def test_sample_graph_is_always_connected(tiny_config):
    rng = random.Random(0)
    for _ in range(200):
        g = sample_graph(rng, tiny_config.id_min_nodes, tiny_config.id_max_nodes, tiny_config.ood_max_nodes, tiny_config)
        assert is_connected(g)


def test_sample_graph_respects_node_count_range(tiny_config):
    rng = random.Random(1)
    for _ in range(200):
        g = sample_graph(rng, tiny_config.id_min_nodes, tiny_config.id_max_nodes, tiny_config.ood_max_nodes, tiny_config)
        assert tiny_config.id_min_nodes <= len(g.nodes) <= tiny_config.id_max_nodes


def test_sample_graph_node_ids_within_universe(tiny_config):
    rng = random.Random(2)
    for _ in range(200):
        g = sample_graph(rng, tiny_config.ood_min_nodes, tiny_config.ood_max_nodes, tiny_config.ood_max_nodes, tiny_config)
        assert all(0 <= n < tiny_config.ood_max_nodes for n in g.nodes)
        assert len(set(g.nodes)) == len(g.nodes)  # no duplicate node ids


def test_sample_graph_no_self_loops_or_duplicate_edges(tiny_config):
    rng = random.Random(3)
    for _ in range(100):
        g = sample_graph(rng, tiny_config.id_min_nodes, tiny_config.id_max_nodes, tiny_config.ood_max_nodes, tiny_config)
        assert all(u != v for u, v in g.edges)
        assert len(set(g.edges)) == len(g.edges)
        assert all(u < v for u, v in g.edges)


def test_sample_id_and_ood_examples_have_valid_diameters(tiny_config):
    rng = random.Random(4)
    for _ in range(50):
        ex = sample_id_example(rng, tiny_config)
        assert ex.diameter == len(ex.path) - 1
        assert ex.diameter >= 1
    for _ in range(50):
        ex = sample_ood_example(rng, tiny_config)
        assert tiny_config.ood_min_nodes <= len(ex.graph.nodes) <= tiny_config.ood_max_nodes
