"""Tests for the "node ids drawn from the full OOD universe" design decision
(ARCHITECTURE.md, "Node-id label pool"): even ID-sized graphs must sample
their node labels from `range(ood_max_nodes)`, not just `range(id_max_nodes)`,
so every node-id token/embedding is exercised during ID training.
"""

import random

from spelf.graphgen import sample_id_example


def test_id_examples_cover_the_full_node_universe_over_many_samples(tiny_config):
    rng = random.Random(0)
    seen = set()
    for _ in range(500):
        ex = sample_id_example(rng, tiny_config)
        seen.update(ex.graph.nodes)
    # With enough samples, every id in [0, ood_max_nodes) should appear at
    # least once, including ids >= id_max_nodes (never reachable by node
    # *count* in an ID graph, only by *label*).
    assert seen == set(range(tiny_config.ood_max_nodes))
    assert any(n >= tiny_config.id_max_nodes for n in seen)


def test_id_examples_never_use_more_than_id_max_nodes_labels_at_once(tiny_config):
    rng = random.Random(1)
    for _ in range(100):
        ex = sample_id_example(rng, tiny_config)
        assert len(ex.graph.nodes) <= tiny_config.id_max_nodes
