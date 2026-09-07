import numpy as np

from spelf import data_cache, graphgen as gg, tokenizer as tok


def test_sample_graph_labels_drawn_from_full_pool():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        G = gg.sample_graph(7, 3.0, rng, label_pool_size=tok.MAX_NODES)
        assert G.number_of_nodes() == 7
        assert set(G.nodes()) <= set(range(tok.MAX_NODES))
        seen.update(G.nodes())
    # with 200 independent 7-of-14 draws, every label should appear many times over
    assert seen == set(range(tok.MAX_NODES))


def test_sample_graph_rejects_pool_smaller_than_n():
    rng = np.random.default_rng(0)
    try:
        gg.sample_graph(10, 3.0, rng, label_pool_size=6)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sample_graph_default_labeling_unchanged():
    rng = np.random.default_rng(0)
    G = gg.sample_graph(8, 3.0, rng)
    assert set(G.nodes()) == set(range(8))


def test_id_sized_train_graphs_can_use_high_node_ids():
    rng = np.random.default_rng(1)
    examples = data_cache.generate_labeled_split(
        num_graphs=200, max_paths_per_graph=5, node_range=(6, 10), rng=rng, avg_degree=3.0,
    )
    all_ids = {n for ex in examples for n in ex["node_list"]}
    assert max(all_ids) >= 10, "expected some 6-10-node training graphs to use node-id tokens >= 10"
    data_cache.assert_full_node_id_coverage("test_train", examples, tok.MAX_NODES)
