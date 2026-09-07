from spelf import tokenizer as tok


def test_data_sample_example():
    edges = [(3, 7), (1, 4), (7, 4)]
    ids = tok.encode_input(8, edges)
    assert tok.input_length(8, len(edges)) == len(ids)
    decoded = tok.decode_input(ids)
    assert decoded == {
        "n": 8,
        "edges": edges,
        "node_list": list(range(8)),
    }

    path = [3, 7, 4, 5]
    tgt = tok.encode_target(path)
    assert len(tgt) == tok.TARGET_LENGTH
    assert tok.decode_target(tgt) == path


def test_input_roundtrip_with_padding():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    ids = tok.encode_input(4, edges, node_list=[0, 1, 2, 3])
    padded, mask = tok.pad_input(ids, len(ids) + 5)
    assert mask == [True] * len(ids) + [False] * 5
    decoded = tok.decode_input(padded)
    assert decoded["edges"] == edges
    assert decoded["node_list"] == [0, 1, 2, 3]


def test_target_roundtrip_various_lengths():
    for path in ([0, 1], [5, 2, 9, 1, 0], list(range(tok.MAX_NODES))):
        ids = tok.encode_target(path)
        assert len(ids) == tok.TARGET_LENGTH
        assert tok.decode_target(ids) == path


def test_decode_target_rejects_malformed():
    # missing <EOS>
    bad = [tok.P, tok.node_tok(0), tok.node_tok(1)] + [tok.PAD] * (tok.TARGET_LENGTH - 3)
    assert tok.decode_target(bad) is None
    # doesn't start with <P>
    bad2 = [tok.node_tok(0), tok.EOS] + [tok.PAD] * (tok.TARGET_LENGTH - 2)
    assert tok.decode_target(bad2) is None
    # empty
    assert tok.decode_target([]) is None


def test_decode_target_ignores_content_after_eos():
    # Content after <EOS> is truncated, not validated -- see decode_target's own
    # docstring for why (dlm.loss excludes pad positions from both its losses, so
    # nothing trains the DLM on what belongs after <EOS>; a generation is read by
    # finding it, not by checking its tail).
    ids = [tok.P, tok.node_tok(0), tok.EOS, tok.node_tok(2), tok.node_tok(5)]
    assert tok.decode_target(ids) == [0]


def test_decode_input_rejects_malformed():
    assert tok.decode_input([]) is None
    assert tok.decode_input([tok.N]) is None
    # dangling node token after <G>, not part of a complete u,v,<E> triplet
    bad = [tok.G, tok.node_tok(0)]
    assert tok.decode_input(bad) is None
    # edge list missing trailing <E>
    bad2 = [tok.G, tok.node_tok(0), tok.node_tok(1), tok.N, tok.node_tok(0), tok.node_tok(1)]
    assert tok.decode_input(bad2) is None
    # junk after the node list
    bad3 = [tok.G, tok.node_tok(0), tok.node_tok(1), tok.E, tok.N, tok.node_tok(0), tok.node_tok(1), tok.G]
    assert tok.decode_input(bad3) is None


def test_decode_input_empty_graph():
    ids = tok.encode_input(0, [], node_list=[])
    assert tok.decode_input(ids) == {"n": 0, "edges": [], "node_list": []}


def test_node_tok_bounds():
    tok.node_tok(0)
    tok.node_tok(tok.MAX_NODES - 1)
    try:
        tok.node_tok(tok.MAX_NODES)
        assert False, "expected ValueError"
    except ValueError:
        pass
