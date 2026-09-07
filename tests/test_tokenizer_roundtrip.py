import pytest

from spelf.dataset import parse_path_text, serialize_condition_text, serialize_target_text
from spelf.graphgen import Graph
from spelf.tokenizer import get_pad_id


def test_get_pad_id_pad_vs_eos(tokenizer):
    assert get_pad_id(tokenizer, "pad") == tokenizer.pad_token_id
    assert get_pad_id(tokenizer, "eos") == tokenizer.eos_token_id
    assert tokenizer.pad_token_id != tokenizer.eos_token_id


def test_serialize_condition_text_format():
    g = Graph(nodes=(1, 3, 5), edges=((1, 3), (3, 5)))
    text = serialize_condition_text(g)
    assert text == "nodes: 1 3 5 edges: 1 - 3 3 - 5 find diametric path"


def test_serialize_target_text_format():
    assert serialize_target_text([2, 5, 7]) == "2 5 7"


def test_serialize_and_parse_path_roundtrip():
    path = [2, 5, 7]
    text = serialize_target_text(path)
    assert parse_path_text(text) == path


def test_parse_path_text_rejects_non_digit_tokens():
    assert parse_path_text("2 - 5") is None
    assert parse_path_text("2 five 7") is None


def test_parse_path_text_empty_is_none():
    assert parse_path_text("") is None
    assert parse_path_text("   ") is None


def test_decode_encode_roundtrip_preserves_text(tokenizer):
    text = "nodes: 1 3 11 12 edges: 1 - 3 3 - 11 11 - 12 find diametric path"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids, skip_special_tokens=True) == text


@pytest.mark.parametrize("node_ids", [(1, 3, 11, 12), (0, 9, 10, 13), (2, 12, 13)])
def test_adjacent_multi_digit_node_ids_stay_separate_tokens(tokenizer, node_ids):
    """Regression test for the spacing requirement: two adjacent node ids in
    the serialized text must never merge into (or bleed digits into) a
    single token, however many digits each has.
    """
    text = " ".join(str(n) for n in node_ids)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # Each node id should decode back to exactly its own digits, in order,
    # not merged with a neighbor.
    decoded_pieces = tokenizer.decode(ids, skip_special_tokens=True).split()
    assert decoded_pieces == [str(n) for n in node_ids]
    # And no single token's digits span across a node-id boundary: every
    # token that contains a digit is itself entirely digits (after
    # stripping the SentencePiece word-boundary marker).
    for tok in tokens:
        stripped = tok.lstrip("▁")  # SentencePiece's "▁" boundary marker
        if any(c.isdigit() for c in stripped):
            assert stripped.isdigit(), f"token {tok!r} mixes digits with non-digits"
