import re

import torch

from spelf import t5_encoder, tokenizer as tok


def _get_t5_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("t5-small")


def _no_token_spans_two_numbers(t5_tokenizer, text: str) -> bool:
    number_spans = [m.span() for m in re.finditer(r"\d+", text)]
    enc = t5_tokenizer(text, return_offsets_mapping=True)
    for start, end in enc["offset_mapping"]:
        if start == end:
            continue
        overlapping = [i for i, (s, e) in enumerate(number_spans) if start < e and end > s]
        if len(overlapping) > 1:
            return False
    return True


def test_graph_to_text_keeps_every_node_id_in_its_own_token_boundary():
    t5_tokenizer = _get_t5_tokenizer()
    decoded = {
        "n": 6,
        "edges": [(3, 7), (1, 4), (7, 4), (4, 5), (0, 1), (2, 6), (6, 0), (10, 13), (11, 12)],
        "node_list": [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
    }
    text = t5_encoder.graph_to_text(decoded)
    assert _no_token_spans_two_numbers(t5_tokenizer, text)


def test_path_to_text_keeps_every_node_id_in_its_own_token_boundary():
    t5_tokenizer = _get_t5_tokenizer()
    for path in [[3, 7, 4, 5], [0, 6, 7], [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]:
        text = t5_encoder.path_to_text(path)
        assert _no_token_spans_two_numbers(t5_tokenizer, text)


def test_unspaced_format_would_have_fused_tokens_regression_check():
    # Sanity check that the property above is actually meaningful, not vacuously true --
    # the *old* unspaced "u-v" edge format really does fuse distinct node ids into one
    # token for some inputs (verified directly against the real tokenizer), which is
    # exactly the failure mode graph_to_text/path_to_text's spacing avoids.
    t5_tokenizer = _get_t5_tokenizer()
    unspaced = "Graph with nodes: 0 1 4. Edges: 1-4, 4-1."
    assert not _no_token_spans_two_numbers(t5_tokenizer, unspaced)


def test_decode_t5_path_text_round_trips_through_real_tokenizer():
    t5_tokenizer = _get_t5_tokenizer()
    for path in [[3, 7, 4, 5], [0], [13, 0, 6]]:
        text = t5_encoder.path_to_text(path)
        ids = t5_tokenizer(text)["input_ids"]
        decoded_text = t5_tokenizer.decode(ids[:-1], skip_special_tokens=False)  # strip eos
        assert t5_encoder.decode_t5_path_text(decoded_text) == path


def test_decode_t5_path_text_rejects_malformed():
    assert t5_encoder.decode_t5_path_text("not a path at all") is None
    assert t5_encoder.decode_t5_path_text("Path : 3 - 7") is None  # missing trailing "."
    assert t5_encoder.decode_t5_path_text("3 - 7 .") is None  # missing "Path :" prefix


def test_decode_t5_path_ids_handles_eos_and_padding():
    t5_tokenizer = _get_t5_tokenizer()
    text = t5_encoder.path_to_text([3, 7, 4])
    enc = t5_tokenizer(text, padding="max_length", max_length=20)
    path = t5_encoder.decode_t5_path_ids(t5_tokenizer, enc["input_ids"])
    assert path == [3, 7, 4]

    # junk after eos (not just padding) must be rejected, mirroring decode_target
    ids = list(enc["input_ids"])
    eos_pos = ids.index(t5_tokenizer.eos_token_id)
    ids[eos_pos + 1] = 999  # something that isn't pad_token_id
    assert t5_encoder.decode_t5_path_ids(t5_tokenizer, ids) is None


def test_compute_l_tgt_t5_covers_every_random_path():
    t5_tokenizer = _get_t5_tokenizer()
    l_tgt_t5 = t5_encoder.compute_l_tgt_t5(t5_tokenizer, margin=0)
    import random
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(1, tok.MAX_NODES)
        path = rng.sample(range(tok.MAX_NODES), n)
        text = t5_encoder.path_to_text(path)
        n_tokens = len(t5_tokenizer(text)["input_ids"])
        assert n_tokens <= l_tgt_t5, (path, n_tokens, l_tgt_t5)


def test_t5_tied_embedding_matches_underlying_t5_table():
    from transformers import T5EncoderModel

    t5 = T5EncoderModel.from_pretrained("t5-small")
    wrapped = t5_encoder.T5TiedEmbedding(t5.get_input_embeddings())
    ids = torch.tensor([[5, 100, 2000]])
    assert torch.equal(wrapped(ids), t5.get_input_embeddings()(ids))
    x = torch.randn(1, 3, t5.config.d_model)
    assert torch.equal(wrapped.unembed(x), x @ t5.get_input_embeddings().weight.t())
    # no parameters of its own -- it's a view onto T5's own (already-frozen) embedding
    assert list(wrapped.parameters(recurse=False)) == []


def test_t5_diffusion_encoder_has_zero_trainable_parameters():
    from spelf import common

    encoder = t5_encoder.T5DiffusionEncoder(model_name="t5-small")
    assert list(common.trainable_parameters(encoder)) == []
    assert encoder.trainable_state_dict() == {}


def test_t5_space_dlm_adapter_round_trips_a_well_formed_generation():
    t5_tokenizer = _get_t5_tokenizer()
    l_tgt_t5 = t5_encoder.compute_l_tgt_t5(t5_tokenizer)
    path = [3, 7, 4, 5]
    text = t5_encoder.path_to_text(path)
    enc = t5_tokenizer(text, padding="max_length", max_length=l_tgt_t5, return_tensors="pt")

    class _FakeModel:
        def generate(self, context, context_mask, **kwargs):
            return enc["input_ids"]

    adapter = t5_encoder.T5SpaceDLMAdapter(_FakeModel(), t5_tokenizer)
    out = adapter.generate(None, None)
    assert out.shape == (1, tok.TARGET_LENGTH)
    assert tok.decode_target(out[0].tolist()) == path


def test_t5_space_dlm_adapter_falls_back_to_invalid_row_for_garbage():
    t5_tokenizer = _get_t5_tokenizer()

    class _FakeModel:
        def generate(self, context, context_mask, **kwargs):
            return torch.randint(2, t5_tokenizer.vocab_size, (2, 15))

    adapter = t5_encoder.T5SpaceDLMAdapter(_FakeModel(), t5_tokenizer)
    out = adapter.generate(None, None)
    assert out.shape == (2, tok.TARGET_LENGTH)
    assert out.min() >= 0 and out.max() < tok.VOCAB_SIZE
    for row in out.tolist():
        tok.decode_target(row)  # must not raise, regardless of what it decodes to (or None)
