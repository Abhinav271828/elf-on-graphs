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

    # content after eos is truncated, not validated -- mirrors decode_target's own
    # relaxation (see that function's docstring for why)
    ids = list(enc["input_ids"])
    eos_pos = ids.index(t5_tokenizer.eos_token_id)
    ids[eos_pos + 1] = 999  # junk, not pad_token_id -- should be ignored, not rejected
    assert t5_encoder.decode_t5_path_ids(t5_tokenizer, ids) == [3, 7, 4]


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

    adapter = t5_encoder.T5SpaceDecoderAdapter(_FakeModel(), t5_tokenizer)
    out = adapter.generate(None, None)
    assert out.shape == (1, tok.TARGET_LENGTH)
    assert tok.decode_target(out[0].tolist()) == path


def test_t5_space_dlm_adapter_falls_back_to_invalid_row_for_garbage():
    t5_tokenizer = _get_t5_tokenizer()

    class _FakeModel:
        def generate(self, context, context_mask, **kwargs):
            return torch.randint(2, t5_tokenizer.vocab_size, (2, 15))

    adapter = t5_encoder.T5SpaceDecoderAdapter(_FakeModel(), t5_tokenizer)
    out = adapter.generate(None, None)
    assert out.shape == (2, tok.TARGET_LENGTH)
    assert out.min() >= 0 and out.max() < tok.VOCAB_SIZE
    for row in out.tolist():
        tok.decode_target(row)  # must not raise, regardless of what it decodes to (or None)


def test_untied_embedding_embed_and_unembed_are_independent_matrices():
    emb = t5_encoder.UntiedEmbedding(vocab_size=21, d_model=16)
    # different shapes by construction (tok_embedding: [vocab, d_model], lm_head.weight:
    # [vocab, d_model] too, but they must be different parameter *objects*/values, not
    # the same tensor tied together the way SharedEmbedding/T5TiedEmbedding are.
    assert emb.tok_embedding.weight is not emb.lm_head.weight
    assert not torch.equal(emb.tok_embedding.weight, emb.lm_head.weight)

    ids = torch.tensor([[1, 2, 3]])
    x = emb(ids)
    logits = emb.unembed(x)
    assert logits.shape == (1, 3, 21)

    # gradients through unembed must not touch tok_embedding at all (no tying)
    x2 = torch.randn(1, 3, 16, requires_grad=True)
    emb.unembed(x2).sum().backward()
    assert emb.tok_embedding.weight.grad is None
    assert emb.lm_head.weight.grad is not None


def test_t5_graph_encoder_has_no_proj_and_untied_embedding():
    encoder = t5_encoder.T5GraphEncoder(model_name="t5-small")
    assert not hasattr(encoder, "proj")
    assert encoder.d_model == encoder.t5.config.d_model
    assert isinstance(encoder.embedding, t5_encoder.UntiedEmbedding)
    assert encoder.context_requires_grad is False


def test_t5_graph_encoder_trainable_params_are_only_its_own_embedding():
    from spelf import common

    encoder = t5_encoder.T5GraphEncoder(model_name="t5-small")
    names = {name for name, _ in common.trainable_parameters(encoder)}
    assert names == {"embedding.tok_embedding.weight", "embedding.lm_head.weight", "embedding.lm_head.bias"}


def test_t5_graph_encoder_embedding_is_sized_to_t5_vocab():
    encoder = t5_encoder.T5GraphEncoder(model_name="t5-small")
    assert encoder.embedding.tok_embedding.num_embeddings == len(encoder.t5_tokenizer)
    assert encoder.embedding.lm_head.out_features == len(encoder.t5_tokenizer)


def test_tokenize_path_targets_prepends_pad_as_start_marker_and_matches_l_tgt_t5():
    encoder = t5_encoder.T5GraphEncoder(model_name="t5-small")
    path = [3, 7, 4, 5]
    target_ids_row = tok.encode_target(path)
    target_ids = torch.tensor([target_ids_row])

    out = encoder.tokenize_path_targets(target_ids, torch.device("cpu"))
    assert out.shape == (1, encoder.l_tgt_t5)
    assert out[0, 0].item() == encoder.t5_tokenizer.pad_token_id

    # dropping the prepended start marker and decoding the rest must recover the path
    recovered = t5_encoder.decode_t5_path_ids(encoder.t5_tokenizer, out[0, 1:].tolist())
    assert recovered == path


def test_t5_space_decoder_adapter_drop_first_token_for_arlm_style_generation():
    t5_tokenizer = _get_t5_tokenizer()
    path = [3, 7, 4]
    text = t5_encoder.path_to_text(path)
    body = t5_tokenizer(text, padding="max_length", max_length=19)["input_ids"]
    # simulate arlm.sample's shape: [start_marker] + body
    raw = torch.tensor([[t5_tokenizer.pad_token_id] + body])

    class _FakeModel:
        def generate(self, context, context_mask, **kwargs):
            return raw

    adapter = t5_encoder.T5SpaceDecoderAdapter(_FakeModel(), t5_tokenizer, drop_first_token=True)
    out = adapter.generate(None, None)
    assert tok.decode_target(out[0].tolist()) == path

    # without drop_first_token, the leading start marker corrupts the decoded text and
    # the path can't be recovered (falls through to the all-<PAD> invalid default)
    adapter_no_drop = t5_encoder.T5SpaceDecoderAdapter(_FakeModel(), t5_tokenizer, drop_first_token=False)
    out_no_drop = adapter_no_drop.generate(None, None)
    assert tok.decode_target(out_no_drop[0].tolist()) is None


def test_arlm_sample_and_loss_accept_custom_sentinel_ids():
    from spelf.arlm import GPTDecoder, loss, sample

    t5_tokenizer = _get_t5_tokenizer()
    d_model, vocab_size = 16, len(t5_tokenizer)
    model = GPTDecoder(l_tgt=6, d_model=d_model, n_layers=1, n_heads=2, d_mlp=32,
                       embedding=t5_encoder.UntiedEmbedding(vocab_size=vocab_size, d_model=d_model))
    context = torch.randn(2, 5, d_model)
    context_mask = torch.ones(2, 5, dtype=torch.bool)

    gen = sample(model, context, context_mask, max_len=6,
                 start_id=t5_tokenizer.pad_token_id, eos_id=t5_tokenizer.eos_token_id,
                 pad_id=t5_tokenizer.pad_token_id)
    assert gen.shape == (2, 6)
    assert (gen[:, 0] == t5_tokenizer.pad_token_id).all()

    target_ids = gen  # any well-shaped tensor works for a loss-plumbing smoke test
    out = loss(model, context, context_mask, target_ids, pad_id=t5_tokenizer.pad_token_id)
    assert "loss" in out and "token_accuracy" in out

    # defaults (no explicit ids passed) must still match this project's own vocab, for
    # backward compatibility with --encoder_kind custom.
    import inspect
    assert inspect.signature(sample).parameters["start_id"].default == tok.P
    assert inspect.signature(sample).parameters["eos_id"].default == tok.EOS
    assert inspect.signature(sample).parameters["pad_id"].default == tok.PAD
