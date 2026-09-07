import torch

from spelf.t5_encoder import encode_text, load_encoder_profile, save_encoder_profile


def test_encoder_is_frozen(encoder):
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.training is False


def test_encoder_forward_shape_2d_mask(tokenizer, encoder):
    ids = tokenizer("nodes: 0 1 2 edges: 0 - 1 1 - 2 find diametric path", add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids])
    attention_mask = torch.ones_like(input_ids)
    out = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    assert out.shape == (1, len(ids), encoder.d_model)
    assert torch.isfinite(out).all()


def test_encoder_forward_shape_3d_pairwise_mask(encoder):
    # Regression test: our custom cond/target pairwise attention mask
    # (dataset.build_self_attn_cond_masks) is 3D (B, S, S); T5Encoder must
    # accept it (by unsqueezing to the 4D "already prepared" form HF's
    # masking utils expect) rather than crashing.
    B, S = 2, 6
    input_ids = torch.randint(0, 100, (B, S))
    pairwise_mask = torch.ones(B, S, S)
    pairwise_mask[:, :, S - 1] = 0  # last position is padding
    out = encoder(input_ids=input_ids, attention_mask=pairwise_mask, deterministic=True)
    assert out.shape == (B, S, encoder.d_model)
    assert torch.isfinite(out).all()


def test_encode_text_normalizes(tokenizer, encoder):
    ids = tokenizer("nodes: 0 1 2 edges: 0 - 1 find diametric path", add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids])
    attention_mask = torch.ones_like(input_ids)

    raw = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    normalized = encode_text(input_ids, attention_mask, encoder, latent_mean=1.5, latent_std=2.0)
    assert torch.allclose(normalized, (raw - 1.5) / 2.0, atol=1e-5)


def test_save_and_load_encoder_profile_roundtrip(tmp_path):
    path = str(tmp_path / "profile.json")
    save_encoder_profile(path, encoder_model_name="t5-small", tokenizer_name="t5-small",
                          latent_mean=0.25, latent_std=1.75)
    profile = load_encoder_profile(path)
    assert profile == {
        "encoder_model_name": "t5-small", "tokenizer_name": "t5-small",
        "latent_mean": 0.25, "latent_std": 1.75,
    }


def test_load_encoder_profile_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_encoder_profile(str(tmp_path / "does_not_exist.json"))
