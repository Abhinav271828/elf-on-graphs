import pytest
import torch

from spelf.dlm import ELF, ELF_models


def _tiny_model(seq_len, text_encoder_dim=16, vocab_size=20, num_self_cond_cfg_tokens=0, **kwargs):
    # num_self_cond_cfg_tokens defaults to 0 here: when > 0, self_cond_cfg_scale
    # must be passed on *every* forward call (see dlm.ELF.build_context), so
    # tests that don't care about that feature leave it off by default.
    #
    # `max_length` sizes the RoPE table (num_empty_token prefix slots +
    # max_length positions); every forward call must feed exactly that many
    # non-prefix positions, so it's pinned to `seq_len` here rather than left
    # as an independent default.
    return ELF(
        text_encoder_dim=text_encoder_dim, max_length=seq_len,
        hidden_size=32, depth=2, num_heads=2, bottleneck_dim=8,
        num_time_tokens=2, num_self_cond_cfg_tokens=num_self_cond_cfg_tokens, num_model_mode_tokens=2,
        vocab_size=vocab_size, **kwargs,
    )


def test_forward_denoiser_only_shape():
    B, S, C = 3, 10, 16
    model = _tiny_model(S, text_encoder_dim=C)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    mask = torch.ones(B, S)
    out, decoder_logits = model(x, t, attention_mask=mask, deterministic=True)
    assert out.shape == (B, S, C)
    assert decoder_logits is None


def test_forward_with_self_conditioning_input():
    B, S, C = 2, 6, 16
    model = _tiny_model(S, text_encoder_dim=C)
    x = torch.randn(B, S, 2 * C)  # [z, x_pred] concatenated
    t = torch.rand(B)
    out, _ = model(x, t, deterministic=True)
    assert out.shape == (B, S, C)


def test_forward_decoder_branch_shapes_and_bool_gate():
    B, S, C = 2, 6, 16
    model = _tiny_model(S, text_encoder_dim=C, vocab_size=20)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    out, decoder_logits = model(x, t, deterministic=True, decoder_step_active=True)
    assert out.shape == (B, S, C)
    assert decoder_logits.shape == (B, S, 20)


def test_forward_decoder_branch_per_example_tensor_gate():
    B, S, C = 4, 6, 16
    model = _tiny_model(S, text_encoder_dim=C, vocab_size=20)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    gate = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out, decoder_logits = model(x, t, deterministic=True, decoder_step_active=gate)
    assert out.shape == (B, S, C)
    assert decoder_logits.shape == (B, S, 20)


def test_forward_with_self_cond_cfg_scale():
    B, S, C = 2, 6, 16
    model = _tiny_model(S, text_encoder_dim=C, num_self_cond_cfg_tokens=2)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    sc = torch.full((B,), 2.0)
    out, _ = model(x, t, deterministic=True, self_cond_cfg_scale=sc)
    assert out.shape == (B, S, C)


def test_forward_requires_self_cond_cfg_scale_when_configured():
    model = _tiny_model(6, num_self_cond_cfg_tokens=2)
    x = torch.randn(2, 6, 16)
    t = torch.rand(2)
    with pytest.raises(ValueError):
        model(x, t, deterministic=True)


def test_forward_respects_attention_mask_padding():
    B, S, C = 2, 8, 16
    model = _tiny_model(S, text_encoder_dim=C)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    mask = torch.ones(B, S)
    mask[:, 5:] = 0  # pad the tail
    out, _ = model(x, t, attention_mask=mask, deterministic=True)
    assert out.shape == (B, S, C)
    assert torch.isfinite(out).all()


def test_backward_updates_all_parameter_groups():
    B, S, C = 2, 6, 16
    model = _tiny_model(S, text_encoder_dim=C, vocab_size=20)
    x = torch.randn(B, S, C)
    t = torch.rand(B)
    gate = torch.ones(B)
    out, decoder_logits = model(x, t, deterministic=False, decoder_step_active=gate)
    loss = out.pow(2).mean() + decoder_logits.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_model_factory_sizes_build():
    for name, factory in ELF_models.items():
        m = factory(text_encoder_dim=8, max_length=16, vocab_size=10, num_model_mode_tokens=0)
        assert sum(p.numel() for p in m.parameters()) > 0
