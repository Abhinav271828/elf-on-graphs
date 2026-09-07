import torch

from spelf.dlm import ELF
from spelf.sampling import (
    decode_batch, generate_samples_single_batch, get_sampling_steps, restore_cond,
)


def _tiny_model(C=12, S=10, vocab=15):
    return ELF(text_encoder_dim=C, max_length=S, hidden_size=24, depth=2, num_heads=2,
               bottleneck_dim=6, num_time_tokens=2, num_self_cond_cfg_tokens=2,
               num_model_mode_tokens=2, vocab_size=vocab)


def test_restore_cond_pins_condition_positions():
    z = torch.zeros(2, 4, 3)
    cond = torch.ones(2, 4, 3) * 7.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]).unsqueeze(-1)
    out = restore_cond(z, cond, mask)
    assert torch.equal(out[0, :2], cond[0, :2])
    assert torch.equal(out[0, 2:], z[0, 2:])
    assert torch.equal(out[1], z[1])


def test_get_sampling_steps_endpoints_and_length():
    steps = get_sampling_steps(8, "uniform", 0.0, 1.0)
    assert steps.shape[0] == 9
    assert steps[0].item() == 0.0 and steps[-1].item() == 1.0
    steps_ln = get_sampling_steps(8, "logit_normal", 0.0, 1.0)
    assert steps_ln.shape[0] == 9
    assert steps_ln[0].item() == 0.0 and steps_ln[-1].item() == 1.0
    assert torch.all(steps_ln[1:] >= steps_ln[:-1])  # sorted


def test_conditional_ode_rollout_and_decode_shapes(tiny_config):
    C, S, vocab = 12, 10, 15
    model = _tiny_model(C, S, vocab).eval()
    tiny_config = tiny_config.__class__(**{**tiny_config.__dict__,
                                            "num_self_cond_cfg_tokens": 2, "self_cond_prob": 0.5,
                                            "denoiser_noise_scale": 1.0, "t_eps": 0.05})
    B = 3
    cond_seq = torch.randn(B, S, C)
    cond_seq_mask = torch.zeros(B, S)
    cond_seq_mask[:, :4] = 1.0
    z = torch.randn(B, S, C)
    generator = torch.Generator().manual_seed(0)
    t_steps = get_sampling_steps(6, "uniform", 0.0, 1.0)

    latent = generate_samples_single_batch(
        model=model, generator=generator, z=z, t_steps=t_steps,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, config=tiny_config,
        cfg_scale=1.0, self_cond_cfg_scale=1.0,
    )
    assert latent.shape == (B, S, C)
    assert torch.isfinite(latent).all()
    # Condition positions must remain exactly the clean condition embedding.
    assert torch.allclose(latent[:, :4], cond_seq[:, :4], atol=1e-5)

    predicted_ids = decode_batch(latent, model, t_steps[-1].item(), tiny_config, self_cond_cfg_scale=1.0)
    assert predicted_ids.shape == (B, S)
    assert predicted_ids.dtype in (torch.int64, torch.long)
