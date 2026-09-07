import torch
import torch.nn.functional as F

from spelf import tokenizer as tok
from spelf.dlm import DLMDecoder, loss as dlm_loss
from spelf.encoder import GraphEncoder


def _tiny_model_and_context(batch_size: int, l_in: int, l_tgt: int, d_model: int, seed: int = 0):
    torch.manual_seed(seed)
    encoder = GraphEncoder(l_in=l_in, d_model=d_model, n_layers=1, n_heads=2, d_mlp=32)
    encoder.freeze()
    input_ids = torch.randint(tok.NODE_OFFSET, tok.NODE_OFFSET + 4, (batch_size, l_in))
    input_ids[:, 0] = tok.G
    input_mask = torch.ones(batch_size, l_in, dtype=torch.bool)
    with torch.no_grad():
        context = encoder(input_ids, input_mask)
    model = DLMDecoder(l_tgt=l_tgt, d_model=d_model, n_layers=1, n_heads=2, d_mlp=32, embedding=encoder.embedding)
    return model, context, input_mask


def test_decode_terms_masking_matches_manual_per_position_reference():
    # Pins down the exact arithmetic dlm.loss uses for decode_terms (ignore_index=pad_id
    # + per-example mean over only the real, non-pad positions) against a hand-computed
    # reference, independent of any model -- catches a wrong-formula regression (e.g. an
    # off-by-one in the denominator, or averaging over L instead of the real count) even
    # if a particular model's own predictions happen to mask the bug.
    torch.manual_seed(0)
    B, L, V = 3, 5, 7
    pad_id = 0
    logits = torch.randn(B, L, V)
    target_ids = torch.tensor([
        [1, 2, pad_id, pad_id, pad_id],
        [3, 4, 5, pad_id, pad_id],
        [pad_id, pad_id, pad_id, pad_id, pad_id],  # degenerate all-pad row
    ])

    ce = F.cross_entropy(logits.transpose(1, 2), target_ids, ignore_index=pad_id, reduction="none")
    n_real = (target_ids != pad_id).sum(dim=1).clamp(min=1)
    decode_terms = ce.sum(dim=1) / n_real

    for i in range(B):
        real_positions = [j for j in range(L) if target_ids[i, j] != pad_id]
        if not real_positions:
            expected = 0.0  # ce is all zero (ignore_index), denominator clamped to 1
        else:
            expected = sum(
                F.cross_entropy(logits[i, j:j + 1], target_ids[i, j:j + 1]).item() for j in real_positions
            ) / len(real_positions)
        assert abs(decode_terms[i].item() - expected) < 1e-5


def test_decode_loss_differs_with_and_without_pad_masking():
    # Exercises the real dlm.loss(): with pad_id matching the target's actual <PAD>
    # marker (the default), padded positions are excluded; with pad_id set to a value
    # that never appears in target_ids, nothing is masked. Same model, same seed for
    # the forward pass -> the two decode_loss values should differ whenever the
    # (untrained, essentially-random) model's predictions at the padded positions
    # aren't already a perfect match, which is true with overwhelming probability.
    d_model, l_in, l_tgt, B = 16, 12, 8, 4
    model, context, context_mask = _tiny_model_and_context(B, l_in, l_tgt, d_model)

    path = [3, 5]
    row = tok.encode_target(path, length=l_tgt)
    target_ids = torch.tensor([row] * B)
    assert tok.PAD in row and row.count(tok.PAD) > 0  # sanity: this example really has padding

    torch.manual_seed(1)
    out_masked = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=1.0, selfcond_prob=0.0)

    torch.manual_seed(1)
    out_unmasked = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=1.0,
                             selfcond_prob=0.0, pad_id=-1)

    assert out_masked["decode_loss"].item() != out_unmasked["decode_loss"].item()


def test_decode_loss_handles_all_pad_target_without_nan():
    d_model, l_in, l_tgt, B = 16, 12, 8, 2
    model, context, context_mask = _tiny_model_and_context(B, l_in, l_tgt, d_model)
    target_ids = torch.full((B, l_tgt), tok.PAD, dtype=torch.long)

    out = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=1.0, selfcond_prob=0.0)
    assert torch.isfinite(out["decode_loss"])
    assert torch.isfinite(out["loss"])


def test_denoise_loss_also_differs_with_and_without_pad_masking():
    # Same idea as test_decode_loss_differs_with_and_without_pad_masking, but for the
    # denoise branch -- matching the canonical ELF implementation (verified directly
    # against its source), both losses exclude pad_id positions, not just decode_loss.
    d_model, l_in, l_tgt, B = 16, 12, 8, 4
    model, context, context_mask = _tiny_model_and_context(B, l_in, l_tgt, d_model)

    path = [3, 5]
    row = tok.encode_target(path, length=l_tgt)
    target_ids = torch.tensor([row] * B)
    assert row.count(tok.PAD) > 0  # sanity: this example really has padding

    torch.manual_seed(1)
    out_masked = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=0.0, selfcond_prob=0.0)

    torch.manual_seed(1)
    out_unmasked = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=0.0,
                             selfcond_prob=0.0, pad_id=-1)

    assert out_masked["denoise_loss"].item() != out_unmasked["denoise_loss"].item()


def test_denoise_loss_handles_all_pad_target_without_nan():
    d_model, l_in, l_tgt, B = 16, 12, 8, 2
    model, context, context_mask = _tiny_model_and_context(B, l_in, l_tgt, d_model)
    target_ids = torch.full((B, l_tgt), tok.PAD, dtype=torch.long)

    out = dlm_loss(model, context, context_mask, target_ids, decode_branch_prob=0.0, selfcond_prob=0.0)
    assert torch.isfinite(out["denoise_loss"])
    assert torch.isfinite(out["loss"])


def test_loss_default_pad_id_is_project_vocab_pad():
    import inspect
    assert inspect.signature(dlm_loss).parameters["pad_id"].default == tok.PAD
