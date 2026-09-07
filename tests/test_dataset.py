import random

from spelf.dataset import PathDataset, get_dataloader, get_pad_id
from spelf.data_cache import example_to_dict
from spelf.graphgen import sample_id_example


def _raw_examples(tiny_config, n=6, seed=0):
    rng = random.Random(seed)
    return [example_to_dict(sample_id_example(rng, tiny_config)) for _ in range(n)]


def test_collate_masks_are_consistent(tiny_config, tokenizer):
    raw = _raw_examples(tiny_config)
    dataset = PathDataset(raw, tokenizer)
    loader = get_dataloader(dataset, tiny_config, tokenizer, batch_size=len(raw), shuffle=False)
    batch = next(iter(loader))

    B, L = batch["input_ids"].shape
    assert L == tiny_config.max_length
    pad_id = get_pad_id(tokenizer, tiny_config.pad_token)

    for i in range(B):
        cond_len = int(batch["cond_len"][i])
        cond_mask = batch["cond_seq_mask"][i]
        attn_mask = batch["attention_mask"][i]
        assert cond_mask[:cond_len].sum() == cond_len
        assert cond_mask[cond_len:].sum() == 0
        # attention_mask must cover at least the condition region
        assert attn_mask[:cond_len].min() == 1
        # padded tail (beyond total valid length) must be the pad token
        valid_len = int(attn_mask.sum().item())
        if valid_len < L:
            assert batch["input_ids"][i, valid_len] == pad_id


def test_build_self_attn_cond_masks_shapes():
    import numpy as np
    from spelf.dataset import build_self_attn_cond_masks

    is_cond = np.array([[True, True, False, False]])
    is_valid = np.array([[True, True, True, False]])
    enc_mask, attn_mask, cond_mask = build_self_attn_cond_masks(is_cond, is_valid)
    assert enc_mask.shape == (1, 4, 4)
    # cond token (0) attends only to cond tokens (0,1), not target (2)
    assert enc_mask[0, 0, 0] == 1 and enc_mask[0, 0, 1] == 1 and enc_mask[0, 0, 2] == 0
    # target token (2) attends to everything valid (0,1,2) but not pad (3)
    assert enc_mask[0, 2, 0] == 1 and enc_mask[0, 2, 2] == 1 and enc_mask[0, 2, 3] == 0
    assert (attn_mask == [[1, 1, 1, 0]]).all()
    assert (cond_mask == [[1, 1, 0, 0]]).all()
