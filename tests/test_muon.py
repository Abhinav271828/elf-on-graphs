import torch
import torch.nn as nn

from spelf import common
from spelf.muon import Muon, _newton_schulz5


def test_newton_schulz5_orthogonalizes_relative_to_raw_input():
    # This quintic iteration (fixed coefficients, bf16 arithmetic -- see the function's
    # docstring) converges to "good enough" orthogonality within 1-2 steps and then
    # oscillates around that floor rather than monotonically improving with more steps
    # (verified empirically -- error at steps=5 is not reliably smaller than at
    # steps=1-2), so the property worth checking is that Muon's default (5 steps)
    # dramatically improves orthogonality relative to the untouched raw gradient, not
    # that more steps beat fewer.
    torch.manual_seed(0)
    for shape in [(8, 8), (4, 16), (16, 4)]:
        G = torch.randn(*shape)
        k = min(shape)

        def orthogonality_error(mat: torch.Tensor) -> float:
            gram = mat @ mat.mT if shape[0] <= shape[1] else mat.mT @ mat
            return (gram - torch.eye(k)).norm().item()

        raw_error = orthogonality_error(G)
        ns_error = orthogonality_error(_newton_schulz5(G, steps=5).float())
        assert ns_error < raw_error * 0.1


def test_muon_rejects_1d_params():
    p = nn.Parameter(torch.randn(8))
    try:
        Muon([p])
        assert False, "expected ValueError for a 1D parameter"
    except ValueError:
        pass


def test_muon_reduces_loss_on_toy_regression():
    torch.manual_seed(0)
    linear = nn.Linear(6, 6, bias=False)
    target = torch.randn(6, 6)
    opt = Muon(linear.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(32, 6)
    y = x @ target.t()

    def loss_fn():
        return ((linear(x) - y) ** 2).mean()

    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0] * 0.3


def test_is_muon_eligible_partition_rules():
    linear_w = nn.Parameter(torch.randn(8, 8))
    linear_b = nn.Parameter(torch.randn(8))
    emb_w = nn.Parameter(torch.randn(21, 128))
    pos_emb_w = nn.Parameter(torch.randn(16, 128))
    mode_emb_w = nn.Parameter(torch.randn(2, 128))
    null_context = nn.Parameter(torch.randn(1, 1, 128))
    norm_w = nn.Parameter(torch.randn(128))

    assert common._is_muon_eligible("layers.0.mlp.0.weight", linear_w) is True
    assert common._is_muon_eligible("out_proj.weight", linear_w) is True
    assert common._is_muon_eligible("layers.0.mlp.0.bias", linear_b) is False
    assert common._is_muon_eligible("embedding.embedding.weight", emb_w) is False
    assert common._is_muon_eligible("pos_enc.pos_emb.weight", pos_emb_w) is False
    assert common._is_muon_eligible("mode_emb.weight", mode_emb_w) is False
    assert common._is_muon_eligible("null_context", null_context) is False
    assert common._is_muon_eligible("out_ln.weight", norm_w) is False


def test_build_dlm_optimizer_covers_every_trainable_param_exactly_once():
    from spelf.dlm import DLMDecoder
    from spelf.encoder import GraphEncoder

    encoder = GraphEncoder(l_in=20, d_model=16, n_layers=1, n_heads=2, d_mlp=32)
    encoder.freeze()
    decoder = DLMDecoder(l_tgt=8, d_model=16, n_layers=1, n_heads=2, d_mlp=32, embedding=encoder.embedding)

    optimizer = common.build_dlm_optimizer([decoder, encoder], adamw_lr=1e-4, muon_lr=0.02)
    muon_params = list(optimizer.optimizers["muon"].param_groups[0]["params"])
    adamw_params = [p for g in optimizer.optimizers["adamw"].param_groups for p in g["params"]]

    all_ids = {id(p) for p in muon_params} | {id(p) for p in adamw_params}
    assert len(muon_params) + len(adamw_params) == len(all_ids), "a param was assigned to both groups"

    expected_trainable = {id(p) for _, p in common.trainable_parameters([decoder, encoder])}
    assert all_ids == expected_trainable

    # null_context and both embedding-ish tables must never end up in the muon group.
    muon_ids = {id(p) for p in muon_params}
    assert id(decoder.null_context) not in muon_ids
    assert id(decoder.pos_enc.pos_emb.weight) not in muon_ids
    assert id(decoder.mode_emb.weight) not in muon_ids
    # but real hidden weight matrices should be there.
    assert id(decoder.out_proj.weight) in muon_ids
    assert id(decoder.input_proj.weight) in muon_ids


def test_multi_optimizer_state_dict_roundtrip():
    lin = nn.Linear(4, 4, bias=False)
    bias_like = nn.Parameter(torch.randn(4))
    opt = common.MultiOptimizer({
        "muon": Muon(lin.parameters(), lr=0.02),
        "adamw": torch.optim.AdamW([bias_like], lr=1e-3),
    })
    x = torch.randn(5, 4)
    loss = (lin(x) ** 2).sum() + (bias_like ** 2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    state = opt.state_dict()
    assert set(state.keys()) == {"muon", "adamw"}

    lin2 = nn.Linear(4, 4, bias=False)
    lin2.weight.data.copy_(lin.weight.data)  # avoid re-triggering a different random step
    bias_like2 = nn.Parameter(bias_like.data.clone())
    opt2 = common.MultiOptimizer({
        "muon": Muon(lin2.parameters(), lr=0.02),
        "adamw": torch.optim.AdamW([bias_like2], lr=1e-3),
    })
    opt2.load_state_dict(state)
    buf1 = opt.optimizers["muon"].state[lin.weight]["momentum_buffer"]
    buf2 = opt2.optimizers["muon"].state[lin2.weight]["momentum_buffer"]
    assert torch.equal(buf1, buf2)


def test_build_multi_lr_schedule_shares_warmup_shape_across_optimizers():
    lin = nn.Linear(4, 4, bias=False)
    bias_like = nn.Parameter(torch.randn(4))
    optimizer = common.MultiOptimizer({
        "muon": Muon(lin.parameters(), lr=0.02),
        "adamw": torch.optim.AdamW([bias_like], lr=1e-4),
    })
    scheduler = common.build_multi_lr_schedule(optimizer, warmup_steps=10, total_steps=100)

    lrs0 = scheduler.get_last_lr()
    assert set(lrs0.keys()) == {"muon", "adamw"}
    assert lrs0["muon"][0] == 0.0 and lrs0["adamw"][0] == 0.0  # step 0, warmup not yet advanced

    for _ in range(10):
        scheduler.step()
    lrs_after_warmup = scheduler.get_last_lr()
    # both should have reached (approximately) their own peak lr at the same relative point
    assert abs(lrs_after_warmup["muon"][0] - 0.02) < 1e-6
    assert abs(lrs_after_warmup["adamw"][0] - 1e-4) < 1e-9
