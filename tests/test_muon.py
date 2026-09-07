import torch
import torch.nn as nn

from spelf.muon import Muon, muon_with_aux_adam, zeropower_via_newtonschulz5


def test_newtonschulz5_preserves_shape_and_is_finite():
    torch.manual_seed(0)
    G = torch.randn(8, 5) * 3.0
    O = zeropower_via_newtonschulz5(G, steps=5)
    assert O.shape == G.shape
    assert torch.isfinite(O).all()


def test_newtonschulz5_preserves_singular_vectors_of_an_orthonormal_matrix():
    # zeropower_via_newtonschulz5(X) = f(X X^T) @ X for a scalar polynomial f
    # (each iteration only mixes in powers of X X^T), so it can only rescale
    # singular values -- it never rotates X's singular vectors. For an
    # already-orthonormal Q, that means the output must stay *parallel* to Q
    # (same directions), even though the initial Frobenius-norm
    # normalization step means its overall scale isn't fixed at 1 after only
    # a few steps.
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(8, 5))
    O = zeropower_via_newtonschulz5(Q, steps=5)
    cos_sim = (O.flatten() @ Q.flatten()) / (O.norm() * Q.norm())
    assert cos_sim.item() > 0.99


def test_newtonschulz5_pulls_singular_values_toward_one():
    torch.manual_seed(1)
    G = torch.randn(8, 5) * 5.0  # deliberately poorly scaled
    sv_before = torch.linalg.svdvals(G)
    O = zeropower_via_newtonschulz5(G, steps=5)
    sv_after = torch.linalg.svdvals(O)
    assert (sv_after - 1.0).abs().mean() < (sv_before - 1.0).abs().mean()


def test_muon_partitions_params_by_ndim():
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3))
    opt = muon_with_aux_adam(model, lr=0.01)
    muon_group = next(g for g in opt.param_groups if g["use_muon"])
    adam_group = next(g for g in opt.param_groups if not g["use_muon"])
    assert all(p.ndim == 2 for p in muon_group["params"])
    assert all(p.ndim != 2 for p in adam_group["params"])
    assert len(muon_group["params"]) == 2   # the two Linear weights
    assert len(adam_group["params"]) == 2   # the two Linear biases


def test_muon_step_reduces_loss_on_a_toy_regression():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 4))
    opt = muon_with_aux_adam(model, lr=0.05)
    x = torch.randn(32, 4)
    target = torch.randn(32, 4)

    def loss_fn():
        return ((model(x) - target) ** 2).mean()

    first_loss = loss_fn().item()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
    last_loss = loss_fn().item()
    assert last_loss < first_loss


def test_muon_handles_bare_in_out_layout_parameter():
    # Mimics dlm.ELF's proj_kernel/unembed_kernel: a bare 2D nn.Parameter
    # stored (in, out), not produced by nn.Linear.
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = nn.Parameter(torch.randn(6, 3) * 0.1)

        def forward(self, x):
            return x @ self.kernel

    model = Toy()
    opt = muon_with_aux_adam(model, lr=0.05)
    x = torch.randn(8, 6)
    target = torch.randn(8, 3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(model.kernel).all()


def test_muon_zero_fills_missing_grad():
    model = nn.Linear(3, 3)
    opt = Muon([dict(params=list(model.parameters())[:1], lr=0.01, momentum=0.95,
                      weight_decay=0.0, use_muon=True, in_out_layout={})], dict(lr=0.01))
    # No backward() called -> grad is None; step() must not raise.
    opt.step()
