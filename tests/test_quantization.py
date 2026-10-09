import pytest
import torch

from hestia import HestiaConfig
from hestia.quantization import (
    Fairy2iQuantizer,
    SoftmaxQuantizer,
    STEQuantizer,
    build_quantizer,
    fairy2i_quantize,
    group_reshape,
    hard_quantize,
    soft_quantize,
    softmax_expectation,
)

TERNARY = torch.tensor([-1.0, 0.0, 1.0])


def _reference_expectation(z, codebook, tau):
    """Plain autograd implementation of the Softmax expectation."""
    prob = torch.softmax(-(z.unsqueeze(-1) - codebook).square() / tau, dim=-1)
    return (prob * codebook).sum(-1)


@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_group_reshape(group_size):
    w = torch.randn(6, 8)
    wg, shape = group_reshape(w, group_size)
    expected = {0: (1, 48), -1: (6, 8), 4: (12, 4)}[group_size]
    assert wg.shape == expected and wg.reshape(shape).equal(w)


def test_group_reshape_rejects_indivisible():
    with pytest.raises(ValueError):
        group_reshape(torch.randn(4, 6), 4)


@pytest.mark.parametrize("group_size", [0, -1, 8])
def test_hard_quantize_is_round_and_clip(group_size):
    """AbsMean scale + round-and-clip for the ternary codebook."""
    w = torch.randn(16, 32)
    wg, shape = group_reshape(w, group_size)
    gamma = wg.abs().mean(-1, keepdim=True) + 1e-5
    expected = (gamma * torch.clamp(torch.round(wg / gamma), -1, 1)).reshape(shape)
    assert torch.allclose(hard_quantize(w, TERNARY, group_size), expected)


def test_soft_converges_to_hard():
    """H(w; tau) -> Q(w) as tau -> 0."""
    w = torch.randn(64, 64, dtype=torch.float64)
    hard = hard_quantize(w, TERNARY, 16)
    errs = [(soft_quantize(w, TERNARY, tau, 16) - hard).abs().mean().item() for tau in (0.3, 0.05, 1e-3)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3
    # Away from the boundaries +-gamma/2 the convergence is exponential.
    wg, _ = group_reshape(w, 16)
    z = (wg / (wg.abs().mean(-1, keepdim=True) + 1e-5)).reshape(w.shape)
    far = ((z.abs() - 0.5).abs() > 0.1)
    assert (soft_quantize(w, TERNARY, 1e-3, 16) - hard)[far].abs().max() < 1e-5


@pytest.mark.parametrize("codebook", [[-1.0, 0.0, 1.0], [-1.0, 1.0], [-2.0, -1.0, 0.0, 1.0]])
def test_lemma_4_1_closed_form_gradient(codebook):
    """Custom backward (2/tau) V_tau equals autograd of the Softmax expectation."""
    cb = torch.tensor(codebook, dtype=torch.float64)
    z = (torch.randn(500, dtype=torch.float64) * 1.5).requires_grad_(True)
    tau = 0.3
    out = softmax_expectation(z, cb, tau)
    (g_custom,) = torch.autograd.grad(out.sum(), z)
    z_ref = z.detach().clone().requires_grad_(True)
    ref = _reference_expectation(z_ref, cb, tau)
    (g_ref,) = torch.autograd.grad(ref.sum(), z_ref)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(g_custom, g_ref, atol=1e-6)


def test_gamma_is_detached():
    """dH/dw = (2/tau) V_tau(w/gamma) elementwise, i.e. no gradient through gamma."""
    w = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    tau = 0.2
    (grad,) = torch.autograd.grad(soft_quantize(w, TERNARY.double(), tau, 0).sum(), w)
    z = w.detach() / (w.detach().abs().mean() + 1e-5)
    prob = torch.softmax(-(z.unsqueeze(-1) - TERNARY.double()).square() / tau, -1)
    mu = (prob * TERNARY.double()).sum(-1)
    var = (prob * TERNARY.double().square()).sum(-1) - mu.square()
    assert torch.allclose(grad, 2.0 / tau * var, atol=1e-8)


def test_dead_zone_receives_gradient():
    """Weights inside a hard dead zone still get a non-zero gradient at finite tau."""
    w = torch.tensor([[0.40, 1.0, -1.0, 0.1]], requires_grad=True)
    (g_soft,) = torch.autograd.grad(soft_quantize(w, TERNARY, 0.3, 0)[0, 0], w)
    assert g_soft[0, 0] > 0


def test_quantizer_modules():
    w = torch.randn(8, 16, requires_grad=True)
    soft = SoftmaxQuantizer([-1, 0, 1], group_size=8)
    assert torch.allclose(soft.hard(w), hard_quantize(w, TERNARY, 8))
    soft.train()
    q = soft(w, 0.0)  # tau = 0 in training -> hard forward, STE backward
    q.sum().backward()
    assert torch.allclose(q, soft.hard(w)) and torch.allclose(w.grad, torch.ones_like(w))

    ste = STEQuantizer([-1, 0, 1], group_size=8)
    assert torch.allclose(ste(w, 0.3), soft.hard(w))


def test_fairy2i_quantizer():
    w = torch.randn(8, 12, dtype=torch.float64)
    hard = fairy2i_quantize(w, 0.0, steps=1)
    soft = fairy2i_quantize(w, 1e-4, steps=1)
    assert hard.shape == w.shape
    assert torch.allclose(soft, hard, atol=1e-4)
    q = Fairy2iQuantizer(steps=2)
    w.requires_grad_(True)
    q(w, 0.2).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_build_quantizer():
    assert isinstance(build_quantizer(HestiaConfig()), SoftmaxQuantizer)
    assert isinstance(build_quantizer(HestiaConfig(method="ste")), STEQuantizer)
    assert isinstance(build_quantizer(HestiaConfig(codebook="fairy2i")), Fairy2iQuantizer)
    q = build_quantizer(HestiaConfig(codebook="-2,-1,0,1", group_size=64))
    assert q.codebook.tolist() == [-2.0, -1.0, 0.0, 1.0] and q.group_size == 64
