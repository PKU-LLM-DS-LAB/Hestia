"""Functional form of the HESTIA quantizers.

Notation: ``w`` is the latent weight, ``gamma`` the
(detached) group scale, ``z = w / gamma`` the normalized weight,
``Q`` the codebook and ``tau`` the temperature.
"""

from __future__ import annotations

from typing import Tuple

import torch


def group_reshape(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Size]:
    """Reshape ``w`` to ``[num_groups, group_len]`` for group-wise quantization.

    ``group_size > 0``: contiguous groups along the last (input) dimension;
    ``-1``: one group per row (output channel); ``0``: a single group.
    """
    shape = w.shape
    if group_size > 0:
        if shape[-1] % group_size != 0:
            raise ValueError(
                f"Last dimension {shape[-1]} is not divisible by group_size={group_size}."
            )
        return w.reshape(-1, group_size), shape
    if group_size == -1:
        return w.reshape(-1, shape[-1]), shape
    if group_size == 0:
        return w.reshape(1, -1), shape
    raise ValueError(f"Invalid group_size: {group_size}")


def absmean_scale(w_grouped: torch.Tensor, eps: float) -> torch.Tensor:
    """``gamma = mean(|w|) + eps``, detached from the graph."""
    return (w_grouped.detach().abs().mean(dim=-1, keepdim=True) + eps)


def nearest_code(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map normalized weights to their nearest code.

    For the ternary codebook this equals ``Clip(Round(z), -1, 1)``.
    """
    codebook = codebook.to(device=z.device, dtype=z.dtype)
    idx = (z.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    return codebook[idx]


class _SoftmaxExpectation(torch.autograd.Function):
    """``mu_tau(z) = sum_q q * pi_tau(q | z)``.

    The backward pass uses the closed-form Jacobian
    ``d mu / d z = (2 / tau) * V_tau(z)``, where ``V_tau`` is the code variance
    under ``pi_tau``, so only one value per weight is kept for backward instead
    of the full ``|Q|``-way probability tensor.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, codebook: torch.Tensor, tau: float) -> torch.Tensor:
        q = codebook.to(device=z.device, dtype=torch.float32)
        diff = z.float().unsqueeze(-1) - q
        prob = torch.softmax(-diff.square() / tau, dim=-1)
        mu = (prob * q).sum(dim=-1)
        var = (prob * (q - mu.unsqueeze(-1)).square()).sum(dim=-1)
        ctx.tau = tau
        ctx.save_for_backward(var.to(z.dtype))  # one value per weight
        return mu.to(z.dtype)

    @staticmethod
    def backward(ctx, grad_mu: torch.Tensor):
        (var,) = ctx.saved_tensors
        grad_z = grad_mu.float() * (2.0 / ctx.tau) * var
        return grad_z.to(grad_mu.dtype), None, None


def softmax_expectation(z: torch.Tensor, codebook: torch.Tensor, tau: float) -> torch.Tensor:
    return _SoftmaxExpectation.apply(z, codebook, tau)


def hard_quantize(
    w: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int,
    eps: float = 1e-5,
    ste: bool = False,
) -> torch.Tensor:
    """Hard quantizer ``Q(W) = gamma * nearest(W / gamma)``.

    With ``ste=True`` the backward pass is the identity.
    """
    wg, shape = group_reshape(w, group_size)
    gamma = absmean_scale(wg, eps)
    q = (nearest_code(wg / gamma, codebook) * gamma).reshape(shape).to(w.dtype)
    if ste:
        return w + (q - w).detach()
    return q


def soft_quantize(
    w: torch.Tensor,
    codebook: torch.Tensor,
    tau: float,
    group_size: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Differentiable quantizer ``H(W; tau) = gamma * mu_tau(W / gamma)``."""
    if tau <= 0.0:
        raise ValueError("soft_quantize requires tau > 0; use hard_quantize for tau = 0.")
    wg, shape = group_reshape(w, group_size)
    gamma = absmean_scale(wg, eps)
    out = softmax_expectation(wg / gamma, codebook, tau) * gamma
    return out.reshape(shape).to(w.dtype)
