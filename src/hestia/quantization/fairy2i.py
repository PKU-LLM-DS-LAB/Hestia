"""HESTIA relaxation over the complex codebook {+1, +i, -1, -i} (Fairy2i).

A real linear layer ``A = [[A11, A12], [A21, A22]]`` is rewritten as a
widely-linear complex map with two complex matrices ``U`` and ``W``.
Each complex weight is assigned to one of the four codes by its phase and
scaled by a per-matrix real / imaginary scale. HESTIA replaces the hard phase
assignment by the same temperature-controlled Softmax expectation used in the
real-valued case, with distances measured on the phase circle of
the real-imaginary plane. Several residual stages can be stacked.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Phase centres of the codes +1, +i, -1, -i.
_PHASE_CENTRES = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def _sectors(phase: torch.Tensor):
    """Hard phase sectors of the codes +1, -1, +i, -i."""
    q = math.pi / 4
    real_pos = (phase >= -q) & (phase < q)
    real_neg = (phase >= 3 * q) | (phase < -3 * q)
    imag_pos = (phase >= q) & (phase < 3 * q)
    imag_neg = (phase >= -3 * q) & (phase < -q)
    return real_pos, real_neg, imag_pos, imag_neg


def _scales(w_re: torch.Tensor, w_im: torch.Tensor, on_real: torch.Tensor, on_imag: torch.Tensor):
    """Per-matrix scales of the real-axis and imaginary-axis codes (detached)."""
    zero = w_re.new_zeros((), dtype=torch.float32)
    s_re = w_re.detach()[on_real].abs().float().mean() if on_real.any() else zero
    s_im = w_im.detach()[on_imag].abs().float().mean() if on_imag.any() else zero
    return s_re.clamp(min=1e-6), s_im.clamp(min=1e-6)


def phase_quantize(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    tau: float,
    ste: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One stage of (soft) phase quantization to {+1, +i, -1, -i}."""
    real_pos, real_neg, imag_pos, imag_neg = _sectors(torch.atan2(w_im.detach(), w_re.detach()))
    s_re, s_im = _scales(w_re, w_im, real_pos | real_neg, imag_pos | imag_neg)

    if tau > 0.0:
        phase = torch.atan2(w_im.float(), w_re.float())
        centres = phase.new_tensor(_PHASE_CENTRES)
        delta = phase.unsqueeze(-1) - centres
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))  # wrap to (-pi, pi]
        prob = torch.softmax(-delta.square() / tau, dim=-1)
        q_re = (prob[..., 0] - prob[..., 2]) * s_re
        q_im = (prob[..., 1] - prob[..., 3]) * s_im
        return q_re.to(w_re.dtype), q_im.to(w_im.dtype)

    q_re = (real_pos.float() - real_neg.float()) * s_re
    q_im = (imag_pos.float() - imag_neg.float()) * s_im
    q_re, q_im = q_re.to(w_re.dtype), q_im.to(w_im.dtype)
    if ste:
        q_re = w_re + (q_re - w_re).detach()
        q_im = w_im + (q_im - w_im).detach()
    return q_re, q_im


def residual_phase_quantize(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    tau: float,
    steps: int,
    ste: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack ``steps`` phase-quantization stages on the running residual."""
    q_re = torch.zeros_like(w_re)
    q_im = torch.zeros_like(w_im)
    r_re, r_im = w_re, w_im
    for _ in range(steps):
        d_re, d_im = phase_quantize(r_re, r_im, tau, ste=ste)
        q_re, q_im = q_re + d_re, q_im + d_im
        r_re, r_im = r_re - d_re, r_im - d_im
    return q_re, q_im


def fairy2i_quantize(
    weight: torch.Tensor,
    tau: float,
    steps: int = 2,
    ste: bool = False,
) -> torch.Tensor:
    """Quantize a real ``[2n, 2m]`` weight through its widely-linear complex form."""
    n, m = weight.shape[0] // 2, weight.shape[1] // 2
    a11, a12 = weight[:n, :m], weight[:n, m:]
    a21, a22 = weight[n:, :m], weight[n:, m:]

    u_re, u_im = 0.5 * (a11 + a22), 0.5 * (a21 - a12)
    w_re, w_im = 0.5 * (a11 - a22), 0.5 * (a12 + a21)

    u_re, u_im = residual_phase_quantize(u_re, u_im, tau, steps, ste=ste)
    w_re, w_im = residual_phase_quantize(w_re, w_im, tau, steps, ste=ste)

    top = torch.cat([w_re + u_re, w_im - u_im], dim=1)
    bottom = torch.cat([w_im + u_im, -w_re + u_re], dim=1)
    return torch.cat([top, bottom], dim=0)
