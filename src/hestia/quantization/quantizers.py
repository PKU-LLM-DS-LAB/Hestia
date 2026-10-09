"""Weight quantizer modules.

Every quantizer maps a latent weight tensor and a temperature to a (relaxed)
quantized weight of the same shape:

* :class:`SoftmaxQuantizer` -- HESTIA, ``H(W; tau)`` for ``tau > 0`` and
  the target hard quantizer ``Q(W)`` for ``tau = 0``.
* :class:`STEQuantizer` -- hard round-and-clip with the straight-through
  estimator, i.e. the AbsMean baseline over the same codebook.
* :class:`Fairy2iQuantizer` -- HESTIA over the complex codebook {+-1, +-i}.

When ``tau = 0`` is reached during training the hard quantizer is used with an
identity backward so that the last optimization steps remain well defined.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from hestia.config import HestiaConfig
from hestia.quantization.fairy2i import fairy2i_quantize
from hestia.quantization.functional import hard_quantize, soft_quantize


class WeightQuantizer(nn.Module):
    """Base class. Sub-classes implement :meth:`forward` and :meth:`hard`."""

    def forward(self, weight: torch.Tensor, temperature: float) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def hard(self, weight: torch.Tensor) -> torch.Tensor:
        """Target inference-time quantizer (no gradient path)."""
        raise NotImplementedError  # pragma: no cover


class SoftmaxQuantizer(WeightQuantizer):
    def __init__(self, codebook: List[float], group_size: int = 128, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer(
            "codebook", torch.tensor(sorted(codebook), dtype=torch.float32), persistent=False
        )
        self.group_size = group_size
        self.eps = eps

    def forward(self, weight: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0.0:
            return soft_quantize(weight, self.codebook, temperature, self.group_size, self.eps)
        return hard_quantize(weight, self.codebook, self.group_size, self.eps, ste=self.training)

    @torch.no_grad()
    def hard(self, weight: torch.Tensor) -> torch.Tensor:
        return hard_quantize(weight, self.codebook, self.group_size, self.eps)

    def extra_repr(self) -> str:
        return f"codebook={self.codebook.tolist()}, group_size={self.group_size}"


class STEQuantizer(SoftmaxQuantizer):
    """Hard quantizer with straight-through gradients; ignores the temperature."""

    def forward(self, weight: torch.Tensor, temperature: float) -> torch.Tensor:
        return hard_quantize(weight, self.codebook, self.group_size, self.eps, ste=True)


class Fairy2iQuantizer(WeightQuantizer):
    def __init__(self, steps: int = 2) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, weight: torch.Tensor, temperature: float) -> torch.Tensor:
        return fairy2i_quantize(weight, temperature, self.steps, ste=self.training)

    @torch.no_grad()
    def hard(self, weight: torch.Tensor) -> torch.Tensor:
        return fairy2i_quantize(weight, 0.0, self.steps, ste=False)

    def extra_repr(self) -> str:
        return f"codebook={{+1, +i, -1, -i}}, steps={self.steps}"


def build_quantizer(config: HestiaConfig) -> WeightQuantizer:
    if config.is_complex:
        return Fairy2iQuantizer(steps=config.fairy2i_steps)
    cls = SoftmaxQuantizer if config.method == "hestia" else STEQuantizer
    return cls(config.codebook_values(), group_size=config.group_size, eps=config.scale_eps)
