"""Quantization-aware linear layer."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hestia.quantization import WeightQuantizer


class HestiaLinear(nn.Linear):
    """Linear layer whose forward weight is ``W_eff = (1 - p) W + p H(W; tau_i)``.

    The latent weight ``W`` is the trainable parameter. The pressure ``p`` and
    the tensor-wise temperature ``tau_i`` are set externally by
    :class:`hestia.schedule.HestiaScheduler` at every optimization step.
    ``sensitivity`` stores the Hessian-trace score ``s_i``.

    A freshly converted layer uses ``p = 1`` and ``tau = 0``, i.e. the target
    hard quantizer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantizer: WeightQuantizer,
        bias: bool = True,
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.quantizer = quantizer
        self.hestia_name = name
        self.sensitivity: Optional[float] = None
        self.pressure: float = 1.0
        self.temperature: float = 0.0

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, quantizer: WeightQuantizer, name: Optional[str] = None
    ) -> "HestiaLinear":
        """Wrap an existing ``nn.Linear`` and *share* its parameters (ZeRO-3 safe)."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            quantizer,
            bias=linear.bias is not None,
            name=name,
            device="meta",
        )
        layer.weight = linear.weight
        layer.bias = linear.bias
        return layer

    def temperature_scale(self, alpha: float) -> float:
        """``exp(alpha * s_i)``; ``1`` if no sensitivity is attached."""
        return 1.0 if self.sensitivity is None else math.exp(alpha * self.sensitivity)

    def effective_weight(self) -> torch.Tensor:
        w = self.weight
        if self.pressure <= 0.0:
            return w
        q = self.quantizer(w, self.temperature)
        return q if self.pressure >= 1.0 else torch.lerp(w, q, self.pressure)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """Return an ``nn.Linear`` holding the hard-quantized (dequantized) weight."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight.copy_(self.quantizer.hard(self.weight))
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, pressure={self.pressure:.3f}, "
            f"temperature={self.temperature:.4f}, sensitivity={self.sensitivity}"
        )
