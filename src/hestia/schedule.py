"""Compress-stage pressure and Hessian-guided temperature annealing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from hestia.modules import HestiaLinear


@dataclass
class AnnealingSchedule:
    """Global schedules shared by all tensors.

    Args:
        total_steps: ``T``, total number of optimization steps.
        compress_ratio: ``rho``; the compress stage ends at ``T_comp = rho * T``.
        init_temp: ``tau_init``.
    """

    total_steps: int
    compress_ratio: float = 0.2
    init_temp: float = 0.3

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if not 0.0 <= self.compress_ratio < 1.0:
            raise ValueError("compress_ratio must lie in [0, 1).")

    @property
    def compress_steps(self) -> float:
        return self.compress_ratio * self.total_steps

    def pressure(self, step: int) -> float:
        """Compress-stage pressure: ``1`` if ``rho = 0`` else ``min(1, t / (rho T))``."""
        if self.compress_ratio == 0.0:
            return 1.0
        return min(1.0, step / self.compress_steps)

    def base_temperature(self, step: int) -> float:
        """Constant ``tau_init`` during compression, then cosine decay to 0."""
        t_comp = self.compress_steps
        if step <= t_comp:
            return self.init_temp
        progress = min(1.0, (step - t_comp) / (self.total_steps - t_comp))
        return 0.5 * self.init_temp * (1.0 + math.cos(math.pi * progress))


def tensor_temperature(base_temperature: float, sensitivity: Optional[float], alpha: float) -> float:
    """``tau_i(t) = tau_bar(t) * exp(alpha * s_i)``."""
    if sensitivity is None:
        return base_temperature
    return base_temperature * math.exp(alpha * sensitivity)


class HestiaScheduler:
    """Pushes ``p_t`` and ``tau_i(t)`` into every :class:`HestiaLinear` layer."""

    def __init__(
        self,
        layers: Iterable[HestiaLinear],
        compress_ratio: float = 0.2,
        init_temp: float = 0.3,
        alpha: float = 0.4,
        total_steps: Optional[int] = None,
    ) -> None:
        self.layers: List[HestiaLinear] = list(layers)
        self.compress_ratio = compress_ratio
        self.init_temp = init_temp
        self.alpha = alpha
        self.schedule: Optional[AnnealingSchedule] = None
        self._state: Dict[str, float] = {}
        if total_steps is not None:
            self.set_total_steps(total_steps)

    @classmethod
    def from_config(cls, layers: Iterable[HestiaLinear], config, total_steps: Optional[int] = None):
        return cls(layers, config.compress_ratio, config.init_temp, config.alpha, total_steps)

    def set_total_steps(self, total_steps: int) -> None:
        self.schedule = AnnealingSchedule(total_steps, self.compress_ratio, self.init_temp)

    def step(self, step: int) -> Dict[str, float]:
        if self.schedule is None:
            raise RuntimeError("Call set_total_steps() before step().")
        pressure = self.schedule.pressure(step)
        base = self.schedule.base_temperature(step)
        temps = []
        for layer in self.layers:
            layer.pressure = pressure
            layer.temperature = tensor_temperature(base, layer.sensitivity, self.alpha)
            temps.append(layer.temperature)
        self._state = {
            "hestia/pressure": pressure,
            "hestia/temperature": base,
            "hestia/temperature_min": min(temps, default=base),
            "hestia/temperature_max": max(temps, default=base),
        }
        return self._state

    def state(self) -> Dict[str, float]:
        return dict(self._state)
