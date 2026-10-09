"""Configuration of the HESTIA quantizer and annealing schedule.

Defaults correspond to the main ternary setting:
ternary codebook, group size 128, compress-stage ratio rho = 0.2,
base initial temperature tau_init = 0.3, temperature-scaling strength
alpha = 0.4 and sensitivity gain kappa = 1.0.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

CONFIG_NAME = "hestia_config.json"

# Real-valued codebooks Q. The scale gamma is applied on top.
REAL_CODEBOOKS = {
    "binary": [-1.0, 1.0],  # 1-bit
    "ternary": [-1.0, 0.0, 1.0],  # 1.58-bit (main setting)
    "int2": [-2.0, -1.0, 0.0, 1.0],  # 2-bit
}
# Complex-valued codebook {+-1, +-i} used by the Fairy2i pipeline.
COMPLEX_CODEBOOKS = ("fairy2i",)

METHODS = ("hestia", "ste")


@dataclass
class HestiaConfig:
    """Quantization and schedule hyper-parameters.

    Attributes:
        method: ``"hestia"`` uses the temperature-controlled Softmax expectation
            over the codebook; ``"ste"`` is the hard round-and-clip +
            straight-through estimator baseline over the same codebook.
        codebook: Name of a preset codebook (``binary``, ``ternary``, ``int2``,
            ``fairy2i``) or a comma separated list of real code values,
            e.g. ``"-2,-1,0,1"``.
        group_size: Quantization group size. ``>0`` groups of contiguous weights
            along the input dimension, ``-1`` one group per output channel,
            ``0`` one group per tensor.
        scale_eps: epsilon_gamma.
        compress_ratio: Compress-stage fraction rho.
        init_temp: Base initial temperature tau_init.
        alpha: Temperature-scaling strength. ``0`` disables
            Hessian guidance (global schedule).
        kappa: Sensitivity gain.
        skip_modules: Linear modules kept in full precision (matched by full
            module name or by name suffix).
        fairy2i_steps: Number of residual phase-quantization stages for the
            ``fairy2i`` codebook.
    """

    method: str = "hestia"
    codebook: str = "ternary"
    group_size: int = 128
    scale_eps: float = 1e-5
    compress_ratio: float = 0.2
    init_temp: float = 0.3
    alpha: float = 0.4
    kappa: float = 1.0
    skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    fairy2i_steps: int = 2

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    @property
    def is_complex(self) -> bool:
        return self.codebook in COMPLEX_CODEBOOKS

    def codebook_values(self) -> List[float]:
        """Return the real codebook values (not defined for complex codebooks)."""
        if self.is_complex:
            raise ValueError(f"Codebook '{self.codebook}' is complex-valued.")
        if self.codebook in REAL_CODEBOOKS:
            return list(REAL_CODEBOOKS[self.codebook])
        try:
            values = [float(v) for v in self.codebook.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid codebook: {self.codebook!r}") from exc
        return sorted(set(values))

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not self.is_complex and len(self.codebook_values()) < 2:
            raise ValueError("A codebook needs at least two values.")
        if self.is_complex and self.method != "hestia":
            raise ValueError("The fairy2i codebook is only supported with method='hestia'.")
        if self.group_size < -1:
            raise ValueError("group_size must be -1, 0 or a positive integer.")
        if not 0.0 <= self.compress_ratio < 1.0:
            raise ValueError("compress_ratio must lie in [0, 1).")
        if self.init_temp <= 0.0:
            raise ValueError("init_temp must be positive.")
        if self.fairy2i_steps < 1:
            raise ValueError("fairy2i_steps must be >= 1.")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HestiaConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown HestiaConfig fields: {sorted(unknown)}")
        return cls(**data)

    def save_pretrained(self, save_directory: str) -> str:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_pretrained(cls, path: str) -> "HestiaConfig":
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def find(cls, directory: str) -> Optional["HestiaConfig"]:
        """Load ``hestia_config.json`` from ``directory`` if it exists."""
        path = os.path.join(directory, CONFIG_NAME)
        return cls.from_pretrained(path) if os.path.isfile(path) else None
