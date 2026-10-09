"""Hessian traces -> sensitivity scores and trace file I/O."""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_LEGACY_PREFIX = re.compile(r"^layer_\d+_")


def sensitivity_scores(
    traces: Dict[str, float],
    kappa: float = 1.0,
    eps: float = 1e-8,
    min_trace: float = 1e-12,
) -> Dict[str, float]:
    """``s_i = Sigmoid(kappa * (log h_i - mu_h) / (sigma_h + eps))``.

    ``mu_h`` and ``sigma_h`` are the mean and (population) standard deviation of
    ``log h_i`` over all given tensors. Non-positive trace estimates, which can
    occur for nearly flat tensors because Hutch++ is stochastic, are clamped to
    ``min_trace`` before taking the logarithm.
    """
    if not traces:
        return {}
    clamped = [k for k, h in traces.items() if h <= min_trace]
    if clamped:
        logger.warning("Clamping %d non-positive Hessian traces: %s", len(clamped), clamped[:5])
    logs = {k: math.log(max(h, min_trace)) for k, h in traces.items()}
    mu = sum(logs.values()) / len(logs)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in logs.values()) / len(logs))
    return {k: 1.0 / (1.0 + math.exp(-kappa * (v - mu) / (sigma + eps))) for k, v in logs.items()}


def save_traces(path: str, traces: Dict[str, float], metadata: Optional[dict] = None) -> None:
    payload = {"traces": traces, "metadata": metadata or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_traces(path: str) -> Dict[str, float]:
    """Load ``{module_name: trace}`` from a JSON (or legacy pickle) file.

    Accepts either ``{"traces": {...}}`` or a flat mapping. Legacy keys of the
    form ``layer_<idx>_<module_name>`` are mapped back to ``<module_name>``.
    """
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("traces"), dict):
        data = data["traces"]
    return {_LEGACY_PREFIX.sub("", str(k)): float(v) for k, v in data.items()}
