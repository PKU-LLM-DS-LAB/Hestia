"""Model-level helpers: convert, attach sensitivities, freeze and load."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from hestia.config import HestiaConfig
from hestia.modules import HestiaLinear
from hestia.quantization import build_quantizer
from hestia.sensitivity import sensitivity_scores

logger = logging.getLogger(__name__)


def _is_skipped(name: str, skip_modules: Sequence[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in skip_modules)


def quantizable_linear_names(model: nn.Module, config: HestiaConfig) -> List[str]:
    """Names of the ``nn.Linear`` modules that :func:`quantize_model` would convert."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, HestiaLinear):
            continue
        if _is_skipped(name, config.skip_modules):
            continue
        if config.is_complex and (module.in_features % 2 or module.out_features % 2):
            logger.warning("Skipping %s: fairy2i needs even in/out features.", name)
            continue
        names.append(name)
    return names


def quantize_model(model: nn.Module, config: HestiaConfig) -> List[HestiaLinear]:
    """Replace the linear layers of ``model`` in place by :class:`HestiaLinear`.

    Parameters are shared with the original layers. The config is attached to
    the model as ``model.hestia_config``.
    """
    modules = dict(model.named_modules())
    layers = []
    for name in quantizable_linear_names(model, config):
        linear = modules[name]
        quantizer = build_quantizer(config).to(linear.weight.device)
        layer = HestiaLinear.from_linear(linear, quantizer, name=name)
        parent_name, _, child_name = name.rpartition(".")
        setattr(modules[parent_name] if parent_name else model, child_name, layer)
        layers.append(layer)
    model.hestia_config = config
    logger.info("Converted %d linear layers to HestiaLinear (%s, %s).", len(layers), config.method, config.codebook)
    return layers


def get_hestia_layers(model: nn.Module) -> List[HestiaLinear]:
    return [m for m in model.modules() if isinstance(m, HestiaLinear)]


def apply_sensitivity(
    layers: Sequence[HestiaLinear],
    traces: Dict[str, float],
    kappa: float = 1.0,
) -> Dict[str, float]:
    """Attach the scores ``s_i`` computed from Hessian traces to the layers.

    Statistics ``mu_h`` / ``sigma_h`` are taken over the quantized linear tensors.
    """
    names = {layer.hestia_name for layer in layers}
    missing = sorted(names - set(traces))
    if missing:
        logger.warning("No Hessian trace for %d layers (global schedule used): %s", len(missing), missing[:5])
    scores = sensitivity_scores({k: v for k, v in traces.items() if k in names}, kappa=kappa)
    for layer in layers:
        layer.sensitivity = scores.get(layer.hestia_name)
    return scores


def freeze_model(model: nn.Module) -> nn.Module:
    """Replace every :class:`HestiaLinear` by an ``nn.Linear`` holding ``Q(W)``."""
    for name, module in list(model.named_modules()):
        if isinstance(module, HestiaLinear):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.to_linear())
    return model


def load_hestia_model(
    model_path: str,
    config: Optional[HestiaConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    freeze: bool = True,
):
    """Load a checkpoint trained with HESTIA for inference.

    The latent weights are loaded with ``transformers`` and quantized with the
    target hard quantizer. ``config`` defaults to ``<model_path>/hestia_config.json``.
    """
    from transformers import AutoModelForCausalLM

    config = config or HestiaConfig.find(model_path)
    if config is None:
        raise FileNotFoundError(f"No hestia_config.json in {model_path}; pass `config` explicitly.")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype or "auto")
    if device is not None:
        model.to(device)
    quantize_model(model, config)
    if freeze:
        freeze_model(model)
    return model.eval()
