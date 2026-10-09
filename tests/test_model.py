import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from hestia import (
    HestiaConfig,
    HestiaLinear,
    HestiaScheduler,
    apply_sensitivity,
    freeze_model,
    get_hestia_layers,
    load_hestia_model,
    quantize_model,
)
from hestia.quantization import hard_quantize


def _tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def test_quantize_model_shares_parameters_and_skips_lm_head():
    model = _tiny_llama()
    q_proj = model.model.layers[0].self_attn.q_proj.weight
    layers = quantize_model(model, HestiaConfig(group_size=32))
    assert len(layers) == 2 * 7
    assert not isinstance(model.lm_head, HestiaLinear)
    assert model.model.layers[0].self_attn.q_proj.weight is q_proj
    assert layers[0].hestia_name == "model.layers.0.self_attn.q_proj"
    assert get_hestia_layers(model) == layers


def test_effective_weight_eq9():
    lin = nn.Linear(16, 8)
    model = nn.Sequential(lin)
    (layer,) = quantize_model(model, HestiaConfig(group_size=0, skip_modules=[]))
    w = layer.weight.detach()
    layer.pressure, layer.temperature = 0.0, 0.3
    assert torch.equal(layer.effective_weight(), layer.weight)
    layer.pressure, layer.temperature = 0.25, 0.0
    layer.eval()
    q = hard_quantize(w, torch.tensor([-1.0, 0.0, 1.0]), 0)
    assert torch.allclose(layer.effective_weight(), 0.75 * w + 0.25 * q)


def test_training_step_and_freeze(tmp_path):
    model = _tiny_llama()
    config = HestiaConfig(group_size=32)
    layers = quantize_model(model, config)
    apply_sensitivity(layers, {l.hestia_name: float(i + 1) for i, l in enumerate(layers)})
    assert all(0.0 < l.sensitivity < 1.0 for l in layers)

    sched = HestiaScheduler.from_config(layers, config, total_steps=10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 128, (2, 16))
    model.train()
    for step in range(10):
        sched.step(step)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        assert torch.isfinite(loss)

    # Saved checkpoints are plain HF checkpoints with latent weights + hestia_config.json.
    model.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    state = torch.load if False else None  # noqa: F841
    model.eval()
    sched.step(10)  # tau = 0, p = 1: hard quantizer
    with torch.no_grad():
        ref_logits = model(input_ids=ids).logits

    loaded = load_hestia_model(str(tmp_path), torch_dtype=torch.float32)
    assert not get_hestia_layers(loaded)
    with torch.no_grad():
        assert torch.allclose(loaded(input_ids=ids).logits, ref_logits, atol=1e-5)

    frozen = freeze_model(model)
    w = frozen.model.layers[0].mlp.up_proj.weight
    for group in w.reshape(-1, 32):  # every group holds {-gamma, 0, +gamma}
        assert len(torch.unique(group.abs())) <= 2


def test_fairy2i_conversion():
    model = _tiny_llama()
    layers = quantize_model(model, HestiaConfig(codebook="fairy2i"))
    assert len(layers) == 14
    ids = torch.randint(0, 128, (1, 8))
    for l in layers:
        l.pressure, l.temperature = 1.0, 0.1
    assert torch.isfinite(model(input_ids=ids).logits).all()


def test_config_roundtrip(tmp_path):
    cfg = HestiaConfig(codebook="int2", group_size=64, alpha=0.2)
    cfg.save_pretrained(tmp_path)
    assert HestiaConfig.from_pretrained(str(tmp_path)) == cfg
    with pytest.raises(ValueError):
        HestiaConfig(method="foo")
    with pytest.raises(ValueError):
        HestiaConfig(compress_ratio=1.0)
