import math

import pytest
import torch.nn as nn

from hestia import AnnealingSchedule, HestiaConfig, HestiaScheduler, quantize_model, tensor_temperature


def test_pressure_eq10():
    s = AnnealingSchedule(total_steps=100, compress_ratio=0.2)
    assert s.pressure(0) == 0.0
    assert s.pressure(10) == pytest.approx(0.5)
    assert s.pressure(20) == 1.0 and s.pressure(80) == 1.0
    assert AnnealingSchedule(100, compress_ratio=0.0).pressure(0) == 1.0


def test_base_temperature_eq14():
    s = AnnealingSchedule(total_steps=100, compress_ratio=0.2, init_temp=0.3)
    assert s.base_temperature(0) == s.base_temperature(20) == 0.3
    assert s.base_temperature(60) == pytest.approx(0.15)
    assert s.base_temperature(100) == pytest.approx(0.0, abs=1e-12)
    assert s.base_temperature(150) == pytest.approx(0.0, abs=1e-12)
    temps = [s.base_temperature(t) for t in range(20, 101)]
    assert all(a >= b for a, b in zip(temps, temps[1:]))


def test_tensor_temperature_eq15():
    assert tensor_temperature(0.2, None, 0.4) == 0.2
    assert tensor_temperature(0.2, 0.5, 0.4) == pytest.approx(0.2 * math.exp(0.2))
    # alpha > 0: more sensitive tensors keep a higher temperature.
    assert tensor_temperature(0.2, 0.9, 0.4) > tensor_temperature(0.2, 0.1, 0.4)


def test_scheduler_updates_layers():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    layers = quantize_model(model, HestiaConfig(group_size=0, skip_modules=[]))
    layers[0].sensitivity, layers[1].sensitivity = 0.9, 0.1
    sched = HestiaScheduler.from_config(layers, HestiaConfig(alpha=0.4), total_steps=10)
    state = sched.step(0)
    assert state["hestia/pressure"] == 0.0 and all(l.pressure == 0.0 for l in layers)
    sched.step(6)
    assert layers[0].temperature > layers[1].temperature > 0
    assert sched.step(10)["hestia/temperature"] == pytest.approx(0.0, abs=1e-12)
