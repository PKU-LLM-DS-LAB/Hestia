import json
import math
import pickle

import pytest
import torch
import torch.nn as nn

from hestia.sensitivity import (
    estimate_hessian_traces,
    hutchpp_trace,
    load_traces,
    save_traces,
    sensitivity_scores,
)


def test_hutchpp_exact_for_low_rank():
    torch.manual_seed(0)
    u = torch.randn(200, 5)
    a = u @ u.T
    est = hutchpp_trace(lambda v: a @ v, 200, num_sketch=10, num_query=20)
    assert est == pytest.approx(torch.trace(a).item(), rel=1e-4)


def test_hutchpp_full_rank_estimate():
    torch.manual_seed(0)
    u = torch.randn(300, 300)
    a = u @ u.T / 300 + torch.diag(torch.linspace(0, 10, 300))
    est = hutchpp_trace(lambda v: a @ v, 300, num_sketch=30, num_query=60)
    assert est == pytest.approx(torch.trace(a).item(), rel=0.1)


class _Toy(nn.Module):
    """Two linear layers with a squared-error loss; Hessians are known in closed form."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)

    def forward(self, x, y):
        loss = 0.5 * (self.fc2(self.fc1(x)) - y).square().sum(-1).mean()
        return type("Out", (), {"loss": loss})()


def test_estimator_matches_exact_hessian_trace():
    torch.manual_seed(0)
    model = _Toy().double()
    batches = [{"x": torch.randn(16, 4, dtype=torch.float64), "y": torch.randn(16, 2, dtype=torch.float64)}]
    # Full rank sketch -> exact traces.
    traces = estimate_hessian_traces(model, batches, num_sketch=100, num_query=0)

    x, y = batches[0]["x"], batches[0]["y"]
    for name in ("fc1", "fc2"):
        p = getattr(model, name).weight

        def loss_of(flat, name=name, p=p):
            w = {"fc1": model.fc1.weight, "fc2": model.fc2.weight}
            w[name] = flat.view_as(p)
            return 0.5 * ((x @ w["fc1"].T) @ w["fc2"].T - y).square().sum(-1).mean()

        h = torch.autograd.functional.hessian(loss_of, p.detach().reshape(-1))
        assert traces[name] == pytest.approx(torch.trace(h).item(), rel=1e-6)
    assert all(p.requires_grad for p in model.parameters())


def test_sensitivity_scores_eq13():
    traces = {"a": 10.0, "b": 100.0, "c": 1000.0}
    scores = sensitivity_scores(traces, kappa=1.0)
    logs = [math.log(v) for v in traces.values()]
    mu = sum(logs) / 3
    sigma = math.sqrt(sum((l - mu) ** 2 for l in logs) / 3)
    expected = {k: 1 / (1 + math.exp(-(math.log(v) - mu) / (sigma + 1e-8))) for k, v in traces.items()}
    assert scores == pytest.approx(expected)
    assert scores["b"] == pytest.approx(0.5) and scores["a"] < 0.5 < scores["c"]


def test_trace_io_and_legacy_keys(tmp_path):
    path = tmp_path / "traces.json"
    save_traces(str(path), {"model.layers.0.mlp.up_proj": 1.5}, {"num_sketch": 10})
    assert load_traces(str(path)) == {"model.layers.0.mlp.up_proj": 1.5}

    legacy = tmp_path / "legacy.pkl"
    with open(legacy, "wb") as f:
        pickle.dump({"traces": {"layer_3_model.layers.0.mlp.up_proj": 2.0}, "scores": {}}, f)
    assert load_traces(str(legacy)) == {"model.layers.0.mlp.up_proj": 2.0}

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"layer_0_x.q_proj": 3.0}))
    assert load_traces(str(flat)) == {"x.q_proj": 3.0}
