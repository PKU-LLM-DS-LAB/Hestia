"""Hutch++ estimation of tensor-wise Hessian traces.

For every weight tensor ``W_i`` we estimate ``h_i = Tr(H_i)``, where ``H_i`` is
the diagonal block of the Hessian of the *full-precision* calibration loss
with respect to ``vec(W_i)``:

    h_i = Tr(Q^T H_i Q) + 1/m * Tr(G^T (I - QQ^T) H_i (I - QQ^T) G)

``Q`` is an orthonormal basis of ``H_i S`` for a Rademacher sketch ``S``
(rank ``r``) and ``G`` holds ``m`` Rademacher probes for the residual. The
Hessian is accessed only through Hessian-vector products (double backward),
averaged over a fixed set of calibration batches.
"""

from __future__ import annotations

import logging
import zlib
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

Batch = Mapping[str, torch.Tensor]


def rademacher(numel: int, k: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, (numel, k), generator=generator).float().mul_(2).sub_(1)


def hutchpp_trace(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_sketch: int = 10,
    num_query: int = 20,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Generic Hutch++ for an implicit symmetric matrix given by ``matvec(V) -> A V``."""
    generator = generator or torch.Generator().manual_seed(0)
    r = min(num_sketch, dim)
    q, _ = torch.linalg.qr(matvec(rademacher(dim, r, generator)), mode="reduced")
    trace = torch.sum(q * matvec(q)).item()
    if r >= dim or num_query <= 0:
        return trace
    g = rademacher(dim, num_query, generator)
    g_perp = g - q @ (q.T @ g)
    return trace + torch.sum(g_perp * matvec(g_perp)).item() / num_query


def _default_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(**batch).loss


class HessianTraceEstimator:
    """Estimate ``Tr(H_i)`` for a set of weight tensors of ``model``.

    Args:
        model: Full-precision model (kept in eval mode).
        batches: Calibration batches (dicts accepted by ``model(**batch)``;
            ``labels`` must be present for the default loss).
        loss_fn: ``loss_fn(model, batch) -> scalar``.
        num_sketch: Sketch rank ``r``.
        num_query: Number of residual probes ``m``.
        chunk_size: Number of tensors processed per forward pass. Larger values
            need fewer forward passes but more memory for the vector buffers.
        device: Device used for the forward / backward passes.
        seed: Seed of the Rademacher probes.
    """

    def __init__(
        self,
        model: nn.Module,
        batches: Sequence[Batch],
        loss_fn: Callable[[nn.Module, Batch], torch.Tensor] = _default_loss,
        num_sketch: int = 10,
        num_query: int = 20,
        chunk_size: int = 1,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.batches = list(batches)
        self.loss_fn = loss_fn
        self.num_sketch = num_sketch
        self.num_query = num_query
        self.chunk_size = max(1, chunk_size)
        self.device = device or next(model.parameters()).device
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _hvp_pass(self, params: List[nn.Parameter], vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return ``mean_b H_i^{(b)} V_i`` for every parameter ``i`` in the chunk."""
        out = [torch.zeros_like(v) for v in vectors]
        for batch in self.batches:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.loss_fn(self.model, batch)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            for i, (p, g, v) in enumerate(zip(params, grads, vectors)):
                g = g.reshape(-1)
                for j in range(v.shape[1]):
                    vj = v[:, j].to(device=self.device, dtype=g.dtype)
                    (hv,) = torch.autograd.grad(g @ vj, p, retain_graph=True)
                    out[i][:, j] += hv.reshape(-1).float().cpu()
            del loss, grads
        n = max(1, len(self.batches))
        return [o / n for o in out]

    def _estimate_chunk(self, names: List[str], params: List[nn.Parameter]) -> Dict[str, float]:
        # Probes depend only on (seed, tensor name): results do not depend on sharding.
        gens = [torch.Generator().manual_seed(self.seed + zlib.crc32(n.encode())) for n in names]
        numels = [p.numel() for p in params]
        ranks = [min(self.num_sketch, n) for n in numels]

        # Stage 1: low-rank sketch, Q = orth(H S).
        sketches = [rademacher(n, r, g) for n, r, g in zip(numels, ranks, gens)]
        bases = [torch.linalg.qr(y, mode="reduced")[0] for y in self._hvp_pass(params, sketches)]
        del sketches

        # Stage 2: exact trace on span(Q) + Hutchinson estimate on the complement.
        vectors, residual = [], []
        for q, n, r, g in zip(bases, numels, ranks, gens):
            if r < n and self.num_query > 0:
                probes = rademacher(n, self.num_query, g)
                probes = probes - q @ (q.T @ probes)
                residual.append(True)
                vectors.append(torch.cat([q, probes], dim=1))
            else:
                residual.append(False)
                vectors.append(q)
        products = self._hvp_pass(params, vectors)

        traces = {}
        for name, v, hv, r, has_res in zip(names, vectors, products, ranks, residual):
            trace = torch.sum(v[:, :r] * hv[:, :r]).item()
            if has_res:
                trace += torch.sum(v[:, r:] * hv[:, r:]).item() / self.num_query
            traces[name] = trace
        return traces

    # ------------------------------------------------------------------ #
    def estimate(self, named_params: Mapping[str, nn.Parameter]) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        requires_grad = {p: p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)

        names = list(named_params)
        traces: Dict[str, float] = {}
        try:
            for start in range(0, len(names), self.chunk_size):
                chunk = names[start : start + self.chunk_size]
                params = [named_params[n] for n in chunk]
                for p in params:
                    p.requires_grad_(True)
                traces.update(self._estimate_chunk(chunk, params))
                for p in params:
                    p.requires_grad_(False)
                logger.info(
                    "Hutch++ %d/%d tensors done (%s)", min(start + len(chunk), len(names)), len(names), chunk[-1]
                )
        finally:
            for p, flag in requires_grad.items():
                p.requires_grad_(flag)
            self.model.train(was_training)
        return traces


def estimate_hessian_traces(
    model: nn.Module,
    batches: Sequence[Batch],
    module_names: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, float]:
    """Estimate ``Tr(H_i)`` of ``module.weight`` for the given linear modules.

    ``module_names`` defaults to every ``nn.Linear`` in ``model``.
    """
    modules = dict(model.named_modules())
    if module_names is None:
        module_names = [n for n, m in modules.items() if isinstance(m, nn.Linear)]
    named_params = {n: modules[n].weight for n in module_names}
    return HessianTraceEstimator(model, batches, **kwargs).estimate(named_params)
