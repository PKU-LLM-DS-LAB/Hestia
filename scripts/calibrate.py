#!/usr/bin/env python
"""Offline Hessian-trace calibration.

Estimates ``Tr(H_i)`` of the full-precision calibration loss for every linear
tensor that will be quantized and writes them to a JSON file consumed by
``scripts/train.py --sensitivity_path``. Tensors are sharded across processes
when launched with ``torchrun``.

    torchrun --nproc_per_node 8 scripts/calibrate.py \
        --model-path <hf_model> --dataset-path <packed_dataset> --output traces.json
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from hestia import HestiaConfig, quantizable_linear_names
from hestia.sensitivity import HessianTraceEstimator, save_traces, sensitivity_scores
from hestia.training import calibration_batches, load_packed_dataset

logger = logging.getLogger("hestia.calibrate")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True, help="Packed dataset (see scripts/prepare_data.py).")
    p.add_argument("--output", required=True, help="Output JSON file.")
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--num-sketch", type=int, default=10, help="Hutch++ sketch rank r.")
    p.add_argument("--num-query", type=int, default=20, help="Hutch++ residual probes m.")
    p.add_argument("--chunk-size", type=int, default=1, help="Tensors per forward pass.")
    p.add_argument("--skip-modules", nargs="*", default=["lm_head"])
    p.add_argument("--codebook", default="ternary", help="Only used to select quantizable layers.")
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=DTYPES[args.dtype]).to(device)
    config = HestiaConfig(codebook=args.codebook, skip_modules=args.skip_modules)
    names = quantizable_linear_names(model, config)
    local_names = names[rank::world]

    dataset = load_packed_dataset(args.dataset_path, seed=args.seed)
    batches = calibration_batches(dataset, args.num_batches, args.batch_size, args.max_seq_len)
    if rank == 0:
        logger.info("Calibrating %d tensors on %d process(es) with %d batches.", len(names), world, len(batches))

    modules = dict(model.named_modules())
    estimator = HessianTraceEstimator(
        model,
        batches,
        num_sketch=args.num_sketch,
        num_query=args.num_query,
        chunk_size=args.chunk_size,
        device=device,
        seed=args.seed,
    )
    start = time.time()
    traces = estimator.estimate({n: modules[n].weight for n in local_names})

    if distributed:
        gathered = [None] * world
        dist.all_gather_object(gathered, traces)
        traces = {k: v for part in gathered for k, v in part.items()}

    if rank == 0:
        traces = {n: traces[n] for n in names}
        metadata = {k: v for k, v in vars(args).items() if k != "output"}
        metadata["elapsed_seconds"] = time.time() - start
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_traces(args.output, traces, metadata)
        scores = sensitivity_scores(traces)
        top = sorted(scores, key=scores.get, reverse=True)[:5]
        logger.info("Saved %d traces to %s (%.1fs).", len(traces), args.output, metadata["elapsed_seconds"])
        logger.info("Most sensitive tensors: %s", top)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
