#!/usr/bin/env python
"""Evaluate a HESTIA checkpoint with the target hard quantizer.

The latent weights are loaded, quantized with ``Q(W)`` using the
``hestia_config.json`` stored with the checkpoint, and evaluated with
lm-evaluation-harness and / or token-level perplexity on WikiText2 and C4.
"""

import argparse
import json
import logging
import math
import os
from typing import Iterator, List

import torch
from transformers import AutoModelForCausalLM

from hestia import HestiaConfig, freeze_model, quantize_model
from hestia.training import load_tokenizer

logger = logging.getLogger("hestia.evaluate")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--hestia-config", default=None, help="Defaults to <model-path>/hestia_config.json.")
    p.add_argument("--full-precision", action="store_true", help="Evaluate without quantization.")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="lm-eval tasks (comma separated, '' to skip).")
    p.add_argument("--ppl-datasets", default="wikitext2,c4", help="Perplexity datasets ('' to skip).")
    p.add_argument("--seq-len", type=int, default=2048, help="Sequence length for perplexity.")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--limit", type=float, default=None, help="Limit samples (for quick tests).")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="Output JSON (default: <model-path>/eval_results.json).")
    return p.parse_args()


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=DTYPES[args.dtype])
    if not args.full_precision:
        config = (
            HestiaConfig.from_pretrained(args.hestia_config)
            if args.hestia_config
            else HestiaConfig.find(args.model_path)
        )
        if config is None:
            raise FileNotFoundError("hestia_config.json not found; pass --hestia-config or --full-precision.")
        quantize_model(model, config)
        freeze_model(model)
    return model.to(args.device).eval()


def _texts(name: str) -> Iterator[str]:
    from datasets import load_dataset

    if name == "wikitext2":
        for row in load_dataset("wikitext", "wikitext-2-raw-v1", split="test"):
            yield from (line for line in row["text"].split("\n") if line)
    elif name == "c4":
        data_files = {"validation": "en/c4-validation.00000-of-00008.json.gz"}
        for row in load_dataset("allenai/c4", data_files=data_files, split="validation"):
            if row["text"]:
                yield row["text"]
    else:
        raise ValueError(f"Unsupported perplexity dataset: {name}")


def _packed_sequences(name: str, tokenizer, seq_len: int) -> Iterator[List[int]]:
    """Greedily pack whole documents (with EOS) into sequences of at most ``seq_len`` tokens."""
    bos = [tokenizer.bos_token_id] if getattr(tokenizer, "add_bos_token", False) else []
    doc = list(bos)
    for text in _texts(name):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(tokens) > seq_len:
            continue
        if len(doc) + len(tokens) > seq_len:
            yield doc
            doc = list(bos)
        doc.extend(tokens)
    if len(doc) > 1:
        yield doc


@torch.inference_mode()
def perplexity(model, tokenizer, name: str, seq_len: int, limit=None) -> dict:
    total_nll, total_tokens = 0.0, 0
    for i, seq in enumerate(_packed_sequences(name, tokenizer, seq_len)):
        if limit is not None and i >= limit:
            break
        ids = torch.tensor([seq], device=model.device)
        logits = model(ids, use_cache=False).logits[:, :-1].float()
        labels = ids[:, 1:].clone()
        labels[ids[:, :-1] == tokenizer.eos_token_id] = -100  # do not predict across documents
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
        )
        total_nll += nll.item()
        total_tokens += int((labels != -100).sum())
    return {"ppl": math.exp(total_nll / max(1, total_tokens)), "tokens": total_tokens}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args)
    results = {"model_path": args.model_path, "full_precision": args.full_precision}

    tasks = [t for t in args.tasks.split(",") if t]
    if tasks:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        out = evaluator.simple_evaluate(model=lm, tasks=tasks, limit=args.limit)
        print(make_table(out))
        results["lm_eval"] = out["results"]

    ppl_sets = [d for d in args.ppl_datasets.split(",") if d]
    if ppl_sets:
        limit = int(args.limit) if args.limit else None
        results["perplexity"] = {d: perplexity(model, tokenizer, d, args.seq_len, limit) for d in ppl_sets}
        for d, r in results["perplexity"].items():
            print(f"{d}: ppl={r['ppl']:.3f} ({r['tokens']} tokens)")

    output = args.output or os.path.join(args.model_path, "eval_results.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", output)


if __name__ == "__main__":
    main()
