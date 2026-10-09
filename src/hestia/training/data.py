"""Data utilities for pre-tokenized, packed causal-LM datasets."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_packed_dataset(path: str, seed: int = 42):
    """Load a dataset produced by ``scripts/prepare_data.py`` (``input_ids`` of fixed length)."""
    from datasets import load_from_disk

    return load_from_disk(path).shuffle(seed=seed)


def lm_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def calibration_batches(
    dataset,
    num_batches: int,
    batch_size: int = 1,
    max_seq_len: int = 512,
) -> List[Dict[str, torch.Tensor]]:
    """First ``num_batches * batch_size`` sequences, truncated to ``max_seq_len``."""
    batches = []
    for b in range(num_batches):
        rows = dataset[b * batch_size : (b + 1) * batch_size]["input_ids"]
        if len(rows) == 0:
            break
        input_ids = torch.tensor([r[:max_seq_len] for r in rows], dtype=torch.long)
        batches.append({"input_ids": input_ids, "labels": input_ids.clone()})
    return batches
