#!/usr/bin/env python
"""Tokenize a text corpus and pack it into fixed-length sequences.

The corpus is streamed, every document is tokenized (optionally followed by
EOS), all tokens are concatenated and cut into ``--seq-len`` chunks. The result
is saved with ``datasets.Dataset.save_to_disk`` and used by ``calibrate.py`` and
``train.py``.

``--dataset`` can be a Hugging Face Hub dataset name (e.g. ``openbmb/Ultra-FineWeb``)
or a local directory / glob of parquet or json(l) files.
"""

import argparse
import glob
import os

from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default=None, help="Dataset config name on the Hub.")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-sequences", type=int, default=0, help="Stop after N packed sequences (0: all).")
    p.add_argument("--no-eos", action="store_true", help="Do not append EOS after each document.")
    p.add_argument("--writer-batch-size", type=int, default=4096)
    return p.parse_args()


def stream_corpus(args):
    path = args.dataset
    if os.path.isdir(path) or any(c in path for c in "*?["):
        pattern = path if not os.path.isdir(path) else os.path.join(path, "**", "*")
        files = sorted(f for f in glob.glob(pattern, recursive=True) if f.endswith((".parquet", ".json", ".jsonl")))
        if not files:
            raise FileNotFoundError(f"No parquet/json files found under {path}")
        builder = "parquet" if files[0].endswith(".parquet") else "json"
        return load_dataset(builder, data_files=files, split="train", streaming=True)
    return load_dataset(path, args.name, split=args.split, streaming=True)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    corpus = stream_corpus(args)
    seq_len = args.seq_len

    def generate():
        buffer, produced = [], 0
        for sample in corpus:
            text = sample.get(args.text_field)
            if not text:
                continue
            buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if not args.no_eos:
                buffer.append(eos)
            while len(buffer) >= seq_len:
                chunk, buffer = buffer[:seq_len], buffer[seq_len:]
                yield {"input_ids": chunk, "labels": list(chunk), "attention_mask": [1] * seq_len}
                produced += 1
                if args.max_sequences and produced >= args.max_sequences:
                    return

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    dataset = Dataset.from_generator(generate, features=features, writer_batch_size=args.writer_batch_size)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved {dataset.num_rows:,} sequences of {seq_len} tokens to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
    # Streaming readers can leave background threads that crash during interpreter
    # finalization on some Python / datasets versions; the output is already on disk.
    os._exit(0)
