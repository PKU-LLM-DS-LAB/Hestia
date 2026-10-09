#!/usr/bin/env python
"""HESTIA quantization-aware training.

All ``transformers.TrainingArguments`` are supported (``--learning_rate``,
``--max_steps``, ``--deepspeed`` ...) together with the fields of
:class:`hestia.HestiaConfig` (``--codebook``, ``--group_size``,
``--compress_ratio``, ``--init_temp``, ``--alpha`` ...). Arguments can also be
given as a single YAML / JSON file: ``python scripts/train.py args.yaml``.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from transformers.trainer_utils import SchedulerType, get_last_checkpoint

from hestia import HestiaConfig, HestiaScheduler, apply_sensitivity, quantize_model
from hestia.sensitivity import load_traces
from hestia.training import HestiaTrainer, lm_collator, load_packed_dataset, load_tokenizer

logger = logging.getLogger("hestia.train")


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Full-precision model to start QAT from."})
    dataset_path: str = field(metadata={"help": "Packed dataset (see scripts/prepare_data.py)."})
    tokenizer_name_or_path: Optional[str] = field(default=None)
    sensitivity_path: Optional[str] = field(
        default=None, metadata={"help": "Hessian traces from scripts/calibrate.py."}
    )
    wsd_decay_ratio: float = field(
        default=0.1, metadata={"help": "Fraction of steps used by the decay phase of the WSD LR schedule."}
    )
    wsd_min_lr_ratio: float = field(default=0.1, metadata={"help": "Final LR / peak LR of the WSD schedule."})


def parse_args():
    parser = HfArgumentParser((ScriptArguments, HestiaConfig, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def configure_wsd(script_args: ScriptArguments, training_args: TrainingArguments) -> None:
    if training_args.lr_scheduler_type != SchedulerType.WARMUP_STABLE_DECAY:
        return
    kwargs = dict(training_args.lr_scheduler_kwargs or {})
    if "num_decay_steps" not in kwargs:
        if training_args.max_steps <= 0:
            raise ValueError("The WSD schedule needs --max_steps (or explicit num_decay_steps).")
        kwargs["num_decay_steps"] = int(script_args.wsd_decay_ratio * training_args.max_steps)
    kwargs.setdefault("min_lr_ratio", script_args.wsd_min_lr_ratio)
    training_args.lr_scheduler_kwargs = kwargs


def main():
    script_args, hestia_config, training_args = parse_args()
    logging.basicConfig(
        level=logging.INFO if training_args.should_log else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    configure_wsd(script_args, training_args)

    tokenizer = load_tokenizer(script_args.tokenizer_name_or_path or script_args.model_name_or_path)
    train_dataset = load_packed_dataset(script_args.dataset_path, seed=training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    layers = quantize_model(model, hestia_config)

    if script_args.sensitivity_path:
        apply_sensitivity(layers, load_traces(script_args.sensitivity_path), kappa=hestia_config.kappa)
        logger.info("Hessian-guided annealing enabled (alpha=%s).", hestia_config.alpha)
    elif hestia_config.method == "hestia" and hestia_config.alpha != 0.0:
        logger.warning("No --sensitivity_path given: using the global temperature schedule (alpha ignored).")

    trainer = HestiaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lm_collator(tokenizer),
        processing_class=tokenizer,
        hestia_scheduler=HestiaScheduler.from_config(layers, hestia_config),
        hestia_config=hestia_config,
    )

    checkpoint = training_args.resume_from_checkpoint
    if checkpoint == "auto":
        out = training_args.output_dir
        checkpoint = get_last_checkpoint(out) if os.path.isdir(out) else None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
