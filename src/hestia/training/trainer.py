"""Integration with the Hugging Face ``Trainer``."""

from __future__ import annotations

from typing import Optional

from transformers import Trainer, TrainerCallback

from hestia.config import HestiaConfig
from hestia.schedule import HestiaScheduler


class HestiaCallback(TrainerCallback):
    """Updates pressure and tensor-wise temperatures before every optimizer step.

    ``T`` is taken from ``state.max_steps`` unless the scheduler already has one.
    Resuming from a checkpoint restores ``global_step`` and therefore the schedule.
    """

    def __init__(self, scheduler: HestiaScheduler) -> None:
        self.scheduler = scheduler

    def on_train_begin(self, args, state, control, **kwargs):
        if self.scheduler.schedule is None:
            self.scheduler.set_total_steps(state.max_steps)
        self.scheduler.step(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.scheduler.step(state.global_step)


class HestiaTrainer(Trainer):
    """``Trainer`` that drives the HESTIA schedule, logs it and saves the config.

    Args:
        hestia_scheduler: Scheduler bound to the model's ``HestiaLinear`` layers.
        hestia_config: Saved as ``hestia_config.json`` next to every checkpoint so
            that :func:`hestia.load_hestia_model` can rebuild the quantizer.
    """

    def __init__(
        self,
        *args,
        hestia_scheduler: HestiaScheduler,
        hestia_config: Optional[HestiaConfig] = None,
        **kwargs,
    ) -> None:
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(HestiaCallback(hestia_scheduler))
        super().__init__(*args, callbacks=callbacks, **kwargs)
        self.hestia_scheduler = hestia_scheduler
        self.hestia_config = hestia_config

    def log(self, logs, *args, **kwargs):
        logs = {**logs, **self.hestia_scheduler.state()}
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Used for both intermediate checkpoints and the final model.
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.hestia_config is not None and self.args.should_save:
            self.hestia_config.save_pretrained(output_dir or self.args.output_dir)
