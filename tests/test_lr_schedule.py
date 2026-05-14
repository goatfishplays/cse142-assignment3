"""Contract tests for ``transformer_lm.lr_schedule.lr_schedule``.

Students may rewrite the body of ``lr_schedule`` to implement any schedule
shape, but they MUST preserve:

  - the function signature ``(step, max_steps, base_lr, min_lr, warmup_steps)``
  - a Python float return type
  - non-negative learning rates everywhere
  - a sensible warmup → peak → decay shape

The training script calls this once per step. A broken implementation
silently tanks val_loss with no test feedback, so these contract checks
catch the easy mistakes.
"""

from __future__ import annotations

import pytest

from transformer_lm.lr_schedule import lr_schedule


def test_signature_returns_float():
    out = lr_schedule(step=0, max_steps=100, base_lr=1e-3, min_lr=1e-5, warmup_steps=10)
    assert isinstance(out, float), (
        f"lr_schedule must return a Python float, got {type(out).__name__}. "
        "The training script multiplies the result by param_group LR fields, "
        "which expect a plain number."
    )


def test_nonnegative_everywhere():
    for step in (0, 1, 5, 10, 50, 99, 100, 999):
        lr = lr_schedule(step=step, max_steps=100, base_lr=1e-3, min_lr=1e-5, warmup_steps=10)
        assert lr >= 0, f"lr_schedule(step={step}) returned negative LR {lr}"


def test_warmup_starts_low():
    """At step 0 with warmup_steps > 0, the LR should be below base_lr."""
    base_lr = 1e-3
    lr0 = lr_schedule(step=0, max_steps=1000, base_lr=base_lr, min_lr=1e-5, warmup_steps=100)
    assert lr0 < base_lr, (
        f"With warmup_steps=100 > 0, lr_schedule(0) must be below base_lr "
        f"({base_lr}), got {lr0}. Either warmup is missing or the schedule "
        f"opens at peak (which destabilizes early training)."
    )


def test_warmup_reaches_peak():
    """Right after warmup, the LR should be at (or very near) base_lr."""
    base_lr = 1e-3
    lr_peak = lr_schedule(step=100, max_steps=1000, base_lr=base_lr, min_lr=1e-5, warmup_steps=100)
    assert lr_peak == pytest.approx(base_lr, rel=1e-2), (
        f"At step==warmup_steps the LR should equal base_lr (~{base_lr}), "
        f"got {lr_peak}. Off-by-one in the warmup boundary will hurt training."
    )


def test_decays_after_warmup():
    """After warmup, LR should be non-increasing — at most flat, otherwise
    decaying toward min_lr."""
    base_lr = 1e-3
    min_lr = 1e-5
    max_steps = 1000
    warmup = 100
    samples = [
        lr_schedule(step=s, max_steps=max_steps, base_lr=base_lr,
                    min_lr=min_lr, warmup_steps=warmup)
        for s in range(warmup, max_steps + 1, 50)
    ]
    for prev, nxt in zip(samples, samples[1:]):
        assert nxt <= prev + 1e-9, (
            f"After warmup the LR must not increase. Saw {prev} → {nxt}."
        )


def test_min_lr_floor():
    """At max_steps and beyond, the LR should be at or above min_lr (never
    smaller — that would underflow training)."""
    min_lr = 1e-5
    lr_end = lr_schedule(step=1000, max_steps=1000, base_lr=1e-3, min_lr=min_lr, warmup_steps=100)
    assert lr_end >= min_lr - 1e-12, (
        f"At max_steps, lr_schedule must be ≥ min_lr ({min_lr}), got {lr_end}."
    )
