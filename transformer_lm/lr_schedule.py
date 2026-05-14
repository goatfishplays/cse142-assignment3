"""Learning rate schedule.

Students may customize this function to implement any LR schedule.
The default is cosine decay with linear warmup.

Interface contract — the function signature must not change:

    lr_schedule(step, max_steps, base_lr, min_lr, warmup_steps) -> float

The training script calls this function once per step to get the current
learning rate.
"""

import math


def lr_schedule(
    step: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Return the learning rate for the given step.

    Args:
        step: Current training step (0-indexed).
        max_steps: Total number of training steps.
        base_lr: Peak learning rate (after warmup).
        min_lr: Minimum learning rate (floor).
        warmup_steps: Number of linear warmup steps.

    Returns:
        Learning rate for this step.
    """
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
