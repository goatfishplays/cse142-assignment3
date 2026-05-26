"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from transformer_lm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """
    # TODO: see if is better to move data initally vs after to device
    if len(data) <= context_length:
        raise ValueError("no valid shifted target window exists")
    x_inds = torch.randint(0, len(data) - context_length, (batch_size, 1)) + torch.arange(0, context_length)  # dude I'm so smart wth, me when I actually learn how broadcasting works
    x = data[x_inds].to(device, dtype=torch.long)
    y = data[x_inds + 1].to(device, dtype=torch.long)
    return (x, y)
    raise NotImplementedError("TODO: Implement get_batch()")


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """
    raise NotImplementedError("TODO: Implement generate()")
