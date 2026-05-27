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
    # TODO: ask whether should use same dtype as batch or not
    # ! AI USE: asked for clarification on how pytorch handles memory(does slicing reallocate, does unsqueezing reallocate, etc.)
    # ! AI USE: ask for clarification on how to get a parameter from model.parameters(), gave me next()
    if context_length == None:
        context_length = model.context_length
    context = torch.tensor(prompt_ids + [-1] * max_new_tokens, device=next(model.parameters()).device)
    for i in range(len(prompt_ids), len(context)):
        # print(context)
        # print(context.shape)
        # print(context[:i].unsqueeze(0).shape)
        # print()
        logits = model(context[max(0, i - context_length) : i].unsqueeze(0))[0, -1] / temperature  # Does it matter if I unsqueeze here or before???? # TODO: test if matters
        # print(logits)
        # print(logits.shape)
        # print

        y_pred = torch.multinomial(softmax(logits), num_samples=1)
        # print(y_pred)
        context[i] = y_pred.item()
    # print(context)
    return list(context)

    raise NotImplementedError("TODO: Implement generate()")
