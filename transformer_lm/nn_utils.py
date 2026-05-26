"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """
    # print(x)
    # print(torch.max(x, dim=dim, keepdim=True)[0])
    # print(torch.max(x, dim=dim, keepdim=True)[0])
    terms = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])  # I'm like 900% sure we need to edit dims # why the frick does it return a tuple :()
    bottom = terms.sum(dim=dim, keepdim=True)
    return terms / bottom

    # raise NotImplementedError("TODO: Implement softmax()")


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """

    return x * x.sigmoid()
    # raise NotImplementedError("TODO: Implement silu()")


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    B, T, V = logits.shape

    maxl = torch.max(logits, dim=-1, keepdim=True)[0]
    # print(maxl.shape)
    # print(logits.shape)  # ! AI Use: helped me find torch.gather
    # print(targets)
    return torch.sum(maxl + torch.log(torch.sum(torch.exp(logits - maxl), dim=-1, keepdim=True)) - logits.gather(-1, targets.unsqueeze(-1))) / (B * T)  # I'm like 90% sure this is wrong

    raise NotImplementedError("TODO: Implement cross_entropy_loss()")
