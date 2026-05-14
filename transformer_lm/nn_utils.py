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
    raise NotImplementedError("TODO: Implement softmax()")


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    raise NotImplementedError("TODO: Implement silu()")


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    raise NotImplementedError("TODO: Implement cross_entropy_loss()")
