"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    raise NotImplementedError("TODO: Implement train_bpe()")


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("TODO: Implement BPETokenizer.__init__()")

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        raise NotImplementedError("TODO: Implement encode()")

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        raise NotImplementedError("TODO: Implement decode()")
