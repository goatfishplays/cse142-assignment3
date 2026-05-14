"""Download TinyStories dataset and prepare reference tokenization.

Usage:
    python scripts/prepare_data.py [--vocab_size 512] [--data_dir data]

This script is self-contained — it does NOT import from transformer_lm,
so it works before students have implemented anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
import urllib.request
from collections import Counter

import numpy as np


TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
# Number of stories to download. Set just below the maximum that fits in our
# fetch buffer (~54MB of TinyStories prefix yields ~49,999 complete stories);
# leaving margin avoids triggering re-downloads on existing caches.
NUM_STORIES = 49000  # ~40M chars, ~15M tokens after BPE


# ---------------------------------------------------------------------------
# Self-contained BPE (does not import transformer_lm)
# ---------------------------------------------------------------------------


def _get_pair_counts(seq: list[int]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(len(seq) - 1):
        counts[(seq[i], seq[i + 1])] += 1
    return counts


def _merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged


def _train_bpe(
    text: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Numpy-vectorized BPE training.

    Same algorithm and tie-breaking as the naive Python version (highest
    pair count; ties broken by lexicographically smallest (a, b)). About
    50-100x faster on 10M+ byte corpora since pair counting and replacement
    are vectorized with numpy. Students implementing their own
    `transformer_lm.tokenizer.train_bpe` may use any algorithmic variant
    (naive, numpy, linked-list, etc.) as long as it produces the same
    vocab/merges for the test corpora.
    """
    raw_bytes = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    initial_len = len(raw_bytes)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    seq = raw_bytes  # np.int32 array
    merges: list[tuple[int, int]] = []
    num_merges = vocab_size - 256 - len(special_tokens)
    print(f"Training BPE: {num_merges} merges over {initial_len:,} bytes "
          "(numpy-vectorized)")
    t0 = time.time()
    report_every = max(1, num_merges // 25)

    for i in range(num_merges):
        if len(seq) < 2:
            break
        # Encode adjacent pairs as int64 keys (high 32 bits = first id,
        # low 32 bits = second id) so np.unique can count them in one pass.
        a = seq[:-1].astype(np.int64)
        b = seq[1:].astype(np.int64)
        pair_keys = (a << 32) | b
        unique_keys, counts = np.unique(pair_keys, return_counts=True)
        if len(unique_keys) == 0:
            break
        max_count = counts.max()
        # Tie-break: among pairs with the max count, pick the one with the
        # smallest (a, b) lex order. Since pair_keys = (a << 32) | b, the
        # smallest int64 key with max count IS the lex-smallest (a, b).
        candidates = unique_keys[counts == max_count]
        best_key = int(candidates.min())
        best_a = best_key >> 32
        best_b = best_key & 0xFFFFFFFF

        new_id = 256 + len(merges)
        # Build mask of positions starting a (best_a, best_b) pair, with
        # left-to-right non-overlapping greedy match (matches the naive
        # _merge_pair semantics — see test_bpe_merges_and_tiebreaking).
        match = (seq[:-1] == best_a) & (seq[1:] == best_b)
        if match.any():
            # Greedy non-overlap: a True at position i blocks position i+1
            # from also matching. Process left-to-right.
            keep_match = np.zeros_like(match)
            i_pos = 0
            match_indices = np.flatnonzero(match)
            for idx in match_indices:
                if idx >= i_pos:
                    keep_match[idx] = True
                    i_pos = idx + 2  # skip the second token of the matched pair
            # Build new seq: at each match position write new_id, skip pos+1.
            # Positions to drop are pos+1 of each match.
            drop_mask = np.zeros(len(seq), dtype=bool)
            keep_indices = np.flatnonzero(keep_match)
            drop_mask[keep_indices + 1] = True
            new_seq = seq.copy()
            new_seq[keep_indices] = new_id
            seq = new_seq[~drop_mask]
        # else: no matches — seq unchanged
        vocab[new_id] = vocab[int(best_a)] + vocab[int(best_b)]
        merges.append((int(best_a), int(best_b)))

        if (i + 1) % report_every == 0 or (i + 1) == num_merges:
            elapsed = time.time() - t0
            done = i + 1
            rate = done / max(elapsed, 1e-9)
            eta = (num_merges - done) / max(rate, 1e-9)
            print(
                f"  merge {done:>3d}/{num_merges} | "
                f"seq={len(seq):>10,} ({100*len(seq)/initial_len:.1f}% of init) | "
                f"elapsed {elapsed:>5.1f}s | eta {eta:>5.1f}s"
            )

    for st in special_tokens:
        st_id = len(vocab)
        vocab[st_id] = st.encode("utf-8")
    return vocab, merges


def _encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Numpy-vectorized BPE encode. Same greedy left-to-right semantics
    as the naive per-position loop, but uses numpy boolean masking to
    apply each merge in O(N) rather than walking the sequence in Python.
    For ~47K TinyStories ~800-char stories with 255 merges, this is
    ~50x faster than the naive version (~30s vs ~30 min total)."""
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for merge_idx, (a, b) in enumerate(merges):
        if len(ids) < 2:
            break
        match = (ids[:-1] == a) & (ids[1:] == b)
        if not match.any():
            continue
        new_id = 256 + merge_idx
        # Greedy non-overlap left-to-right (matches the naive while-loop).
        match_indices = np.flatnonzero(match)
        keep = []
        i_pos = 0
        for idx in match_indices.tolist():
            if idx >= i_pos:
                keep.append(idx)
                i_pos = idx + 2
        if not keep:
            continue
        keep_arr = np.asarray(keep, dtype=np.int64)
        drop_mask = np.zeros(len(ids), dtype=bool)
        drop_mask[keep_arr + 1] = True
        new_ids = ids.copy()
        new_ids[keep_arr] = new_id
        ids = new_ids[~drop_mask]
    return ids.tolist()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def download_tinystories(data_dir: str, num_stories: int = NUM_STORIES) -> str:
    """Download a subset of TinyStories via HTTP range request."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tinystories.txt")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        count = text.count("<|endoftext|>")
        if count >= num_stories:
            print(f"Found existing {filepath} ({count} stories, {len(text):,} chars)")
            return filepath
        print(f"WARNING: {filepath} has {count} stories, need {num_stories}, re-downloading...")
        os.remove(filepath)

    # Download enough bytes to capture num_stories stories.
    # Average story is ~800 chars, so fetch ~20% extra to be safe.
    fetch_bytes = int(num_stories * 900 * 1.2)
    print(f"Downloading TinyStories (first {fetch_bytes // 1_000_000}MB)...")
    req = urllib.request.Request(TINYSTORIES_URL)
    req.add_header("Range", f"bytes=0-{fetch_bytes}")
    with urllib.request.urlopen(req) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")

    # Split into stories and keep first num_stories
    parts = text.split("<|endoftext|>")
    stories = [s.strip() for s in parts[:num_stories] if s.strip()]
    if len(stories) < num_stories:
        print(f"WARNING: Only got {len(stories)} stories (wanted {num_stories})")

    subset_text = "\n<|endoftext|>\n".join(stories) + "\n<|endoftext|>\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(subset_text)
    print(f"Saved {len(stories)} stories ({len(subset_text):,} chars) to {filepath}")
    return filepath


def _split_stories(text: str) -> list[str]:
    """Split text on <|endoftext|> markers into individual stories."""
    parts = text.split("<|endoftext|>")
    return [s.strip() for s in parts if s.strip()]


def _assign_splits(
    stories: list[str], val_fraction: float = 0.10,
) -> tuple[list[str], list[str]]:
    """Assign each story to train or val using a deterministic public hash.

    No secret involved: the split is reproducible by anyone with the same
    story list. The staff hidden test set comes from a SEPARATE file (the
    TinyStories validation file) and is not derivable from this split.
    """
    train, val = [], []
    for i, story in enumerate(stories):
        h = hashlib.sha256(f"public_split:{i}".encode()).hexdigest()
        bucket = int(h[:4], 16) % 1000
        if bucket < int(val_fraction * 1000):
            val.append(story)
        else:
            train.append(story)
    return train, val


def prepare_data(
    vocab_size: int = 512,
    data_dir: str = "data",
) -> None:
    text_path = download_tinystories(data_dir)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Total characters: {len(text):,}")

    # Split stories into train/val at the STORY level (public deterministic split)
    stories = _split_stories(text)
    train_stories, val_stories = _assign_splits(stories)
    print(f"Story split: {len(train_stories)} train, {len(val_stories)} val")

    # Shuffle train and val stories for diversity
    rng = random.Random(42)
    rng.shuffle(train_stories)
    rng.shuffle(val_stories)

    # Train BPE on train+val (entire student-visible corpus)
    train_val_text = "\n".join(train_stories + val_stories)
    print(f"Training BPE with vocab_size={vocab_size} on train+val ({len(train_val_text):,} chars)...")
    special_tokens = ["<|endoftext|>"]
    vocab, merges = _train_bpe(train_val_text, vocab_size, special_tokens)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    ref_path = os.path.join(data_dir, "reference_tokenizer.pkl")
    with open(ref_path, "wb") as f:
        pickle.dump(
            {"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f
        )
    print(f"Saved reference tokenizer to {ref_path}")

    # Save tokenizer.json (v2 format — JSON for security and inspectability)
    tok_json_path = os.path.join(data_dir, "tokenizer.json")
    tok_data = {
        "type": "byte_bpe",
        "vocab_size": vocab_size,
        "merges": [list(m) for m in merges],
        "special_tokens": special_tokens,
    }
    with open(tok_json_path, "w") as f:
        json.dump(tok_data, f)
    print(f"Saved tokenizer.json to {tok_json_path}")

    # Tokenize train and val separately, with EOS between stories
    eos_id = len(vocab) - 1  # last token is <|endoftext|>
    print(f"EOS token ID: {eos_id}")

    print("Tokenizing train stories...")
    train_ids = []
    train_chars = 0
    for story in train_stories:
        train_ids.extend(_encode(story, merges))
        train_ids.append(eos_id)
        train_chars += len(story)
    print(f"Train tokens: {len(train_ids):,} (compression: {train_chars/len(train_ids):.2f}x)")

    print("Tokenizing val stories...")
    val_ids = []
    val_chars = 0
    for story in val_stories:
        val_ids.extend(_encode(story, merges))
        val_ids.append(eos_id)
        val_chars += len(story)
    print(f"Val tokens: {len(val_ids):,} (compression: {val_chars/len(val_ids):.2f}x)")

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    np.array(train_ids, dtype=np.uint16).tofile(train_path)
    np.array(val_ids, dtype=np.uint16).tofile(val_path)
    print(f"Train tokens: {len(train_ids):,} -> {train_path}")
    print(f"Val tokens:   {len(val_ids):,} -> {val_path}")

    print("\n--- Vocabulary sample ---")
    for i in range(256, min(266, len(vocab))):
        print(f"  ID {i}: {vocab[i]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories data")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    prepare_data(args.vocab_size, args.data_dir)


if __name__ == "__main__":
    main()
