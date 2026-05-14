"""Tests for the byte-level BPE tokenizer.

Covers training (vocab construction, merges, special tokens)
and encoding/decoding (roundtrip, single-char fallback, special-token handling, unicode).
"""

from __future__ import annotations

import pytest

from tests.adapters import get_tokenizer, run_train_bpe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_corpus(tmp_path):
    """Write a small corpus with predictable byte-pair frequencies.

    Corpus: "aaabdaaabac"
    Most frequent pair: (97, 97) i.e. 'aa' -- appears 4 times
    After merging 'aa' -> 256: sequence becomes [256, 97, 98, 100, 256, 97, 98, 97, 99]
    Next most frequent pair: (256, 97) and (97, 98) both have count 2.
    Tie-breaking by smallest (p0, p1) gives (97, 98).
    """
    p = tmp_path / "tiny.txt"
    p.write_text("aaabdaaabac", encoding="utf-8")
    return str(p)


@pytest.fixture()
def lorem_corpus(tmp_path):
    """A slightly larger corpus for roundtrip and general tests."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    ) * 20
    p = tmp_path / "lorem.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# BPE Training tests (3)
# ---------------------------------------------------------------------------


def test_bpe_base_vocab(tiny_corpus):
    """With vocab_size=256 and no special tokens, the vocab should contain
    exactly the 256 single-byte entries and zero merges."""
    vocab, merges = run_train_bpe(tiny_corpus, vocab_size=256, special_tokens=[])

    assert len(merges) == 0, "Expected no merges when vocab_size == 256"
    assert len(vocab) == 256, "Expected exactly 256 base byte entries"
    for i in range(256):
        assert vocab[i] == bytes([i]), (
            f"Vocab entry {i} should be {bytes([i])!r}, got {vocab[i]!r}"
        )


def test_bpe_merges_and_tiebreaking(tiny_corpus):
    """With vocab_size=258 (2 merges) on 'aaabdaaabac':

    Corpus bytes: [97,97,97,98,100,97,97,97,98,97,99]
    Most frequent pair: (97,97) with count 4 -> merged as token 256.
    After merge: [256,97,98,100,256,97,98,97,99]
    Next: (256,97) and (97,98) both have count 2.
    Tie-breaking picks the smallest (p[0], p[1]), so (97,98) wins.
    """
    vocab, merges = run_train_bpe(tiny_corpus, vocab_size=258, special_tokens=[])

    assert len(merges) == 2, f"Expected 2 merges, got {len(merges)}"

    # First merge: most frequent pair is (ord('a'), ord('a')) = (97, 97)
    assert merges[0] == (97, 97), (
        f"First merge should be (97, 97), got {merges[0]}"
    )

    # Second merge: tie between (256, 97) and (97, 98) both with count 2.
    # Tie-breaking by smallest (p[0], p[1]) gives (97, 98).
    assert merges[1] == (97, 98), (
        f"Second merge should be (97, 98), got {merges[1]}"
    )

    # The merged vocab entries should concatenate the underlying bytes
    assert vocab[256] == b"aa", f"Token 256 should be b'aa', got {vocab[256]!r}"
    assert vocab[257] == b"ab", f"Token 257 should be b'ab', got {vocab[257]!r}"


def test_bpe_special_tokens(tiny_corpus):
    """Special tokens should be assigned IDs in the vocab beyond the base
    bytes and merges. With vocab_size=258 and 1 special token, expect 1 merge
    (258 - 256 - 1 = 1)."""
    special = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(tiny_corpus, vocab_size=258, special_tokens=special)

    assert len(merges) == 1, f"Expected 1 merge, got {len(merges)}"

    # Special token should be findable in the vocab
    special_bytes = "<|endoftext|>".encode("utf-8")
    special_ids = [
        tid for tid, tbytes in vocab.items()
        if tbytes == special_bytes and tid >= 256 + len(merges)
    ]
    assert len(special_ids) == 1, (
        f"Expected exactly one vocab entry for the special token, found {len(special_ids)}"
    )


# ---------------------------------------------------------------------------
# Encode / Decode tests (5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["hello world", "aaabdaaabac", "a", " "],
    ids=["hello_world", "repeated_chars", "single_char", "space"],
)
def test_encode_decode_roundtrip(lorem_corpus, text):
    """Encoding then decoding should recover the original text exactly."""
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, (
        f"Roundtrip failed: encode->decode produced {decoded!r}, expected {text!r}"
    )


def test_encode_decode_empty_string(lorem_corpus):
    """Empty string encodes to empty list; empty list decodes to empty string."""
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    assert tokenizer.encode("") == [], "encode('') should return []"
    assert tokenizer.decode([]) == "", "decode([]) should return ''"


def test_encode_single_chars(tmp_path):
    """When there are no merges, each character should map to its raw byte
    value(s)."""
    p = tmp_path / "single.txt"
    p.write_text("abc", encoding="utf-8")
    corpus = str(p)

    vocab, merges = run_train_bpe(corpus, vocab_size=256, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    ids = tokenizer.encode("xyz")
    assert ids == [120, 121, 122], (
        f"Without merges, 'xyz' should encode to [120, 121, 122], got {ids}"
    )


def test_special_token_roundtrip(lorem_corpus):
    """A special token embedded in text should be encoded as a single ID
    and survive a full roundtrip."""
    special = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=special)
    tokenizer = get_tokenizer(vocab, merges, special_tokens=special)

    text = "hello<|endoftext|>world"
    ids = tokenizer.encode(text)

    # Find the special token's ID
    special_bytes = "<|endoftext|>".encode("utf-8")
    special_id = next(
        tid for tid, tbytes in vocab.items()
        if tbytes == special_bytes and tid >= 256 + len(merges)
    )

    # The special token should appear exactly once in the encoded output
    assert ids.count(special_id) == 1, (
        f"Special token ID {special_id} should appear exactly once, "
        f"found {ids.count(special_id)} times in {ids}"
    )

    # Decode and verify roundtrip
    decoded = tokenizer.decode(ids)
    assert decoded == text, (
        f"Special token roundtrip failed: got {decoded!r}, expected {text!r}"
    )


def test_encode_applies_merges(tiny_corpus):
    """After training with merges, encoding should produce merged tokens,
    not raw bytes."""
    vocab, merges = run_train_bpe(tiny_corpus, vocab_size=258, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    ids = tokenizer.encode("aa")
    raw_len = len("aa".encode("utf-8"))  # 2
    assert len(ids) < raw_len, (
        f"'aa' should encode to fewer tokens than raw bytes ({raw_len}), got {len(ids)}"
    )
    # Specifically, "aa" should encode to [256] (single merged token)
    assert ids == [256], f"'aa' should encode to [256], got {ids}"


@pytest.mark.parametrize(
    "text",
    ["café", "naïve", "über", "日本語"],
    ids=["cafe_accent", "naive_diaeresis", "uber_umlaut", "japanese"],
)
def test_encode_decode_unicode(lorem_corpus, text):
    """Byte-level BPE must handle multi-byte UTF-8 characters correctly."""
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, (
        f"Unicode roundtrip failed: {text!r} -> {ids} -> {decoded!r}"
    )


# ----------------------------------------------------------------------
# Additional coverage: longest-match special-token priority, decode of
# invalid UTF-8, special-token IDs are contiguous at the top of vocab
# ----------------------------------------------------------------------


def test_special_token_longest_match(lorem_corpus):
    """When two special tokens share a prefix, encode() must use the
    LONGER one. Otherwise a corpus containing ``<|endoftext|>`` could be
    silently split as ``<|end|>`` + ``oftext|>``."""
    specials = ["<|end|>", "<|endoftext|>"]
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=specials)
    tokenizer = get_tokenizer(vocab, merges, special_tokens=specials)

    short_bytes = "<|end|>".encode("utf-8")
    long_bytes = "<|endoftext|>".encode("utf-8")
    short_id = next(tid for tid, b in vocab.items() if b == short_bytes)
    long_id = next(tid for tid, b in vocab.items() if b == long_bytes)
    assert short_id != long_id

    ids = tokenizer.encode("<|endoftext|>")
    assert long_id in ids, (
        f"encode('<|endoftext|>') must contain the longer special-token id "
        f"{long_id}; got {ids}"
    )
    assert short_id not in ids, (
        f"encode('<|endoftext|>') must NOT split as the shorter special "
        f"({short_id}); got {ids}"
    )


def test_decode_invalid_utf8(lorem_corpus):
    """``decode`` must use ``errors='replace'`` so an isolated UTF-8
    lead byte does not raise — generation samples one token at a time
    and routinely produces partial codepoints mid-stream."""
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=[])
    tokenizer = get_tokenizer(vocab, merges, special_tokens=[])

    # Find a single-byte token ID whose byte is a UTF-8 lead byte (0xC3 = start
    # of a 2-byte sequence). Decoding it alone is invalid UTF-8.
    bad_id = next(tid for tid, b in vocab.items() if b == bytes([0xC3]))
    decoded = tokenizer.decode([bad_id])
    assert decoded == "�", (
        f"decode([0xC3]) should produce the U+FFFD replacement character "
        f"(errors='replace'), got {decoded!r}"
    )


def test_special_token_gets_highest_id(lorem_corpus):
    """Special tokens must occupy the highest IDs in the vocab, appended
    after all merges. This is what lets ``encode`` reliably find them."""
    special = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(lorem_corpus, vocab_size=300, special_tokens=special)

    special_bytes = "<|endoftext|>".encode("utf-8")
    special_id = next(tid for tid, b in vocab.items() if b == special_bytes)
    assert special_id == max(vocab.keys()), (
        f"Special token id {special_id} is not the maximum id in vocab "
        f"(max={max(vocab.keys())})."
    )
