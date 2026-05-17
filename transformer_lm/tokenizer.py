# * AI USE GPT and google search AI: Asked how file inputs in python work with UTF8 and working with bytes converting back and forth and such

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

    bytesToInt = {i.to_bytes(1, byteorder="big"): i for i in range(256)}
    merges = []

    corpus = []
    lastByte = None
    pairCounts = dict()

    # setup initial counts and corpus
    with open(input_path, "rb") as f:
        while True:
            byte = f.read(1)
            if not byte:
                break
            corpus.append(byte)
            if lastByte != None:
                pair = (lastByte, byte)
                pairCounts[pair] = pairCounts.get(pair, 0) + 1
            lastByte = byte

    # add all extra vocab words
    for _ in range(vocab_size - 256 - len(special_tokens)):
        # print(pairCounts)
        maxPair = max(pairCounts.items(), key=lambda x: (x[1], -int.from_bytes(x[0][0]), -int.from_bytes(x[0][1])))[0]
        # print(maxPair)
        pairCounts.pop(maxPair)
        merges.append((bytesToInt[maxPair[0]], bytesToInt[maxPair[1]]))
        # vocab[len(vocab), bytes(maxPair[0])]
        maxPairBytes = bytes(b"".join(maxPair))
        bytesToInt[maxPairBytes] = len(bytesToInt)

        i = 1  # they should get a replace function for lists like we have in strings that would be cool
        while i < len(corpus):
            if corpus[i - 1] == maxPair[0] and corpus[i] == maxPair[1]:
                # TODO: I don't rem if they want efficiency, if need make more efficient batch the removals via 2 ptr
                corpus[i - 1] = maxPairBytes
                corpus.pop(i)
                if i < len(corpus):
                    pair = (maxPairBytes, corpus[i])
                    pairCounts[pair] = pairCounts.get(pair, 0) + 1
            i += 1

    for i in special_tokens:
        bytesToInt[bytes(i, "utf-8")] = len(bytesToInt)

    return ({y: x for (x, y) in bytesToInt.items()}, merges)  # also why is this a dict lmaooo it can totally just be an array

    raise NotImplementedError("TODO: Implement train_bpe()")


print(train_bpe("UTF-8-demo.txt", 300, ["frick", "fish", "foggy"]))


class BPETokenizer:
    #! ask if special tokens for train and tokenizer different
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.byteToInt = {y: x for (x, y) in vocab.items()}

        # self.merges = list(map(lambda x: b"".join((vocab[x[0]], vocab[x[1]])), merges))
        self.merges = merges

        self.special_tokens = list(map(lambda x: bytes(x, "utf-8"), special_tokens))
        self.special_tokens.sort(key=len, reverse=True)

        print(vocab)

        # raise NotImplementedError("TODO: Implement BPETokenizer.__init__()")

    def encodeRecurse(self, textBytes: bytes, specialInd: int) -> list[int]:
        if specialInd == len(self.special_tokens):
            # ! Ask if need to follow steps strictly or as long as propperly tokenize is fine?
            # ! rem to double check this later, should be fine cause only ones that are single bytes should be from orig 256
            # print(textBytes)
            # print(self.byteToInt)
            # for b in textBytes:
            #     print(b)

            # tokens = list(map(lambda x: self.byteToInt[x], textBytes))
            tokens = [x for x in textBytes]
            # print(tokens)

            for j in range(len(self.merges)):
                merge = self.merges[j]
                # i = 1
                # while i < len(tokens):
                for i in range(1, len(tokens)):
                    if tokens[i - 1] == merge[0] and tokens[i] == merge[1]:
                        tokens[i - 1] = 256 + j  # ! double check this later
                        tokens.pop(i)  # ! check that we are fine to advance i here, should be fine cause if we just merged i back we won't be merging again at i till next j
                    # else:
                    #     i+= 1
            return tokens

        parts = textBytes.split(self.special_tokens[specialInd])
        ret = []
        for i in range(len(parts) - 1):
            part = parts[i]
            ret.extend(self.encodeRecurse(part, specialInd + 1))
            ret.append(self.byteToInt[self.special_tokens[specialInd]])  # ! ask if this needs to be len vocab + special ind
        ret.extend(self.encodeRecurse(parts[-1], specialInd + 1))
        return ret

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        textBytes = bytes(text, "utf-8")
        print(textBytes)
        return self.encodeRecurse(textBytes, 0)

        raise NotImplementedError("TODO: Implement encode()")

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return b"".join(map(lambda x: self.vocab[x], ids)).decode(errors="replace")  # ! also need augment this if special is different from train special

        raise NotImplementedError("TODO: Implement decode()")
