"""Decoder-only Transformer language model — all components from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from transformer_lm.nn_utils import silu, softmax

# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------


class Linear(nn.Module):
    """Linear layer: ``y = x @ W^T + b``.

    Initialize ``weight`` uniformly in ``[-1/sqrt(in_features), 1/sqrt(in_features)]``.
    Initialize ``bias`` to zero. When ``bias=False``, set ``self.bias = None``
    and skip the bias addition in ``forward`` (the transformer model uses
    ``bias=False`` for all projections, so this is the common case).
    """

    # ! used AI to ask about how to use torch uniform
    # TODO: remind me to ask if we are supposed to do iniitalize random uniform or arange style

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weights = nn.Parameter(torch.empty(in_features, out_features).uniform_(-bound, bound))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        # raise NotImplementedError("TODO: Implement Linear.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias
        # raise NotImplementedError("TODO: Implement Linear.forward()")


class Embedding(nn.Module):
    """Embedding look-up table.

    Initialize the weight matrix with normal distribution ``N(0, 0.02)``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.theEmbedler = nn.Parameter(torch.normal(mean=0, std=0.02, size=(num_embeddings, embedding_dim)))
        # raise NotImplementedError("TODO: Implement Embedding.__init__()")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.theEmbedler[indices]  # can't believe this actually works lmaooooo
        # raise NotImplementedError("TODO: Implement Embedding.forward()")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable parameters)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO: ask wth is d_model for????? especially if we don't have learnable params????
        self.eps = eps
        # raise NotImplementedError("TODO: Implement RMSNorm.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theMeanler = x.square().mean()  # TODO: check if this needs to be set to a dim
        return x / torch.sqrt(theMeanler + self.eps)

        # raise NotImplementedError("TODO: Implement RMSNorm.forward()")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — PROVIDED, not student-implemented
# ---------------------------------------------------------------------------


# --- PROVIDED: do not modify ---
class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Pre-computes sin/cos rotation matrices and applies them to queries
    and keys.  This module is **provided** — students do not need to
    implement it.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        # Pre-compute for max_seq_len positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, d_head/2)
        cos_cached = torch.cos(freqs)  # (T, d_head/2)
        sin_cached = torch.sin(freqs)  # (T, d_head/2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: ``(B, n_heads, T, d_head)``
            k: ``(B, n_heads, T, d_head)``

        Returns:
            Rotated ``(q, k)`` with the same shapes.
        """
        T = q.shape[2]
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head/2)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        # Duplicate cos/sin to full d_head: [cos, cos] for pairs
        cos = torch.cat([cos, cos], dim=-1)  # (1, 1, T, d_head)
        sin = torch.cat([sin, sin], dim=-1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        Q: ``(B, ..., T, d_k)``
        K: ``(B, ..., T, d_k)``
        V: ``(B, ..., T, d_v)``
        mask: Additive mask. Use ``0`` for positions to attend to and a
            large negative value (e.g. ``-1e9``, also written as ``-inf``
            in some references) for positions to mask out.

    Returns:
        ``(B, ..., T, d_v)``
    """
    # ! AI use needed to find full()
    sqrtdk = math.sqrt(Q.shape[-1])
    A = (Q @ K.transpose()) / sqrtdk
    # mask = torch.full(size=A.shape, fill_value=-torch.inf).triu(diagonal=1)
    return softmax(A + mask) @ V

    # raise NotImplementedError("TODO: Implement scaled_dot_product_attention()")


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask and RoPE.

    Uses a fused QKV projection for efficiency.

    Required attributes:
        qkv_proj: ``Linear(d_model, 3 * d_model, bias=False)`` — fused Q, K, V.
            Weight rows are packed as ``[Q_weights, K_weights, V_weights]``.
            Split with ``qkv.split(d_model, dim=-1)`` to recover Q, K, V.
        o_proj: ``Linear(d_model, d_model, bias=False)`` — output projection.
        rope: ``RotaryPositionEmbedding(d_head)`` — the default
            ``max_seq_len=4096`` is fine for our context length of 256.

    Args:
        d_model: Model dimension (must be divisible by ``n_heads``).
        n_heads: Number of attention heads (``d_model // n_heads`` must be even).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = Linear(d_model, d_model, False)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model / n_heads
        self.rope = RotaryPositionEmbedding(self.d_head)
        self.dropout = nn.Dropout(dropout)
        # raise NotImplementedError("TODO: Implement CausalMultiHeadSelfAttention.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``
        Returns:
            ``(B, T, d_model)``
        """
        B, T, d_model = x.shape
        Q, K, V = torch.split(self.qkv_proj(x), 3)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K = self.rope(Q, K)
        mask = torch.full(size=(self.d_model, self.d_model), fill_value=-torch.inf).triu(diagonal=1)
        out = scaled_dot_product_attention(Q, K, V, mask)
        return self.dropout(self.o_proj(out))  # TODO we are here
        # raise NotImplementedError("TODO: Implement CausalMultiHeadSelfAttention.forward()")


# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SwiGLU activation.

    ``FFN(x) = w_down(silu(w_gate(x)) * w_up(x))``

    Required attributes:
        w_gate: ``Linear(d_model, d_ff, bias=False)``
        w_up: ``Linear(d_model, d_ff, bias=False)``
        w_down: ``Linear(d_ff, d_model, bias=False)``

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()  # dude my variable naming is so inconsistent lmaoooo, keep defaulting to snake case and then realizing that looks stupid for variable names in math
        self.W_up = Linear(d_model, d_ff, False)
        self.W_gate = Linear(d_model, d_ff, False)  # check later if this should have bias
        self.W_down = Linear(d_ff, d_model, False)
        self.dropout = nn.Dropout(dropout)
        # raise NotImplementedError("TODO: Implement FeedForward.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.W_down(silu(self.W_gate(x) * self.W_down(x))))
        # raise NotImplementedError("TODO: Implement FeedForward.forward()")


# ---------------------------------------------------------------------------
# Transformer block and full LM
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single pre-LN transformer decoder block.

    ``x = x + attn(ln1(x))``
    ``x = x + ffn(ln2(x))``

    Required attributes:
        ln1, ln2: ``RMSNorm(d_model)``
        attn: ``CausalMultiHeadSelfAttention(d_model, n_heads)``
        ffn: ``FeedForward(d_model, d_ff)``

    Note: zero-init of ``o_proj`` and ``w_down`` is done in
    ``TransformerLM.__init__``, **not** here.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.theAttentler = CausalMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.theFFler = FeedForward(d_model, d_ff, dropout)
        # raise NotImplementedError("TODO: Implement TransformerBlock.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.theAttentler(RMSNorm(x))
        return x + self.theFFler(RMSNorm(x))
        # raise NotImplementedError("TODO: Implement TransformerBlock.forward()")


class TransformerLM(nn.Module):
    """Decoder-only transformer language model.

    Required attributes:
        token_emb: ``Embedding(vocab_size, d_model)``
        blocks: ``nn.ModuleList`` of ``TransformerBlock``
        ln_final: ``RMSNorm(d_model)``
        lm_head: ``Linear(d_model, vocab_size, bias=False)``
        context_length: store ``context_length`` so ``generate()`` and the
            overlength assertion in ``forward`` can read it.

    Weight tying: ``self.lm_head.weight = self.token_emb.weight``
    (must be the **same** ``nn.Parameter`` object, set in ``__init__``).

    Zero-init: after creating blocks, zero ``o_proj.weight`` and
    ``w_down.weight`` in every block so blocks start as identity.

    ``forward`` must raise ``AssertionError`` when the input sequence is
    longer than ``context_length``.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        context_length: Maximum sequence length.
        d_model: Embedding / hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        d_ff: FFN hidden dimension per block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_emb = Embedding(vocab_size, d_model)
        blocks = []
        for _ in range(n_layers):
            cur = TransformerBlock(d_model, n_heads, d_ff, dropout)
            cur.theAttentler.o_proj *= 0
            cur.theFFler.W_down.zero_()  # ! AI USE asked if *= 0 would set all to 0 and it was like do zero_() clown
            blocks.append(cur)
        self.blocks = nn.ModuleList(blocks)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.context_length = context_length
        # TODO: check this is how tying works
        self.lm_head.weights = self.token_emb.theEmbedler  # dude why the heck are my variable names like this lmaooooo, are you actually reading the code or like am I just talking to myself here
        # raise NotImplementedError("TODO: Implement TransformerLM.__init__()")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: ``(B, T)`` integer token IDs.
        Returns:
            ``(B, T, vocab_size)`` raw logits.
        """
        X = self.token_emb(input_ids)
        for block in self.blocks:
            X = block(X)
        X = self.ln_final(X)
        return self.lm_head(X)

        # raise NotImplementedError("TODO: Implement TransformerLM.forward()")
