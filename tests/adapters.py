"""Adapter functions connecting student implementations to the test suite.

This file is pre-filled and you should not need to modify it.
It routes test calls to the implementations in transformer_lm/.
"""

from __future__ import annotations

import torch


# ==========================================================================
# Part 1: Tokenizer
# ==========================================================================


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer. Return (vocab, merges)."""
    from transformer_lm.tokenizer import train_bpe
    return train_bpe(input_path, vocab_size, special_tokens)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[int, int]],
    special_tokens: list[str] | None = None,
):
    """Return a BPETokenizer instance."""
    from transformer_lm.tokenizer import BPETokenizer
    return BPETokenizer(vocab, merges, special_tokens)


# ==========================================================================
# Part 2: Model components
# ==========================================================================


def run_linear(
    in_features: int,
    out_features: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a from-scratch Linear layer with the given weight and bias."""
    from transformer_lm.model import Linear
    layer = Linear(in_features, out_features, bias=bias is not None)
    layer.weight = torch.nn.Parameter(weight)
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias)
    return layer(x)


def run_embedding(
    num_embeddings: int,
    embedding_dim: int,
    weight: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Run a from-scratch Embedding layer with the given weight."""
    from transformer_lm.model import Embedding
    emb = Embedding(num_embeddings, embedding_dim)
    emb.weight = torch.nn.Parameter(weight)
    return emb(indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run RMSNorm (no learnable params) and multiply by weight externally."""
    from transformer_lm.model import RMSNorm
    norm = RMSNorm(d_model, eps)
    return norm(x) * weight


def run_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Run the from-scratch softmax."""
    from transformer_lm.nn_utils import softmax
    return softmax(x, dim)


def run_silu(x: torch.Tensor) -> torch.Tensor:
    """Run the from-scratch SiLU."""
    from transformer_lm.nn_utils import silu
    return silu(x)



def run_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run scaled dot-product attention."""
    from transformer_lm.model import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)


def run_causal_multi_head_self_attention(
    d_model: int,
    n_heads: int,
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    """Run CausalMultiHeadSelfAttention with given weights.

    Accepts separate Q, K, V weights for test compatibility and
    packs them into the fused qkv_proj weight.
    """
    from transformer_lm.model import CausalMultiHeadSelfAttention
    attn = CausalMultiHeadSelfAttention(d_model, n_heads)
    # Pack Q, K, V weights into the fused qkv_proj
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    attn.qkv_proj.weight = torch.nn.Parameter(qkv_weight)
    attn.o_proj.weight = torch.nn.Parameter(o_weight)
    return attn(x)


def run_feed_forward(
    d_model: int,
    d_ff: int,
    x: torch.Tensor,
    w_gate_weight: torch.Tensor,
    w_up_weight: torch.Tensor,
    w_down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run FeedForward (SwiGLU) with given weights."""
    from transformer_lm.model import FeedForward
    ffn = FeedForward(d_model, d_ff)
    ffn.w_gate.weight = torch.nn.Parameter(w_gate_weight)
    ffn.w_up.weight = torch.nn.Parameter(w_up_weight)
    ffn.w_down.weight = torch.nn.Parameter(w_down_weight)
    return ffn(x)


def run_transformer_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    x: torch.Tensor,
    block_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run a TransformerBlock with given parameters."""
    from transformer_lm.model import TransformerBlock
    block = TransformerBlock(d_model, n_heads, d_ff)
    block.load_state_dict(block_params)
    return block(x)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    input_ids: torch.Tensor,
    model_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run TransformerLM with given parameters."""
    from transformer_lm.model import TransformerLM
    model = TransformerLM(vocab_size, context_length, d_model, n_layers, n_heads, d_ff)
    model.load_state_dict(model_params)
    return model(input_ids)


# ==========================================================================
# Part 3: Loss
# ==========================================================================


def run_cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Run the from-scratch cross-entropy loss."""
    from transformer_lm.nn_utils import cross_entropy_loss
    return cross_entropy_loss(logits, targets)


# ==========================================================================
# Part 4: Training utilities
# ==========================================================================


def run_get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch."""
    from transformer_lm.training_utils import get_batch
    return get_batch(data, batch_size, context_length, device)


def run_generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Generate tokens autoregressively."""
    from transformer_lm.training_utils import generate
    return generate(model, prompt_ids, max_new_tokens, temperature, context_length)
