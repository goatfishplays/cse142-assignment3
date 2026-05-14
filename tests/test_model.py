"""Comprehensive tests for transformer model components.

Each test uses adapter functions from tests/adapters.py so the test suite
is decoupled from the student implementation's internal API.
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from tests.adapters import (
    run_causal_multi_head_self_attention,
    run_embedding,
    run_feed_forward,
    run_linear,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
)


@contextmanager
def _block_torch_shortcuts():
    """Block PyTorch built-ins that students must not delegate to. Also drops
    cached `transformer_lm.*` modules so module-level alias caches rebind."""
    import sys as _sys

    def make_blocker(name):
        def _blocked(*args, **kwargs):
            raise AssertionError(
                f"Implementation called the forbidden shortcut `{name}` — "
                "the from-scratch primitive must implement the math itself."
            )
        return _blocked

    blocked = {}
    _torch_C = getattr(torch, '_C', None)
    _nn = getattr(_torch_C, '_nn', None) if _torch_C else None
    _torch_VF = getattr(torch, '_VF', None)
    _torch_functional = getattr(torch, 'functional', None)
    _torch_refs = getattr(torch, '_refs', None)
    _torch_decomp = getattr(torch, '_decomp', None)
    _torch_ops = getattr(torch, 'ops', None)
    _torch_ops_aten = getattr(_torch_ops, 'aten', None) if _torch_ops else None
    targets = [
        (torch.nn.functional, 'softmax'),
        (torch.nn.functional, 'log_softmax'),
        (torch.nn.functional, 'silu'),
        (torch.nn.functional, 'gelu'),
        (torch.nn.functional, 'cross_entropy'),
        (torch.nn.functional, 'nll_loss'),
        (torch.nn.functional, 'linear'),
        (torch.nn.functional, 'embedding'),
        (torch.nn.functional, 'scaled_dot_product_attention'),
        (torch, 'softmax'),
        (torch, 'log_softmax'),
        (torch, 'logsumexp'),
        (torch, 'cross_entropy'),
        (torch, 'nll_loss'),
        (torch.Tensor, 'softmax'),
        (torch.Tensor, 'log_softmax'),
        (_nn, 'cross_entropy_loss') if _nn else None,
        (_nn, 'log_softmax') if _nn else None,
        (_nn, 'softmax') if _nn else None,
        (_nn, 'linear') if _nn else None,
        (_nn, 'silu') if _nn else None,
        (_nn, 'scaled_dot_product_attention') if _nn else None,
        # torch._VF backdoor
        (_torch_VF, 'log_softmax') if _torch_VF else None,
        (_torch_VF, 'softmax') if _torch_VF else None,
        (_torch_VF, 'logsumexp') if _torch_VF else None,
        # torch.functional (NOT torch.nn.functional — a separate module)
        (_torch_functional, 'softmax') if _torch_functional else None,
        (_torch_functional, 'log_softmax') if _torch_functional else None,
        # torch._refs Python reference impls
        (_torch_refs, 'softmax') if _torch_refs else None,
        (_torch_refs, 'log_softmax') if _torch_refs else None,
        (_torch_refs, 'cross_entropy') if _torch_refs else None,
        (_torch_refs, 'nll_loss') if _torch_refs else None,
        # torch._decomp decomposition table
        (_torch_decomp, 'softmax') if _torch_decomp else None,
        (_torch_decomp, 'log_softmax') if _torch_decomp else None,
        # torch.ops.aten.* dispatch path
        (_torch_ops_aten, 'softmax') if _torch_ops_aten else None,
        (_torch_ops_aten, 'log_softmax') if _torch_ops_aten else None,
        (_torch_ops_aten, 'logsumexp') if _torch_ops_aten else None,
        (_torch_ops_aten, 'silu') if _torch_ops_aten else None,
        (_torch_ops_aten, 'gelu') if _torch_ops_aten else None,
        (_torch_ops_aten, 'linear') if _torch_ops_aten else None,
        (_torch_ops_aten, 'embedding') if _torch_ops_aten else None,
        (_torch_ops_aten, 'cross_entropy_loss') if _torch_ops_aten else None,
        (_torch_ops_aten, 'nll_loss') if _torch_ops_aten else None,
        (_torch_ops_aten, 'scaled_dot_product_attention') if _torch_ops_aten else None,
        # Pretrained-weight serialization paths
        (torch, 'load'),
        (torch, 'save'),
        (torch, 'from_file'),
    ]
    targets = [t for t in targets if t is not None and t[0] is not None]

    for module, attr in targets:
        if hasattr(module, attr):
            try:
                blocked[(module, attr)] = getattr(module, attr)
                setattr(module, attr, make_blocker(f"{getattr(module, '__name__', repr(module))}.{attr}"))
            except (TypeError, AttributeError):
                continue

    student_mods = [m for m in list(_sys.modules) if m.startswith('transformer_lm')]
    cached = {m: _sys.modules.pop(m) for m in student_mods}

    try:
        yield
    finally:
        for (module, attr), original in blocked.items():
            try:
                setattr(module, attr, original)
            except (TypeError, AttributeError):
                continue
        for m, mod in cached.items():
            _sys.modules[m] = mod
        for m in [m for m in list(_sys.modules) if m.startswith('transformer_lm') and m not in cached]:
            _sys.modules.pop(m, None)


@pytest.fixture(autouse=True)
def seed():
    """Set the random seed before every test for reproducibility."""
    torch.manual_seed(42)


# ======================================================================
# Linear (2 tests)
# ======================================================================


class TestLinear:
    def test_linear_correctness(self):
        """Inject weight/bias and compare against manual x @ W.T + b."""
        in_features, out_features = 8, 16
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        # (a) 2D input with bias
        x_2d = torch.randn(3, in_features)
        out = run_linear(in_features, out_features, x_2d, weight, bias)
        expected = x_2d @ weight.T + bias
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        # (b) 2D without bias
        out_no_bias = run_linear(in_features, out_features, x_2d, weight, bias=None)
        expected_no_bias = x_2d @ weight.T
        torch.testing.assert_close(out_no_bias, expected_no_bias, atol=1e-5, rtol=1e-5)

        # (c) 3D input with bias
        x_3d = torch.randn(2, 5, in_features)
        out_3d = run_linear(in_features, out_features, x_3d, weight, bias)
        expected_3d = x_3d @ weight.T + bias
        torch.testing.assert_close(out_3d, expected_3d, atol=1e-5, rtol=1e-5)

    def test_linear_weight_is_parameter(self):
        """Verify weight/bias are nn.Parameter and parameter counts are correct."""
        from transformer_lm.model import Linear

        # With bias
        layer = Linear(8, 16)
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert len(list(layer.parameters())) == 2

        # Without bias
        layer_no_bias = Linear(8, 16, bias=False)
        assert len(list(layer_no_bias.parameters())) == 1

    def test_linear_init_range(self):
        """Spec: weights are uniform in [-1/sqrt(d_in), 1/sqrt(d_in)], bias=0.
        Tested over many instantiations to defeat seed luck."""
        import math
        from transformer_lm.model import Linear

        d_in, d_out = 64, 128
        bound = 1.0 / math.sqrt(d_in)
        for _ in range(5):
            layer = Linear(d_in, d_out)
            w = layer.weight.detach()
            assert w.min() >= -bound - 1e-6 and w.max() <= bound + 1e-6, (
                f"Linear weights must lie in [-{bound:.4f}, {bound:.4f}], "
                f"got [{w.min().item():.4f}, {w.max().item():.4f}]"
            )
            assert layer.bias is not None
            torch.testing.assert_close(
                layer.bias.detach(), torch.zeros_like(layer.bias),
                msg="Linear bias must be initialized to zero",
            )


# ======================================================================
# Embedding (2 tests)
# ======================================================================


class TestEmbedding:
    def test_embedding_correctness(self):
        """Compare against direct indexing weight[indices]."""
        num_embeddings, embedding_dim = 100, 32
        weight = torch.randn(num_embeddings, embedding_dim)

        # 1D indices
        indices = torch.tensor([3, 0, 99, 42])
        out = run_embedding(num_embeddings, embedding_dim, weight, indices)
        expected = weight[indices]
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        # 2D indices
        indices_2d = torch.randint(0, num_embeddings, (2, 7))
        out_2d = run_embedding(num_embeddings, embedding_dim, weight, indices_2d)
        expected_2d = weight[indices_2d]
        torch.testing.assert_close(out_2d, expected_2d, atol=1e-5, rtol=1e-5)

    def test_embedding_init_distribution(self):
        """Spec: weights are sampled from N(0, 0.02). Use a large embedding
        to make the std estimate reliable."""
        from transformer_lm.model import Embedding

        emb = Embedding(2000, 64)
        w = emb.weight.detach()
        std = w.std().item()
        mean = w.mean().item()
        assert 0.015 <= std <= 0.026, (
            f"Embedding weight std should be ~0.02, got {std:.4f}. "
            f"Did you use N(0, 0.02)?"
        )
        assert abs(mean) < 0.005, (
            f"Embedding weight mean should be ~0, got {mean:.4f}"
        )


# ======================================================================
# RMSNorm (6 tests)
# ======================================================================


class TestRMSNorm:
    def test_rmsnorm_correctness(self):
        """Known input [[1,2,3,4]], d_model=4, weight=ones. Manual RMS computation."""
        d_model = 4
        eps = 1e-5
        weight = torch.ones(d_model)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        out = run_rmsnorm(d_model, eps, weight, x)

        # mean(x^2) = (1+4+9+16)/4 = 7.5
        # rms = sqrt(7.5 + 1e-5)
        rms = math.sqrt(7.5 + eps)
        expected = x / rms
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_not_layernorm(self):
        """With non-zero-mean input and non-unit weight, output differs from LayerNorm."""
        d_model = 4
        eps = 1e-5
        weight = torch.tensor([1.0, 2.0, 0.5, 3.0])
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        out = run_rmsnorm(d_model, eps, weight, x)

        # RMSNorm expected: (x / rms) * weight
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        rmsnorm_expected = (x / rms) * weight
        torch.testing.assert_close(out, rmsnorm_expected, atol=1e-5, rtol=1e-5)

        # LayerNorm-style: (x - mean) / rms * weight -- should differ
        mean = x.mean(dim=-1, keepdim=True)
        layernorm_result = ((x - mean) / rms) * weight
        assert not torch.allclose(out, layernorm_result, atol=1e-3), (
            "Output matches LayerNorm -- RMSNorm should NOT subtract the mean"
        )

    def test_rmsnorm_no_learnable_params(self):
        """RMSNorm should have zero learnable parameters."""
        from transformer_lm.model import RMSNorm

        norm = RMSNorm(64)
        assert len(list(norm.parameters())) == 0

    def test_rmsnorm_batched_3d(self):
        """Multi-row 3D input: normalization must operate along last dim only."""
        d_model = 4
        eps = 1e-5
        weight = torch.ones(d_model)
        x = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [10.0, 20.0, 30.0, 40.0]],
            [[0.1, 0.2, 0.3, 0.4],
             [5.0, 5.0, 5.0, 5.0]],
        ])  # (2, 2, 4) — rows have very different scales

        out = run_rmsnorm(d_model, eps, weight, x)

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        expected = (x / rms) * weight
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_zero_input(self):
        """All-zero input must not produce NaN (requires epsilon in denominator)."""
        d_model = 4
        eps = 1e-5
        weight = torch.ones(d_model)
        x = torch.zeros(2, 3, d_model)

        out = run_rmsnorm(d_model, eps, weight, x)
        assert not torch.isnan(out).any(), (
            "RMSNorm produced NaN on zero input -- epsilon may be missing"
        )
        assert not torch.isinf(out).any(), (
            "RMSNorm produced Inf on zero input"
        )

    def test_rmsnorm_eps_inside_sqrt(self):
        """Distinguish ``x / sqrt(mean(x^2) + eps)`` (correct) from
        ``x / (sqrt(mean(x^2)) + eps)`` (wrong). On a tiny-norm input the
        two formulas diverge by several orders of magnitude."""
        d_model = 4
        eps = 1e-5
        weight = torch.ones(d_model)
        # Tiny uniform input — mean(x^2) is much smaller than eps, so the
        # `eps`-inside-sqrt form dominates.
        x = torch.full((1, d_model), 1e-3)

        out = run_rmsnorm(d_model, eps, weight, x)

        # Correct (eps inside sqrt): rms = sqrt(1e-6 + 1e-5) ≈ 3.317e-3,
        # output ≈ x / 3.317e-3 ≈ 0.3015.
        # Wrong (eps outside sqrt): rms_wrong = sqrt(1e-6) + 1e-5 = 1.01e-3,
        # output ≈ x / 1.01e-3 ≈ 0.99 — about 3× larger.
        rms_correct = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        expected_correct = x / rms_correct
        torch.testing.assert_close(
            out, expected_correct, atol=1e-6, rtol=1e-4
        )


# ======================================================================
# Attention (4 tests)
# ======================================================================


class TestScaledDotProductAttention:
    def test_attention_correctness(self):
        """B=1, T=4, d_k=16. No mask. Manual softmax(QK^T/sqrt(d_k)) @ V."""
        B, T, d_k = 1, 4, 16
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)

        out = run_scaled_dot_product_attention(Q, K, V)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        expected = weights @ V
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_attention_causal_mask(self):
        """V = identity, causal mask. out[0,i,j] ~= 0 for j > i, and the
        lower triangle must NOT be all zero (catches a stubbed
        ``return zeros_like(V)`` that would trivially pass the upper-zero
        check)."""
        B, T, d_k = 1, 5, 8
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.eye(T, dtype=torch.float32).unsqueeze(0)  # (1, T, T)

        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        out = run_scaled_dot_product_attention(Q, K, V, mask)

        for i in range(T):
            for j in range(i + 1, T):
                assert out[0, i, j].item() == pytest.approx(0.0, abs=1e-5), (
                    f"out[0, {i}, {j}] should be ~0 but got {out[0, i, j].item()}"
                )
        # Lower triangle must contain real attention weights. With V=I,
        # row i's first (i+1) entries are the softmax distribution; they
        # cannot all be zero, and their sum should be ≈ 1.
        for i in range(T):
            row_sum = out[0, i, : i + 1].sum().item()
            assert row_sum == pytest.approx(1.0, abs=1e-4), (
                f"out[0, {i}, :{i+1}].sum() should be ≈ 1 (a probability "
                f"distribution over visible positions), got {row_sum}"
            )

    def test_attention_scaling(self):
        """Use d_k=64. Compare student to manual softmax(QK^T/8) @ V."""
        B, T, d_k = 1, 4, 64
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)

        out = run_scaled_dot_product_attention(Q, K, V)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # divide by 8
        weights = torch.softmax(scores, dim=-1)
        expected = weights @ V
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        # Without scaling, result would differ dramatically
        scores_unscaled = Q @ K.transpose(-2, -1)
        weights_unscaled = torch.softmax(scores_unscaled, dim=-1)
        expected_unscaled = weights_unscaled @ V
        assert not torch.allclose(out, expected_unscaled, atol=1e-2), (
            "Output matches unscaled attention -- sqrt(d_k) scaling is missing"
        )

    def test_attention_causal_mask_oracle(self):
        """Full masked-attention oracle with non-identity V."""
        B, T, d_k = 1, 4, 8
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)

        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        out = run_scaled_dot_product_attention(Q, K, V, mask)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        expected = weights @ V
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ======================================================================
# Multi-Head Attention (3 tests)
# ======================================================================


class TestCausalMultiHeadSelfAttention:
    def test_mha_causal_prefix_invariance(self):
        """Modifying last 2 positions must not change positions 0..T-3."""
        d_model, n_heads, T = 16, 2, 6
        B = 1
        x = torch.randn(B, T, d_model)

        q_weight = torch.randn(d_model, d_model)
        k_weight = torch.randn(d_model, d_model)
        v_weight = torch.randn(d_model, d_model)
        o_weight = torch.randn(d_model, d_model)

        out_original = run_causal_multi_head_self_attention(
            d_model, n_heads, x, q_weight, k_weight, v_weight, o_weight
        )

        # Modify last 2 positions
        x_modified = x.clone()
        x_modified[0, -2:, :] = torch.randn(2, d_model)

        out_modified = run_causal_multi_head_self_attention(
            d_model, n_heads, x_modified, q_weight, k_weight, v_weight, o_weight
        )

        # Positions 0..T-3 should be unchanged
        torch.testing.assert_close(
            out_original[0, : T - 2, :],
            out_modified[0, : T - 2, :],
            atol=1e-5,
            rtol=1e-5,
        )

        # Position T-1 should change
        assert not torch.allclose(
            out_original[0, -1, :], out_modified[0, -1, :], atol=1e-5
        ), "Last position output should change when its input changes"

    def test_mha_validates_head_geometry(self):
        """d_model=15, n_heads=4 (15%4!=0) should raise. d_model=12, n_heads=4 (d_head=3 odd) should raise."""
        from transformer_lm.model import CausalMultiHeadSelfAttention

        with pytest.raises(AssertionError):
            CausalMultiHeadSelfAttention(15, 4)

        with pytest.raises(AssertionError):
            CausalMultiHeadSelfAttention(12, 4)

    def test_mha_numerical_correctness(self):
        """Full oracle: QKV split, head reshape, RoPE, causal attention, concat, output proj."""
        from transformer_lm.model import RotaryPositionEmbedding

        d_model, n_heads = 8, 2
        d_head = d_model // n_heads
        B, T = 1, 4

        x = torch.randn(B, T, d_model)
        q_weight = torch.randn(d_model, d_model)
        k_weight = torch.randn(d_model, d_model)
        v_weight = torch.randn(d_model, d_model)
        o_weight = torch.randn(d_model, d_model)

        out = run_causal_multi_head_self_attention(
            d_model, n_heads, x, q_weight, k_weight, v_weight, o_weight
        )

        # Manual oracle
        Q = x @ q_weight.T
        K = x @ k_weight.T
        V = x @ v_weight.T

        Q = Q.view(B, T, n_heads, d_head).transpose(1, 2)
        K = K.view(B, T, n_heads, d_head).transpose(1, 2)
        V = V.view(B, T, n_heads, d_head).transpose(1, 2)

        rope = RotaryPositionEmbedding(d_head)
        Q, K = rope(Q, K)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ V

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_model)
        expected = attn_out @ o_weight.T

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ======================================================================
# FeedForward (2 tests)
# ======================================================================


class TestFeedForward:
    def test_ffn_correctness(self):
        """Inject weights, compare to PyTorch F.silu-based oracle."""
        B, T, d_model, d_ff = 2, 8, 32, 64
        x = torch.randn(B, T, d_model)
        w_gate_weight = torch.randn(d_ff, d_model)
        w_up_weight = torch.randn(d_ff, d_model)
        w_down_weight = torch.randn(d_model, d_ff)

        out = run_feed_forward(d_model, d_ff, x, w_gate_weight, w_up_weight, w_down_weight)

        # Oracle uses PyTorch's F.silu -- NOT the student's silu
        expected = (F.silu(x @ w_gate_weight.T) * (x @ w_up_weight.T)) @ w_down_weight.T
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_ffn_uses_silu_not_gelu(self):
        """Compute GELU-based variant and assert student output differs."""
        B, T, d_model, d_ff = 2, 8, 32, 64
        x = torch.randn(B, T, d_model)
        w_gate_weight = torch.randn(d_ff, d_model)
        w_up_weight = torch.randn(d_ff, d_model)
        w_down_weight = torch.randn(d_model, d_ff)

        out = run_feed_forward(d_model, d_ff, x, w_gate_weight, w_up_weight, w_down_weight)

        # GELU-based variant
        gelu_expected = (
            F.gelu(x @ w_gate_weight.T, approximate="tanh") * (x @ w_up_weight.T)
        ) @ w_down_weight.T
        diff = (out - gelu_expected).abs().max().item()
        assert diff > 1e-3, (
            f"FFN output matches GELU variant (max diff={diff}) -- should use SiLU"
        )


# ======================================================================
# TransformerBlock (2 tests)
# ======================================================================


class TestTransformerBlock:
    def _make_block_params(self, d_model, n_heads, d_ff):
        """Create a TransformerBlock and return its state_dict."""
        from transformer_lm.model import TransformerBlock

        torch.manual_seed(42)
        block = TransformerBlock(d_model, n_heads, d_ff)
        return block.state_dict()

    def test_block_zero_init_identity(self):
        """Directly instantiate TransformerBlock. With zero-init o_proj/w_down,
        block output should equal input (both sublayers output zero)."""
        from transformer_lm.model import TransformerBlock

        block = TransformerBlock(32, 4, 128)
        # Zero-init o_proj and w_down (as TransformerLM does)
        block.attn.o_proj.weight.data.zero_()
        block.ffn.w_down.weight.data.zero_()
        block.eval()

        x = torch.randn(2, 8, 32)
        with torch.no_grad():
            out = block(x)
        assert torch.allclose(out, x, atol=1e-6), (
            "With zero-init o_proj and w_down, block should act as identity"
        )

    def test_block_residual_active(self):
        """Use adapter with _make_block_params. Forward random x. Output should differ from input."""
        d_model, n_heads, d_ff = 32, 4, 128
        block_params = self._make_block_params(d_model, n_heads, d_ff)

        torch.manual_seed(123)
        x = torch.randn(2, 8, d_model)
        out = run_transformer_block(d_model, n_heads, d_ff, x, block_params)

        assert not torch.allclose(out, x, atol=1e-5), (
            "Block output is identical to input -- sublayers should contribute"
        )


# ======================================================================
# TransformerLM (10 tests)
# ======================================================================


class TestTransformerLM:
    def test_weight_tying_identity(self):
        """lm_head.weight is (same object as) token_emb.weight."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(512, 256, 64, 5, 4, 256)
        assert model.lm_head.weight is model.token_emb.weight, (
            "Weight tying failed: lm_head.weight must be the SAME tensor as token_emb.weight"
        )

    def test_weight_tying_mutation(self):
        """Mutating token_emb.weight must be visible through lm_head.weight."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(512, 256, 64, 5, 4, 256)
        model.token_emb.weight.data[0, 0] = 999.0
        assert model.lm_head.weight.data[0, 0] == 999.0, (
            "Mutation of token_emb.weight not reflected in lm_head.weight -- "
            "they are not the same tensor"
        )

    def test_zero_init_projections(self):
        """For each block, o_proj.weight and w_down.weight should be all zeros."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(512, 256, 64, 5, 4, 256)
        for i, block in enumerate(model.blocks):
            assert torch.all(block.attn.o_proj.weight == 0), (
                f"Block {i} o_proj.weight is not all zeros"
            )
            assert torch.all(block.ffn.w_down.weight == 0), (
                f"Block {i} w_down.weight is not all zeros"
            )

    def test_fresh_model_is_embedding_baseline(self):
        """At init, model(ids) should equal lm_head(ln_final(token_emb(ids)))
        because zero-init makes blocks identity."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(64, 16, 32, 2, 2, 64)
        model.eval()

        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            out = model(ids)
            expected = model.lm_head(model.ln_final(model.token_emb(ids)))
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_lm_output_shape_and_param_count(self):
        """Default config: shape (2, 256, 512), param count 360448."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(512, 256, 64, 5, 4, 256)
        x = torch.randint(0, 512, (2, 256))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 256, 512)

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 360448, f"Expected 360448 params, got {n_params}"

    def test_lm_outputs_raw_logits(self):
        """Output should contain negative values (no softmax applied)."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(64, 16, 32, 2, 2, 64)
        model.eval()
        x = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            out = model(x)
        assert out.min().item() < 0, (
            "All logits are non-negative -- model may be applying softmax"
        )

    def test_lm_rejects_overlength_sequence(self):
        """Sequence longer than context_length should raise."""
        from transformer_lm.model import TransformerLM

        context_length = 16
        model = TransformerLM(64, context_length, 32, 2, 2, 64)
        x = torch.randint(0, 64, (1, context_length + 1))
        with pytest.raises(AssertionError):
            model(x)

    def test_gradient_flow(self):
        """Forward + backward with cross_entropy_loss. Every param has non-None grad."""
        from transformer_lm.model import TransformerLM
        from transformer_lm.nn_utils import cross_entropy_loss

        model = TransformerLM(64, 32, 32, 2, 2, 64)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()

        # Names of params behind zero-init gates that may have zero grad
        zero_gate_keywords = {"o_proj", "w_down", "qkv_proj", "w_gate", "w_up"}

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            if any(kw in name for kw in zero_gate_keywords):
                continue
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_no_forbidden_modules(self):
        """Walk named_modules(), check no nn.Linear, nn.LayerNorm, etc."""
        from transformer_lm.model import TransformerLM

        model = TransformerLM(64, 32, 32, 1, 2, 64)
        forbidden = [
            torch.nn.Linear,
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.MultiheadAttention,
            torch.nn.GELU,
            torch.nn.SiLU,
            torch.nn.Softmax,
        ]
        if hasattr(torch.nn, "RMSNorm"):
            forbidden.append(torch.nn.RMSNorm)

        for name, module in model.named_modules():
            for ft in forbidden:
                assert not isinstance(module, ft), (
                    f"{name} uses {ft.__name__} (forbidden)"
                )

    def test_no_forbidden_imports(self):
        """Static check on raw source bytes (NOT inspect.getsource): no
        torch.nn.functional usage in any form, no dynamic import sneaks, no
        torch.__dict__/globals/locals bypasses. Covers all 5 student modules.

        IMPORTANT: we parse the file from disk BEFORE the module is imported
        anywhere in this test, so import-time side effects (e.g., patching
        inspect.getsource itself) cannot hide the cheat.
        """
        import ast
        import importlib.util
        from pathlib import Path

        ALLOWED_NN = {'Parameter', 'Module', 'ModuleList', 'Sequential', 'Dropout'}

        # Forbidden internal/private/back-door entry points off `torch`. Any
        # of these as the second element of an attribute chain rooted at
        # `torch` is rejected.
        FORBIDDEN_TORCH_BRANCHES = {
            'ops', '_C', '_VF', 'functional', '_refs', '_decomp', 'special',
            '_meta_registrations', '_dynamo', '_inductor',
            '__dict__',  # subscript access bypass
        }

        # Direct torch.<func>(...) calls that would substitute for student work.
        FORBIDDEN_TORCH_FUNCS = {
            'softmax', 'log_softmax', 'logsumexp', 'cross_entropy', 'nll_loss',
        }

        # Built-ins that allow dynamic attribute/code access. Also covers
        # reflection paths a creative student might use to defeat the static
        # scanner (e.g., `getattr(torch.nn, "Lin" + "ear")`).
        FORBIDDEN_BUILTINS = {
            '__import__', 'exec', 'eval', 'compile',
            'getattr', 'setattr', 'delattr', 'vars',
            'globals', 'locals', 'dir',
            # `builtins` and `__builtins__` give Name-only access to the
            # forbidden names above — block them at the import level.
            'builtins', '__builtins__',
        }

        # Resolve all 5 student modules to file paths WITHOUT importing them.
        # tokenizer.py legitimately needs file I/O for train_bpe(input_path),
        # but it should NOT have free run of subprocess/network/serialization
        # APIs either — those are bytewise dangerous and unnecessary for BPE.
        IO_BANNED_MODULES = {
            'transformer_lm.nn_utils',
            'transformer_lm.model',
            'transformer_lm.training_utils',
            'transformer_lm.lr_schedule',
        }
        # Tokenizer is allowed to use `open()` and `pathlib` for the corpus
        # path passed to train_bpe(), but nothing else from the danger list.
        TOKENIZER_BANNED_IMPORTS = {
            'urllib', 'requests', 'http', 'socket', 'ssl',
            'ftplib', 'smtplib', 'telnetlib',
            'subprocess', 'multiprocessing',
            'pickle', 'shelve', 'marshal',
            'ctypes',
            'sys', 'inspect', 'gc', 'platform', 'mmap',
            'atexit',
            'builtins',
            # Test / autograder / scripts surface: a malicious tokenizer
            # imported by validate_submission could mutate the staff
            # test harness before pytest runs.
            'tests', 'autograder', 'scripts', 'pytest', 'unittest',
        }
        IO_BANNED_IMPORTS = {
            'urllib', 'requests', 'http', 'socket', 'ssl',
            'ftplib', 'smtplib', 'telnetlib',
            'subprocess', 'multiprocessing',
            'pickle', 'shelve', 'marshal',
            'ctypes',
            # Filesystem / process / introspection at the import level.
            # These were also caught via attribute-chain checks below, but
            # `from os import getcwd` or `import os as o` would bypass that.
            'os', 'pathlib', 'inspect', 'gc', 'platform', 'shutil',
            'glob', 'fnmatch', 'tempfile', 'mmap',
            # sys gives access to argv (junit XML path), modules, _getframe,
            # path, executable, etc. — all useful for cheats and unnecessary
            # for from-scratch model/loss/utils/lr code.
            'sys',
            # atexit lets students forge pytest results.
            'atexit',
            # builtins gives Name-only access to getattr/exec/eval/etc.
            'builtins',
            # Test / autograder / scripts surface: model code must not
            # introspect, mutate, or import its grading harness.
            'tests', 'autograder', 'scripts', 'pytest', 'unittest',
            # Generic decoders that double as encoded-weight smugglers.
            'base64', 'binascii', 'codecs', 'zlib', 'gzip', 'bz2', 'lzma',
            'struct', 'array',
        }
        IO_BANNED_CALLS = {
            'open', 'input', 'breakpoint',
        }
        IO_BANNED_ATTR_CHAINS = {
            ('os',): "os.* (filesystem/process)",
            ('pathlib',): "pathlib (filesystem)",
            ('sys', 'argv'): "sys.argv (introspection)",
            ('sys', 'modules'): "sys.modules (introspection)",
            ('sys', '_getframe'): "sys._getframe (introspection)",
            ('inspect',): "inspect (introspection)",
            ('gc',): "gc (introspection)",
            ('builtins',): "builtins.* (dynamic dispatch)",
            ('__builtins__',): "__builtins__.* (dynamic dispatch)",
            # `torch.save`/`torch.load`/`torch.from_file` are serialization
            # paths a student could use to smuggle pretrained weights or
            # exfiltrate state. None are needed for from-scratch training.
            ('torch', 'save'): "torch.save (serialization)",
            ('torch', 'load'): "torch.load (serialization)",
            ('torch', 'from_file'): "torch.from_file (serialization)",
            ('torch', 'hub'): "torch.hub (pretrained loader)",
            ('torch', 'serialization'): "torch.serialization (pretrained loader)",
            # torch.nn.modules.* and torch.nn.functional.* are subpackage
            # paths to the banned high-level modules; block them too.
            ('torch', 'nn', 'modules'): "torch.nn.modules.* (use the Module class only via torch.nn)",
        }
        module_names = [
            'transformer_lm.nn_utils',
            'transformer_lm.model',
            'transformer_lm.training_utils',
            'transformer_lm.tokenizer',
            'transformer_lm.lr_schedule',
        ]
        for mod_name in module_names:
            spec = importlib.util.find_spec(mod_name)
            assert spec is not None and spec.origin, f"cannot locate {mod_name}"
            source = Path(spec.origin).read_text()
            tree = ast.parse(source)

            is_io_banned = mod_name in IO_BANNED_MODULES
            is_tokenizer = mod_name == 'transformer_lm.tokenizer'
            # Tokenizer gets a narrower import ban (it's allowed to use
            # `open` and `pathlib` for the corpus); everything else gets the
            # full ban.
            active_import_ban = (
                TOKENIZER_BANNED_IMPORTS if is_tokenizer else IO_BANNED_IMPORTS
            )

            if is_io_banned or is_tokenizer:
                # Walk early so we fail fast.
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            head = alias.name.split('.', 1)[0]
                            if head in active_import_ban:
                                raise AssertionError(
                                    f"{mod_name} imports {alias.name} "
                                    "(forbidden — no network / subprocess / "
                                    "serialization / FFI in this module)"
                                )
                    if isinstance(node, ast.ImportFrom) and node.module:
                        head = node.module.split('.', 1)[0]
                        if head in active_import_ban:
                            raise AssertionError(
                                f"{mod_name} imports from {node.module} "
                                "(forbidden — no network / subprocess / "
                                "serialization / FFI in this module)"
                            )
                    if isinstance(node, ast.Call) and is_io_banned:
                        f = node.func
                        if isinstance(f, ast.Name) and f.id in IO_BANNED_CALLS:
                            raise AssertionError(
                                f"{mod_name} calls {f.id}() — forbidden "
                                "(no filesystem access in this module)"
                            )
                    if isinstance(node, ast.Attribute):
                        parts = []
                        cur = node
                        while isinstance(cur, ast.Attribute):
                            parts.append(cur.attr)
                            cur = cur.value
                        if isinstance(cur, ast.Name):
                            parts.append(cur.id)
                            chain = tuple(reversed(parts))
                            # For tokenizer, allow os/pathlib chains (it
                            # legitimately needs the corpus path) but still
                            # ban everything else.
                            tokenizer_chain_exceptions = {('os',), ('pathlib',)}
                            for ban_prefix, desc in IO_BANNED_ATTR_CHAINS.items():
                                if chain[:len(ban_prefix)] == ban_prefix:
                                    if is_tokenizer and ban_prefix in tokenizer_chain_exceptions:
                                        continue
                                    raise AssertionError(
                                        f"{mod_name} uses {desc} "
                                        f"({'.'.join(chain)}) — forbidden in this module"
                                    )

            # First pass: collect import aliases so we can resolve
            # `import torch as T` and then catch `T.ops.aten...` etc.
            # Maps local-name -> canonical-module-name.
            torch_aliases: dict[str, str] = {"torch": "torch"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "torch":
                            torch_aliases[alias.asname or "torch"] = "torch"
                        if alias.name == "torch.nn":
                            torch_aliases[alias.asname or "torch.nn"] = "torch.nn"

            for node in ast.walk(tree):
                # Ban dynamic-access built-ins.
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
                        raise AssertionError(
                            f"{mod_name} uses {func.id}() (forbidden — "
                            "no dynamic attribute access / code execution)"
                        )
                    if (isinstance(func, ast.Attribute)
                            and func.attr in {'import_module', 'reload',
                                              'exec', 'eval', 'compile',
                                              '__getattribute__', '__getattr__',
                                              'attrgetter'}):
                        raise AssertionError(
                            f"{mod_name} calls .{func.attr}(...) "
                            "(forbidden — no dynamic dispatch allowed)"
                        )

                # Ban `torch.__dict__[...]`, `obj.__dict__[...]` subscripting.
                if isinstance(node, ast.Subscript):
                    val = node.value
                    if isinstance(val, ast.Attribute) and val.attr == '__dict__':
                        raise AssertionError(
                            f"{mod_name} uses .__dict__[...] subscript "
                            "(forbidden — dynamic dispatch bypass)"
                        )

                # Ban: import torch.nn.functional / torch._refs / etc.
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in ('torch.nn.functional', 'torch._refs',
                                    'torch._C', 'torch._VF', 'torch.ops',
                                    'torch.functional', 'torch._decomp',
                                    'torch.special', 'operator', 'importlib'):
                            assert bad not in alias.name, (
                                f"{mod_name} imports {alias.name} (forbidden)"
                            )

                if isinstance(node, ast.ImportFrom) and node.module:
                    for bad in ('torch.nn.functional', 'torch._refs',
                                'torch._C', 'torch._VF', 'torch.ops',
                                'torch.functional', 'torch._decomp',
                                'torch.special'):
                        assert bad not in node.module, (
                            f"{mod_name} imports from {node.module} (forbidden)"
                        )
                    if node.module == 'torch.nn':
                        for alias in node.names:
                            assert alias.name in ALLOWED_NN, (
                                f"{mod_name} imports torch.nn.{alias.name} "
                                f"(allowed: {sorted(ALLOWED_NN)})"
                            )
                    if node.module == 'torch':
                        for alias in node.names:
                            # Block back-door imports off `torch`.
                            if alias.name in (FORBIDDEN_TORCH_BRANCHES
                                              | FORBIDDEN_TORCH_FUNCS):
                                raise AssertionError(
                                    f"{mod_name} uses 'from torch import "
                                    f"{alias.name}' (forbidden — would let "
                                    "you call the forbidden surface under "
                                    "an alias)"
                                )
                            assert alias.name not in {'nn'}, (
                                f"{mod_name} uses 'from torch import nn' — "
                                "use the explicit 'import torch.nn as nn'"
                            )
                    if node.module == 'operator':
                        # attrgetter / itemgetter / methodcaller etc. are
                        # dynamic-dispatch helpers; ban wholesale.
                        raise AssertionError(
                            f"{mod_name} imports from operator (forbidden)"
                        )
                    if node.module in {'importlib', 'importlib.util'}:
                        raise AssertionError(
                            f"{mod_name} imports importlib (forbidden — no "
                            "dynamic module loading)"
                        )

                # Ban attribute chains rooted at torch.*.
                if isinstance(node, ast.Attribute):
                    parts = []
                    cur = node
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                        chain = list(reversed(parts))
                        # Resolve alias: if chain[0] is an alias for `torch`
                        # or `torch.nn`, rewrite the chain to canonical form
                        # so `import torch as T; T.ops...` is caught.
                        canonical = torch_aliases.get(chain[0])
                        if canonical == 'torch':
                            chain[0] = 'torch'
                        elif canonical == 'torch.nn':
                            chain = ['torch', 'nn'] + chain[1:]
                        # Block torch.<forbidden-branch>.*
                        if (len(chain) >= 2
                                and chain[0] == 'torch'
                                and chain[1] in FORBIDDEN_TORCH_BRANCHES):
                            raise AssertionError(
                                f"{mod_name} uses {'.'.join(chain)} — "
                                "forbidden (private/back-door torch surface)"
                            )
                        # Block torch.<func>(...) shortcuts.
                        if (len(chain) == 2
                                and chain[0] == 'torch'
                                and chain[1] in FORBIDDEN_TORCH_FUNCS):
                            raise AssertionError(
                                f"{mod_name} calls torch.{chain[1]}(...) — "
                                "forbidden (implement from scratch)"
                            )
                        # Block torch.nn.<not-allowed>
                        if (len(chain) >= 3
                                and chain[0] == 'torch'
                                and chain[1] == 'nn'
                                and chain[2] not in ALLOWED_NN
                                and chain[2] not in {'utils'}):
                            raise AssertionError(
                                f"{mod_name} uses torch.nn.{chain[2]} "
                                f"(allowed: {sorted(ALLOWED_NN)})"
                            )


# ======================================================================
# Anti-cheat runtime tests (3)
# ======================================================================
#
# The AST checks above can be defeated by string-eval, attribute-shenanigans,
# or third-party indirection. These runtime tests block PyTorch's built-ins
# at call time and re-run the from-scratch primitives — anything that secretly
# delegates to torch.nn.functional.* or torch.softmax raises immediately.


class TestAntiCheat:
    def test_linear_implements_math_from_scratch(self):
        in_features, out_features = 8, 16
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        x = torch.randn(3, in_features)
        with _block_torch_shortcuts():
            out = run_linear(in_features, out_features, x, weight, bias)
        torch.testing.assert_close(out, x @ weight.T + bias, atol=1e-5, rtol=1e-5)

    def test_attention_implements_math_from_scratch(self):
        B, T, d_k = 1, 4, 16
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)
        with _block_torch_shortcuts():
            out = run_scaled_dot_product_attention(Q, K, V)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        torch.testing.assert_close(out, weights @ V, atol=1e-5, rtol=1e-5)

    def test_mha_implements_math_from_scratch(self):
        d_model, n_heads = 8, 2
        B, T = 1, 4
        x = torch.randn(B, T, d_model)
        q_w = torch.randn(d_model, d_model)
        k_w = torch.randn(d_model, d_model)
        v_w = torch.randn(d_model, d_model)
        o_w = torch.randn(d_model, d_model)
        with _block_torch_shortcuts():
            run_causal_multi_head_self_attention(
                d_model, n_heads, x, q_w, k_w, v_w, o_w
            )
