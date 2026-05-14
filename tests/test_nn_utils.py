"""Tests for softmax, activation functions, and cross-entropy loss."""

import math
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from tests.adapters import run_cross_entropy_loss, run_silu, run_softmax


@contextmanager
def _block_torch_shortcuts():
    """Temporarily replace torch shortcuts that would let students delegate
    instead of implementing the math from scratch. Anything that calls these
    functions raises a clear AssertionError.

    Crucially, we also drop and re-import student modules WITHIN the patch
    context so any module-level alias (e.g., `_F = torch.nn.functional`) gets
    rebuilt with the blocked attributes — closing the alias-caching bypass.
    """
    import importlib
    import sys as _sys

    def make_blocker(name):
        def _blocked(*args, **kwargs):
            raise AssertionError(
                f"Implementation called the forbidden shortcut `{name}` — "
                "the from-scratch primitives must not delegate to PyTorch's "
                "built-in softmax/silu/cross_entropy/etc."
            )
        return _blocked

    blocked = {}
    # Resolve potentially-missing internal modules once.
    _torch_C = getattr(torch, '_C', None)
    _torch_C_nn = getattr(_torch_C, '_nn', None) if _torch_C else None
    _torch_VF = getattr(torch, '_VF', None)
    _torch_functional = getattr(torch, 'functional', None)
    _torch_refs = getattr(torch, '_refs', None)
    _torch_ops_aten = None
    try:
        _torch_ops_aten = torch.ops.aten
    except Exception:
        _torch_ops_aten = None

    targets = [
        # torch.nn.functional public API
        (torch.nn.functional, 'softmax'),
        (torch.nn.functional, 'log_softmax'),
        (torch.nn.functional, 'silu'),
        (torch.nn.functional, 'gelu'),
        (torch.nn.functional, 'cross_entropy'),
        (torch.nn.functional, 'nll_loss'),
        (torch.nn.functional, 'linear'),
        (torch.nn.functional, 'embedding'),
        (torch.nn.functional, 'scaled_dot_product_attention'),
        # torch top-level
        (torch, 'softmax'),
        (torch, 'log_softmax'),
        (torch, 'logsumexp'),
        (torch, 'cross_entropy'),
        (torch, 'nll_loss'),
        # torch.Tensor methods (e.g., x.softmax(dim=-1))
        (torch.Tensor, 'softmax'),
        (torch.Tensor, 'log_softmax'),
        # torch._C._nn internal C bindings (also a bypass path)
        (_torch_C_nn, 'cross_entropy_loss'),
        (_torch_C_nn, 'log_softmax'),
        (_torch_C_nn, 'softmax'),
        (_torch_C_nn, 'linear'),
        (_torch_C_nn, 'silu'),
        (_torch_C_nn, 'scaled_dot_product_attention'),
        # torch._VF backdoor (used by some F.* helpers)
        (_torch_VF, 'log_softmax'),
        (_torch_VF, 'softmax'),
        (_torch_VF, 'logsumexp'),
        # torch.functional module (different from torch.nn.functional!)
        (_torch_functional, 'softmax'),
        (_torch_functional, 'log_softmax'),
        # torch._refs (Python reference implementations; same effect)
        (_torch_refs, 'softmax'),
        (_torch_refs, 'log_softmax'),
        (_torch_refs, 'cross_entropy'),
        (_torch_refs, 'nll_loss'),
        # rms_norm sometimes available
        (torch.nn.functional, 'rms_norm') if hasattr(torch.nn.functional, 'rms_norm') else None,
        # torch.ops.aten.* dispatch path (private but reachable)
        (_torch_ops_aten, 'softmax') if _torch_ops_aten else None,
        (_torch_ops_aten, 'log_softmax') if _torch_ops_aten else None,
        (_torch_ops_aten, 'logsumexp') if _torch_ops_aten else None,
        (_torch_ops_aten, 'silu') if _torch_ops_aten else None,
        (_torch_ops_aten, 'gelu') if _torch_ops_aten else None,
        (_torch_ops_aten, 'cross_entropy_loss') if _torch_ops_aten else None,
        (_torch_ops_aten, 'nll_loss') if _torch_ops_aten else None,
        # Pretrained-weight serialization
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
                # Some C-level attrs are read-only; skip silently.
                continue

    # Drop cached student modules so module-level `from torch.nn.functional
    # import softmax as _SM` rebinds to the blocker on reimport.
    student_mods = [m for m in list(_sys.modules) if m.startswith('transformer_lm')]
    cached = {m: _sys.modules.pop(m) for m in student_mods}

    try:
        # Reimport at least the modules under test so cached aliases rebind.
        # The actual reimport happens lazily when the test calls run_*
        # (adapters import inside each function), so this just ensures the
        # next adapter call gets a fresh module.
        yield
    finally:
        for (module, attr), original in blocked.items():
            try:
                setattr(module, attr, original)
            except (TypeError, AttributeError):
                continue
        # Restore cached student modules so subsequent tests see the same
        # imports they always did.
        for m, mod in cached.items():
            _sys.modules[m] = mod
        for m in [m for m in list(_sys.modules) if m.startswith('transformer_lm') and m not in cached]:
            _sys.modules.pop(m, None)


# ===========================================================================
# Softmax
# ===========================================================================


def test_softmax_correctness():
    torch.manual_seed(42)
    # (a) random (3,5) tensor with dim=-1
    x1 = torch.randn(3, 5)
    torch.testing.assert_close(
        run_softmax(x1, dim=-1), torch.softmax(x1, dim=-1), atol=1e-5, rtol=1e-5
    )
    # (b) random (3,4) tensor with dim=0
    x2 = torch.randn(3, 4)
    torch.testing.assert_close(
        run_softmax(x2, dim=0), torch.softmax(x2, dim=0), atol=1e-5, rtol=1e-5
    )
    # (c) 1D tensor
    x3 = torch.randn(6)
    torch.testing.assert_close(
        run_softmax(x3, dim=-1), torch.softmax(x3, dim=-1), atol=1e-5, rtol=1e-5
    )


def test_softmax_stability():
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out = run_softmax(x, dim=-1)
    assert not torch.isnan(out).any(), "softmax produced NaN on large inputs"
    assert not torch.isinf(out).any(), "softmax produced Inf on large inputs"
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


def test_softmax_shift_invariance():
    torch.manual_seed(42)
    x = torch.randn(3, 5)
    out_original = run_softmax(x, dim=-1)
    out_shifted = run_softmax(x + 1000.0, dim=-1)
    torch.testing.assert_close(out_original, out_shifted, atol=1e-5, rtol=1e-5)


# ===========================================================================
# SiLU
# ===========================================================================


def test_silu_correctness():
    torch.manual_seed(42)
    x = torch.randn(4, 5)
    torch.testing.assert_close(run_silu(x), F.silu(x), atol=1e-5, rtol=1e-5)
    # Assert NOT close to GELU at probe points where they differ
    probe = torch.tensor([-2.0, -1.0, 0.5, 1.0, 2.0])
    silu_out = run_silu(probe)
    gelu_out = F.gelu(probe, approximate="tanh")
    assert not torch.allclose(
        silu_out, gelu_out, atol=1e-2
    ), "SiLU should differ from GELU at probe points"


# ===========================================================================
# Cross-entropy loss
# ===========================================================================


def test_cross_entropy_correctness():
    torch.manual_seed(42)
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    loss = run_cross_entropy_loss(logits, targets)
    expected = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)


def test_cross_entropy_stability():
    torch.manual_seed(42)
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V) * 1000
    targets = torch.randint(0, V, (B, T))
    loss = run_cross_entropy_loss(logits, targets)
    assert not torch.isnan(loss).any(), "cross-entropy produced NaN on scaled logits"
    expected = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    torch.testing.assert_close(loss, expected, atol=1e-3, rtol=1e-3)


def test_cross_entropy_uniform_logits():
    V = 10
    logits = torch.zeros(1, 1, V)
    targets = torch.zeros(1, 1, dtype=torch.long)
    loss = run_cross_entropy_loss(logits, targets)
    expected = torch.tensor(math.log(V))
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)


def test_cross_entropy_shift_invariance():
    torch.manual_seed(42)
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    loss_original = run_cross_entropy_loss(logits, targets)
    loss_shifted = run_cross_entropy_loss(logits + 1000.0, targets)
    torch.testing.assert_close(loss_original, loss_shifted, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Anti-cheat: ensure from-scratch primitives don't delegate to PyTorch
# ===========================================================================


def test_softmax_implements_math_from_scratch():
    """softmax() must not delegate to torch.softmax / F.softmax / etc."""
    torch.manual_seed(42)
    x = torch.randn(3, 5)
    with _block_torch_shortcuts():
        out = run_softmax(x, dim=-1)
    # Verify it's still a valid softmax
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


def test_silu_implements_math_from_scratch():
    """silu() must not delegate to F.silu / torch.nn.functional.silu."""
    torch.manual_seed(42)
    x = torch.randn(4, 5)
    with _block_torch_shortcuts():
        out = run_silu(x)
    torch.testing.assert_close(out, F.silu(x), atol=1e-5, rtol=1e-5)


def test_cross_entropy_implements_math_from_scratch():
    """cross_entropy_loss() must not delegate to F.cross_entropy / F.nll_loss."""
    torch.manual_seed(42)
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    with _block_torch_shortcuts():
        loss = run_cross_entropy_loss(logits, targets)
    expected = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
