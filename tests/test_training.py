"""Tests for training utilities: get_batch and generate."""

import pytest
import torch

from tests.adapters import run_generate, run_get_batch


# ===========================================================================
# get_batch
# ===========================================================================


def test_get_batch_correctness():
    torch.manual_seed(42)
    data = torch.arange(1000)
    x, y = run_get_batch(data, batch_size=4, context_length=16, device="cpu")
    for i in range(4):
        start = x[i, 0].item()
        assert torch.equal(x[i], data[start : start + 16])
        assert torch.equal(y[i], data[start + 1 : start + 17])


def test_get_batch_bounds():
    torch.manual_seed(42)
    data = torch.arange(1000)
    context_length = 16
    x, y = run_get_batch(data, batch_size=8, context_length=context_length, device="cpu")
    assert (x >= 0).all() and (x < len(data)).all()
    assert (y >= 0).all() and (y < len(data)).all()
    for i in range(8):
        start = x[i, 0].item()
        assert start + context_length < len(data)


def test_get_batch_randomness():
    data = torch.arange(100_000)
    x1, y1 = run_get_batch(data, batch_size=4, context_length=16, device="cpu")
    x2, y2 = run_get_batch(data, batch_size=4, context_length=16, device="cpu")
    assert not torch.equal(x1, x2) or not torch.equal(y1, y2)


def test_get_batch_rejects_short_data():
    data = torch.arange(16)
    with pytest.raises(ValueError):
        run_get_batch(data, batch_size=4, context_length=16, device="cpu")


def test_get_batch_device():
    """Returned tensors must be on the requested device."""
    data = torch.arange(1000)
    x, y = run_get_batch(data, batch_size=4, context_length=16, device="cpu")
    assert x.device.type == "cpu", f"x on {x.device}, expected cpu"
    assert y.device.type == "cpu", f"y on {y.device}, expected cpu"


# ===========================================================================
# generate
# ===========================================================================


def _make_model():
    from transformer_lm.model import TransformerLM

    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
    )
    model.eval()
    return model


def test_generate_basic():
    model = _make_model()
    prompt = [1, 2, 3, 4, 5]
    output = run_generate(model, prompt, max_new_tokens=10, temperature=1.0, context_length=32)
    assert len(output) == 15
    assert output[:5] == [1, 2, 3, 4, 5]


def test_generate_deterministic():
    model = _make_model()
    prompt = [1, 2, 3, 4, 5]

    torch.manual_seed(123)
    out1 = run_generate(model, prompt, max_new_tokens=15, temperature=1.0, context_length=32)

    torch.manual_seed(123)
    out2 = run_generate(model, prompt, max_new_tokens=15, temperature=1.0, context_length=32)

    assert out1 == out2


def test_generate_tokens_in_vocab():
    model = _make_model()
    prompt = [1, 2, 3]
    torch.manual_seed(42)
    output = run_generate(model, prompt, max_new_tokens=20, temperature=1.0, context_length=32)
    for t in output[len(prompt) :]:
        assert 0 <= t < 64, f"Generated token {t} outside vocab range [0, 64)"


def test_generate_not_constant():
    model = _make_model()
    prompt = [1, 2, 3]
    torch.manual_seed(42)
    output = run_generate(model, prompt, max_new_tokens=30, temperature=5.0, context_length=32)
    generated = output[len(prompt) :]
    assert len(set(generated)) > 1, "All generated tokens are identical"


def test_generate_truncates_context():
    model = _make_model()
    prompt = [1, 2, 3]
    output = run_generate(model, prompt, max_new_tokens=50, temperature=1.0, context_length=32)
    assert len(output) == 53


def test_generate_default_context_length():
    model = _make_model()
    prompt = [1, 2, 3]
    output = run_generate(model, prompt, 5, temperature=1.0)
    assert len(output) == len(prompt) + 5


def test_generate_uses_model_logits():
    """Generate must use model output to select tokens, not random sampling."""

    class ConstantLogitModel(torch.nn.Module):
        """Always predicts token 7 with overwhelming logits."""

        def __init__(self):
            super().__init__()
            self.context_length = 32
            self._dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            B, T = x.shape
            logits = torch.full((B, T, 64), -100.0)
            logits[:, :, 7] = 100.0
            return logits

    model = ConstantLogitModel()
    model.eval()
    prompt = [1, 2, 3]
    output = run_generate(model, prompt, max_new_tokens=10, temperature=1.0, context_length=32)
    assert output[:3] == [1, 2, 3], "Prompt should be preserved"
    generated = output[3:]
    assert all(t == 7 for t in generated), (
        f"Model always predicts token 7 but got {generated} -- "
        "generate() may not be using model logits"
    )


def test_generate_temperature_effect():
    """Different temperatures must produce different outputs."""
    model = _make_model()
    prompt = [1, 2, 3]

    torch.manual_seed(42)
    out_low = run_generate(model, prompt, max_new_tokens=50, temperature=0.01, context_length=32)

    torch.manual_seed(42)
    out_high = run_generate(model, prompt, max_new_tokens=50, temperature=100.0, context_length=32)

    assert out_low != out_high, (
        "Outputs identical at temperature=0.01 and temperature=100.0 -- "
        "temperature parameter may be ignored"
    )


# ===========================================================================
# Training smoke test (integration)
# ===========================================================================


def test_training_smoke_overfit():
    """Smoke test: model + loss + backward must work together and reduce loss.

    Creates a tiny model, overfits a fixed random batch for 100 steps, and
    verifies loss drops well below the random baseline (~log(64) ≈ 4.16).
    Catches: broken forward, broken loss, broken gradients, shape mismatches.
    Expected runtime: ~10-20 seconds on CPU.
    """
    from transformer_lm.model import TransformerLM
    from transformer_lm.nn_utils import cross_entropy_loss

    torch.manual_seed(42)

    model = TransformerLM(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

    # Fixed batch to memorize
    x = torch.randint(0, 64, (4, 32))
    y = torch.randint(0, 64, (4, 32))

    # Record initial loss
    model.eval()
    with torch.no_grad():
        initial_loss = cross_entropy_loss(model(x), y).item()
    assert initial_loss > 2.0, f"Initial loss {initial_loss:.2f} suspiciously low for random init"

    # Train for 100 steps (high LR + small model = fast overfit)
    model.train()
    for _ in range(100):
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert final_loss < 0.5, (
        f"After 100 overfit steps, loss should drop below 0.5 "
        f"(initial={initial_loss:.2f}, final={final_loss:.2f}). "
        f"Check model, loss function, and gradient flow."
    )


# ----------------------------------------------------------------------
# Additional coverage: get_batch dtype contract
# ----------------------------------------------------------------------


def test_get_batch_returns_long_dtype():
    """Both x and y must have dtype torch.long (integer IDs for embedding
    and cross-entropy). Returning floats or smaller ints will crash
    downstream layers in subtle ways."""
    data = torch.arange(1000, dtype=torch.long)
    x, y = run_get_batch(data, batch_size=4, context_length=16, device="cpu")
    assert x.dtype == torch.long, f"x.dtype must be torch.long, got {x.dtype}"
    assert y.dtype == torch.long, f"y.dtype must be torch.long, got {y.dtype}"

    # Also verify the contract holds when the source data is int32 — the
    # spec says outputs are long regardless of input dtype.
    data_int32 = torch.arange(1000, dtype=torch.int32)
    x32, y32 = run_get_batch(data_int32, batch_size=4, context_length=16, device="cpu")
    assert x32.dtype == torch.long, (
        "x must be torch.long even when input data is int32"
    )
    assert y32.dtype == torch.long, (
        "y must be torch.long even when input data is int32"
    )
