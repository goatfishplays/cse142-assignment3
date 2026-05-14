"""Sanity check: overfit a tiny slice of data.

This script verifies that the model can memorize a small number of tokens.
If this fails, there is likely a bug in the model or loss function.

Expected: train loss < 0.5 within 100 steps (~5-10 seconds on CPU).
"""

from __future__ import annotations

import torch

from transformer_lm.model import TransformerLM
from transformer_lm.nn_utils import cross_entropy_loss
from torch.optim import AdamW


def sanity_check() -> None:
    torch.manual_seed(42)
    device = "cpu"

    # Tiny config — matches tests/test_training.py::test_training_smoke_overfit
    # so a passing sanity check implies the smoke test will also pass.
    vocab_size = 64
    context_length = 32
    d_model = 32
    n_layers = 1
    n_heads = 2
    d_ff = 64

    model = TransformerLM(
        vocab_size, context_length, d_model, n_layers, n_heads, d_ff
    ).to(device)

    # Higher LR than train.py default — small model overfits quickly at 3e-2.
    optimizer = AdamW(model.parameters(), lr=3e-2)

    # Fixed tiny batch (memorize this)
    x = torch.randint(0, vocab_size, (2, context_length), device=device)
    y = torch.randint(0, vocab_size, (2, context_length), device=device)

    baseline = float(torch.log(torch.tensor(float(vocab_size))))
    print("Sanity check: overfitting a tiny batch...")
    print(f"Random baseline loss: {baseline:.4f}")

    model.train()
    for step in range(100):
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  step {step:3d} | loss {loss.item():.4f}")

    final_loss = loss.item()
    THRESHOLD = 0.5  # well below random baseline (~4.16) but lenient enough
                     # for slightly slower-converging valid implementations
    if final_loss < THRESHOLD:
        print(f"\nPASSED: Final loss = {final_loss:.4f} < {THRESHOLD}")
    else:
        print(f"\nFAILED: Final loss = {final_loss:.4f} >= {THRESHOLD}")
        print("If the loss is decreasing but not low enough, your model may be")
        print("correct but slow to converge. Run pytest before debugging.")
        print("If the loss is flat or random, check your forward, loss, or grad flow.")
        raise SystemExit(1)


if __name__ == "__main__":
    sanity_check()
