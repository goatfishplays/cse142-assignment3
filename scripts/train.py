"""Training script for the Transformer LM on TinyStories.

Usage:
    python scripts/train.py [--device cpu|cuda|mps] [--max_steps N] ...
    python scripts/train.py --config config.yaml  # load from YAML

Reads hyperparameters from config.yaml (if present), with CLI args taking
precedence.  Students implement their code in transformer_lm/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import numpy as np

# SECURITY: snapshot staff-controlled references BEFORE any student module is
# imported. A student `transformer_lm.*` module could otherwise monkey-patch
# `torch.optim.AdamW`, `torch.nn.utils.clip_grad_norm_`, or
# `torch.nn.Module.named_parameters/state_dict` during its import and
# substitute a different optimizer/clipping/save path at grading time.
from torch.optim import AdamW as _STAFF_ADAMW
from torch.nn.utils import clip_grad_norm_ as _STAFF_CLIP_GRAD_NORM
_STAFF_NAMED_PARAMETERS = torch.nn.Module.named_parameters
_STAFF_NAMED_BUFFERS = torch.nn.Module.named_buffers
_STAFF_STATE_DICT = torch.nn.Module.state_dict
_STAFF_PARAMETERS = torch.nn.Module.parameters
_STAFF_TORCH_SAVE = torch.save
_STAFF_TORCH_LOAD = torch.load
_STAFF_AUTOCAST = torch.autocast
_STAFF_NO_GRAD = torch.no_grad
AdamW = _STAFF_ADAMW  # re-export under the name the training code uses

from transformer_lm.model import TransformerLM
from transformer_lm.nn_utils import cross_entropy_loss
from transformer_lm.lr_schedule import lr_schedule
from transformer_lm.training_utils import generate, get_batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""

    # --- model ---
    vocab_size: int = 512       # FIXED — must be 512 (matches tokenizer)
    context_length: int = 256   # FIXED — must be 256 (matches evaluation)
    d_model: int = 64
    n_layers: int = 5
    n_heads: int = 4            # Must divide d_model; d_model/n_heads must be even
    d_ff: int = 256

    # --- training ---
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    dropout: float = 0.0

    # --- evaluation ---
    eval_interval: int = 100

    # --- I/O ---
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    device: str = "auto"
    seed: int = 42
    resume: bool = False


# ---------------------------------------------------------------------------
# Config YAML loading
# ---------------------------------------------------------------------------

# Keys in config.yaml that map to TrainConfig fields
YAML_CONFIG_KEYS = {
    "d_model", "n_layers", "n_heads", "d_ff", "dropout",
    "batch_size", "learning_rate", "min_lr", "warmup_steps",
    "weight_decay", "gradient_clip_norm", "beta1", "beta2",
}


def load_yaml_config(config_path: str | Path) -> dict:
    """Load and validate a config.yaml file.

    Returns a dict of validated key-value pairs suitable for overriding
    TrainConfig fields.  Warns on unknown keys and rejects bad types.
    """
    try:
        import yaml
    except ImportError:
        print("WARNING: pyyaml not installed, skipping config.yaml")
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # A malformed config.yaml must be a hard failure for grading. The
        # autograder runs validate_submission.py first which already rejects
        # this case, but a student running train.py directly used to see a
        # raw PyYAML traceback — now they get a clear actionable message.
        raise SystemExit(f"FATAL: {path} is not valid YAML: {e}") from None

    if not isinstance(raw, dict):
        raise SystemExit(
            f"FATAL: {path} must be a YAML mapping (key: value pairs); "
            f"got {type(raw).__name__}."
        )

    # Fields that must be integers (float values would crash downstream code)
    _INT_FIELDS = {"d_model", "n_layers", "n_heads", "d_ff", "batch_size", "warmup_steps"}

    validated: dict = {}
    for key, value in raw.items():
        if key not in YAML_CONFIG_KEYS:
            raise SystemExit(
                f"FATAL: config.yaml contains unknown key '{key}'. "
                f"Allowed keys: {sorted(YAML_CONFIG_KEYS)}. "
                "Unknown keys are rejected to avoid silent typos and "
                "to keep submissions reproducible."
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            validated[key] = value
        elif isinstance(value, str):
            # Handle scientific notation (e.g., 1e-3) which YAML parses as string
            try:
                validated[key] = float(value)
            except ValueError:
                raise SystemExit(
                    f"FATAL: config key '{key}' must be numeric, got '{value}'. "
                    "Submissions with non-numeric config values are rejected "
                    "instead of silently falling back to defaults."
                ) from None
        else:
            raise SystemExit(
                f"FATAL: config key '{key}' must be numeric, got "
                f"{type(value).__name__}."
            )
        # Coerce integer fields (e.g., d_model: 6.4e1 → 64)
        if key in _INT_FIELDS:
            validated[key] = int(validated[key])

    return validated


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tokens(path: str) -> torch.Tensor:
    """Load a .bin file of uint16 tokens into a 1-D long tensor."""
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_data: torch.Tensor,
    config: TrainConfig,
) -> float:
    """Compute validation loss over deterministic, non-overlapping windows.

    Splits val_data into consecutive chunks of context_length and evaluates
    each one in batches for efficiency. This is deterministic — same model
    always gets the same score.

    Args:
        model: The transformer LM.
        val_data: 1-D tensor of validation token IDs.
        config: Training configuration.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    ctx = config.context_length
    # Collect all windows
    xs, ys = [], []
    for start in range(0, len(val_data) - ctx, ctx):
        xs.append(val_data[start : start + ctx])
        ys.append(val_data[start + 1 : start + ctx + 1])
    if not xs:
        model.train()
        return 0.0
    xs = torch.stack(xs).to(config.device)  # (N, ctx)
    ys = torch.stack(ys).to(config.device)  # (N, ctx)
    # Process in batches. Use bf16 autocast on CUDA so val_loss is computed
    # with the same numerics as training.
    eval_batch_size = 64
    use_bf16 = (torch.device(config.device).type == "cuda")
    total_loss = 0.0
    n_chunks = 0
    for i in range(0, len(xs), eval_batch_size):
        xb = xs[i:i+eval_batch_size]
        yb = ys[i:i+eval_batch_size]
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = cross_entropy_loss(logits, yb)
        else:
            logits = model(xb)
            loss = cross_entropy_loss(logits, yb)
        total_loss += loss.item() * len(xb)
        n_chunks += len(xb)
    model.train()
    return total_loss / max(n_chunks, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _save_metrics(config: TrainConfig, metrics: dict, best_val_loss: float) -> None:
    """Save metrics.json with config — called on every best checkpoint and at end."""
    out = dict(metrics)
    out["best_val_loss"] = best_val_loss
    out["config"] = asdict(config)
    with open(os.path.join(config.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


def _generate_sample(model, train_data, data_dir, device):
    """Generate and print a short text sample from the model."""
    model.eval()
    prompt_ids = train_data[:16].tolist()
    sample_ids = generate(model, prompt_ids, max_new_tokens=100, temperature=0.8)
    ref_path = os.path.join(data_dir, "reference_tokenizer.pkl")
    if os.path.exists(ref_path):
        import pickle
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        vocab = ref["vocab"]
        decoded = b"".join(vocab[i] for i in sample_ids if i in vocab)
        print(f"  >> {decoded.decode('utf-8', errors='replace')[:200]}")
    else:
        print(f"  >> Token IDs: {sample_ids[:20]}...")
    model.train()


SAMPLE_MILESTONES = {0, 1000, 2500}  # also at final step


def train(config: TrainConfig) -> None:
    """Run the full training loop."""
    # Seed every RNG that can influence training. Torch covers itself and
    # (for newer versions) CUDA/MPS, but Python's `random` and NumPy must be
    # seeded explicitly — student code may legitimately use either during
    # initialization or sampling. Without this, --seed 42 is incomplete.
    import random as _random
    _random.seed(config.seed)
    try:
        import numpy as _np
        _np.random.seed(config.seed)
    except ImportError:
        pass
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(config.seed)
        except Exception:
            pass

    # Enable TF32 for faster fp32 matmul on Ampere+ GPUs (used as fallback
    # when bf16 autocast is unavailable, e.g., CPU/MPS).
    torch.set_float32_matmul_precision("high")
    # Mixed-precision training: bf16 autocast on CUDA gives a ~1.2x speedup
    # at our scale (small matmuls are mostly bandwidth-bound, not compute-
    # bound). Model parameters and AdamW state stay fp32; only forward-pass
    # activations are bf16 inside the autocast region. CPU/MPS skip autocast.
    # Normalize the device string so `cuda:0` etc. still hit the bf16 path.
    _device_type = torch.device(config.device).type
    use_bf16 = (_device_type == "cuda")

    # --- Load data ---
    train_bin = os.path.join(config.data_dir, "train.bin")
    val_bin = os.path.join(config.data_dir, "val.bin")
    for path in (train_bin, val_bin):
        if not os.path.exists(path):
            raise SystemExit(
                f"FATAL: {path} not found.\n"
                f"Run `PYTHONPATH=. python3 scripts/prepare_data.py` first to "
                f"download TinyStories and tokenize it."
            )
    train_data = load_tokens(train_bin)
    val_data = load_tokens(val_bin)
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    # --- Initialize model ---
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(config.device)

    # SECURITY: count parameters via the staff snapshot of nn.Module's base
    # methods, taken at module import time before any student module ran.
    # This prevents a student `transformer_lm.*` from monkey-patching
    # `torch.nn.Module.named_parameters/named_buffers` during import.
    #
    # Buffers count toward the 500K cap UNLESS they live inside a
    # RotaryPositionEmbedding module (isinstance check — matches the
    # evaluator's logic at autograder/evaluate_submission.py:113). This
    # blocks the bypass where a student registers a learnable scale called
    # `inv_freq` on some other module.
    from transformer_lm.model import RotaryPositionEmbedding
    _BASE_NAMED_MODULES = torch.nn.Module.named_modules
    rope_module_ids: set[int] = set()
    for _, module in _BASE_NAMED_MODULES(model):
        if isinstance(module, RotaryPositionEmbedding):
            rope_module_ids.add(id(module))

    seen: set[int] = set()
    n_params = 0
    for _, p in _STAFF_NAMED_PARAMETERS(model):
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            n_params += p.numel()
    for _, module in _BASE_NAMED_MODULES(model):
        if id(module) in rope_module_ids:
            continue
        for _, b in _STAFF_NAMED_BUFFERS(module, recurse=False):
            if b.data_ptr() not in seen:
                seen.add(b.data_ptr())
                n_params += b.numel()
    PARAM_CAP = 500_000
    print(f"Model parameters: {n_params:,}")
    if n_params > PARAM_CAP:
        raise SystemExit(
            f"FATAL: live model has {n_params:,} parameters/non-RoPE buffers, "
            f"which exceeds the {PARAM_CAP:,} cap. Reduce d_model/n_layers/d_ff."
        )

    # SECURITY: check initial loss is near the random baseline log(vocab_size).
    # A student loading pretrained weights in __init__ would start with loss
    # far below ~log(512)≈6.24. Use a STAFF-OWNED batch sampler and a
    # STAFF-OWNED cross-entropy here — calling student `get_batch` or
    # `cross_entropy_loss` would let those functions misreport the loss to
    # bypass the check.
    import math as _math
    def _staff_initial_loss_check() -> float:
        # Deterministic staff batch: fixed slices from the front of train_data.
        bsz_chk = 32
        ctx = config.context_length
        starts = torch.arange(bsz_chk) * ctx  # non-overlapping windows
        if starts[-1].item() + ctx + 1 > len(train_data):
            starts = torch.arange(min(bsz_chk, max(1, (len(train_data) - 1) // ctx))) * ctx
        x_chk = torch.stack([
            torch.from_numpy(np.array(train_data[int(s): int(s) + ctx], copy=True)).long()
            for s in starts
        ]).to(config.device)
        y_chk = torch.stack([
            torch.from_numpy(np.array(train_data[int(s) + 1: int(s) + ctx + 1], copy=True)).long()
            for s in starts
        ]).to(config.device)
        # Staff cross-entropy via torch.nn.functional, computed inline.
        with torch.no_grad():
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_chk).float()
            else:
                logits = model(x_chk)
            # Manual log-sum-exp; never trust student F-shortcuts here either.
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
            flat_targets = y_chk.reshape(-1)
            max_v = flat_logits.max(dim=-1, keepdim=True).values
            lse = (flat_logits - max_v).exp().sum(dim=-1).log() + max_v.squeeze(-1)
            picked = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
            return float((lse - picked).mean().item())
    model.eval()
    _init_loss = _staff_initial_loss_check()
    _RANDOM_BASELINE = _math.log(config.vocab_size)
    _MIN_INIT_LOSS = _RANDOM_BASELINE * 0.6  # ~3.74 for vocab=512
    print(f"Initial loss (staff check): {_init_loss:.4f} (random baseline ~{_RANDOM_BASELINE:.2f})")
    if _init_loss < _MIN_INIT_LOSS:
        raise SystemExit(
            f"FATAL: initial loss {_init_loss:.4f} is below the random-init "
            f"floor ~{_MIN_INIT_LOSS:.4f}. This suggests pretrained weights "
            "are being loaded inside the model constructor, which is "
            "prohibited (assignment.pdf §What You May Not Use)."
        )
    model.train()

    # --- Initialize optimizer ---
    optimizer = AdamW(
        _STAFF_PARAMETERS(model),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # --- Training ---
    os.makedirs(config.out_dir, exist_ok=True)
    metrics: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "time": [],
    }
    best_val_loss = float("inf")
    start_step = 0

    # --- Resume from checkpoint ---
    if config.resume:
        ckpt_path = os.path.join(config.out_dir, "resume_checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = _STAFF_TORCH_LOAD(ckpt_path, map_location=config.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            metrics = ckpt.get("metrics", metrics)
            print(f"Resumed from step {start_step} (best val loss: {best_val_loss:.4f})")
        else:
            print("No resume checkpoint found, starting from scratch.")

    # Write metrics.json upfront with the live model config so the evaluator
    # can recover dims even if training crashes before the first eval-loss
    # improvement. Without this, a NaN-on-first-eval failure falls back to
    # default dims, mis-loads the checkpoint, and the autograder reports
    # "Eval error" instead of the real student bug.
    _save_metrics(config, metrics, best_val_loss=float("inf"))

    t0 = time.time()

    model.train()
    for step in range(start_step, config.max_steps):
        # --- Learning rate schedule (from transformer_lm/lr_schedule.py) ---
        lr = lr_schedule(
            step, config.max_steps, config.learning_rate,
            config.min_lr, config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Forward pass ---
        x, y = get_batch(
            train_data, config.batch_size, config.context_length, config.device
        )
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
        else:
            logits = model(x)
            loss = cross_entropy_loss(logits, y)

        # Fail fast on non-finite loss. Otherwise NaN/Inf silently propagates
        # for the rest of training and only surfaces later in sample
        # generation as an opaque multinomial error.
        if not torch.isfinite(loss):
            raise SystemExit(
                f"FATAL: non-finite train loss ({loss.item()}) at step {step}. "
                "Check cross_entropy_loss, attention scaling, RMSNorm epsilon, "
                "or learning rate. Inspect logits.min/max() for clues."
            )

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        if config.gradient_clipping:
            _STAFF_CLIP_GRAD_NORM(_STAFF_PARAMETERS(model), config.gradient_clip_norm)
        optimizer.step()

        # --- Progress dots between eval intervals ---
        is_eval_step = (step % config.eval_interval == 0 or step == config.max_steps - 1)
        if not is_eval_step and step % 10 == 0:
            print(".", end="", flush=True)

        # --- Time estimate after step 100 ---
        if step == 100:
            elapsed_100 = time.time() - t0
            remaining = config.max_steps - step
            est_total = elapsed_100 / 100 * config.max_steps
            est_remain = elapsed_100 / 100 * remaining
            print(
                f"\n  Time estimate: {est_total:.0f}s total, "
                f"~{est_remain:.0f}s remaining"
            )

        # --- Logging and evaluation ---
        if is_eval_step:
            val_loss = evaluate(model, val_data, config)
            elapsed = time.time() - t0
            print(
                f"\nstep {step:5d} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"time {elapsed:.1f}s"
            )
            metrics["step"].append(step)
            metrics["train_loss"].append(loss.item())
            metrics["val_loss"].append(val_loss)
            metrics["lr"].append(lr)
            metrics["time"].append(elapsed)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # SECURITY: save via the base-class state_dict so a student
                # `def state_dict(self): return staff_pretrained_weights` cannot
                # smuggle a different checkpoint into grading.
                _STAFF_TORCH_SAVE(
                    _STAFF_STATE_DICT(model),
                    os.path.join(config.out_dir, "best_model.pt"),
                )
                _save_metrics(config, metrics, best_val_loss)

            # Save resume checkpoint (model + optimizer state) — same hardening.
            _STAFF_TORCH_SAVE(
                {
                    "model": _STAFF_STATE_DICT(model),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "metrics": metrics,
                },
                os.path.join(config.out_dir, "resume_checkpoint.pt"),
            )

            # Generate sample at milestones
            if step in SAMPLE_MILESTONES or step == config.max_steps - 1:
                print(f"--- Sample at step {step} ---")
                _generate_sample(model, train_data, config.data_dir, config.device)

    # --- Save final checkpoint and metrics ---
    _STAFF_TORCH_SAVE(_STAFF_STATE_DICT(model), os.path.join(config.out_dir, "final_model.pt"))
    _save_metrics(config, metrics, best_val_loss)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # --- Generate a final sample from best model ---
    best_path = os.path.join(config.out_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(
            _STAFF_TORCH_LOAD(best_path, map_location=config.device, weights_only=True)
        )
    print("\n--- Final sample (best model) ---")
    _generate_sample(model, train_data, config.data_dir, config.device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (detect GPU), 'cpu', 'cuda', or 'mps'",
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_clipping", action=argparse.BooleanOptionalAction,
                        default=None, help="Enable gradient clipping")
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta2")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--d_model", type=int, default=None, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=None,
                        help="Number of attention heads (must divide d_model)")
    parser.add_argument("--d_ff", type=int, default=None, help="FFN hidden dimension")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Build config: TrainConfig defaults → config.yaml → CLI args
    config = TrainConfig()

    # Layer 2: config.yaml overrides
    yaml_overrides = load_yaml_config(args.config)
    if yaml_overrides:
        print(f"Loaded config from {args.config}: {yaml_overrides}")
    for key, val in yaml_overrides.items():
        setattr(config, key, val)

    # Layer 3: CLI args override (only non-None values)
    cli_fields = {
        "d_model", "n_layers", "n_heads", "d_ff", "max_steps", "batch_size",
        "learning_rate", "min_lr", "warmup_steps", "weight_decay",
        "gradient_clipping", "gradient_clip_norm", "eval_interval", "seed",
        "data_dir", "out_dir", "beta1", "beta2", "dropout",
    }
    for key in cli_fields:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    # Resume is a flag — always apply if set
    if args.resume:
        config.resume = True

    # Auto-detect best available device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    config.device = device
    print(f"Using device: {device}")

    train(config)


if __name__ == "__main__":
    main()
