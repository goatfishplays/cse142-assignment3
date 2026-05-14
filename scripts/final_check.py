"""Final pre-submission check.

Orchestrates every validation step in one command and stops at the first
failure. Run this after `pytest` and before zipping/uploading to Canvas.

Steps:
  1. File structure         — all six required files present
  2. config.yaml schema     — keys, types, arch constraints, param budget
  3. Python imports         — no syntax errors in the five modules
  4. Unit tests             — full pytest suite (`tests/`)
  5. Sanity overfit         — `scripts/sanity_check.py`
  6. Training preview       — 50 real-data steps; verify val loss decreases
  7. State-dict integrity   — strict round-trip + weight tying preserved

Usage:
    PYTHONPATH=. python3 scripts/final_check.py
    PYTHONPATH=. python3 scripts/final_check.py submission.zip
    PYTHONPATH=. python3 scripts/final_check.py --skip-training  # ~30s faster
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Reuse logic from validate_submission to avoid two copies of the schema rules.
sys.path.insert(0, str(Path(__file__).parent))
from validate_submission import (  # noqa: E402
    REQUIRED_FILES,
    find_file_in_zip,
    validate_config,
    validate_imports,
    validate_zip,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOTAL_STEPS = 7


def _header(step: int, name: str) -> None:
    print(f"\n[{step}/{TOTAL_STEPS}] {name}", flush=True)


def _result(ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  -> {mark}{': ' + detail if detail else ''}", flush=True)


def _fail(step: int, msg: str = "") -> None:
    if msg:
        print(f"  {msg}")
    print(f"\nFAILED at step {step}. Fix the issue above and re-run.")
    sys.exit(1)


# --- Step 1: Structure --------------------------------------------------

def check_structure(zip_path: Path | None) -> Path:
    """Return a workdir containing the six submission files.

    For ZIP mode, extracts into a tempdir and copies tests/ from REPO_ROOT.
    For local mode, returns REPO_ROOT directly.
    """
    if zip_path is not None:
        ok, errors = validate_zip(zip_path)
        if not ok:
            for e in errors:
                print(f"  - {e}")
            _fail(1)
        workdir = Path(tempfile.mkdtemp(prefix="final_check_"))
        try:
            # Mirror validate_submission.py: cap per-file and total uncompressed
            # size so a malicious / oversized ZIP can't fill the disk.
            _MAX_MEMBER = 5 * 1024 * 1024
            _MAX_TOTAL = 10 * 1024 * 1024
            total_bytes = 0
            with zipfile.ZipFile(zip_path) as zf:
                for req in REQUIRED_FILES:
                    match = find_file_in_zip(zf, req)
                    info = zf.getinfo(match)
                    if info.file_size > _MAX_MEMBER:
                        _fail(1, f"{req} is too large ({info.file_size:,} bytes); "
                                 f"cap is {_MAX_MEMBER:,}")
                    dest = workdir / req
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    written = 0
                    with zf.open(match) as src, open(dest, "wb") as dst:
                        while True:
                            chunk = src.read(64 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            total_bytes += len(chunk)
                            if written > _MAX_MEMBER:
                                _fail(1, f"{req} exceeded per-file cap during read")
                            if total_bytes > _MAX_TOTAL:
                                _fail(1, f"submission total exceeds "
                                         f"{_MAX_TOTAL:,} bytes (zip-bomb?)")
                            dst.write(chunk)
            (workdir / "transformer_lm" / "__init__.py").touch(exist_ok=True)
            shutil.copytree(REPO_ROOT / "tests", workdir / "tests")
        except BaseException:
            # On any failure (including SystemExit from _fail), clean up the
            # tempdir we just created so it doesn't leak.
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        _result(True, f"all 6 required files in {zip_path.name}")
        return workdir

    # Local mode
    missing = [r for r in REQUIRED_FILES if not (REPO_ROOT / r).exists()]
    if missing:
        for m in missing:
            print(f"  - Missing: {m}")
        _fail(1)
    _result(True, "all 6 required files present in working directory")
    return REPO_ROOT


# --- Step 2: Config -----------------------------------------------------

def check_config(workdir: Path) -> None:
    errors = validate_config(workdir / "config.yaml")
    hard = [e for e in errors if not e.startswith("WARN:")]
    if hard:
        for e in hard:
            print(f"  - {e}")
        _fail(2)
    for e in errors:
        if e.startswith("WARN:"):
            print(f"  warn: {e[5:].lstrip()}")
    _result(True, "config.yaml is valid")


# --- Step 3: Imports ----------------------------------------------------

def check_imports(workdir: Path) -> None:
    errors = validate_imports(workdir)
    if errors:
        for e in errors:
            print(f"  - {e}")
        _fail(3)
    _result(True, "all 5 modules import cleanly")


# --- Step 4: Unit tests -------------------------------------------------

def check_unit_tests(workdir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=line"],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(workdir)},
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        # Include stderr — pytest puts collection errors and Python tracebacks
        # there, so dropping it can produce an almost blank failure report.
        combined = (result.stdout + "\n" + result.stderr).strip()
        tail = "\n  ".join(combined.splitlines()[-12:])
        print(f"  {tail}")
        _fail(4)
    summary = (result.stdout.strip().splitlines() or ["pytest passed"])[-1]
    _result(True, summary)


# --- Step 5: Sanity overfit ---------------------------------------------

def check_sanity(workdir: Path) -> None:
    sanity_script = REPO_ROOT / "scripts" / "sanity_check.py"
    result = subprocess.run(
        [sys.executable, str(sanity_script)],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(workdir)},
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        combined = (result.stdout + "\n" + result.stderr).strip()
        tail = "\n  ".join(combined.splitlines()[-8:])
        print(f"  {tail}")
        _fail(5)
    detail = next(
        (line.split("PASSED: ", 1)[1] for line in result.stdout.splitlines()
         if line.startswith("PASSED:")),
        "loss converged",
    )
    _result(True, detail)


# --- Step 6: Training preview -------------------------------------------

def check_training_preview(workdir: Path) -> None:
    train_bin = REPO_ROOT / "data" / "train.bin"
    val_bin = REPO_ROOT / "data" / "val.bin"
    if not train_bin.exists() or not val_bin.exists():
        _fail(6, "data/train.bin or data/val.bin not found. "
                 "Run `PYTHONPATH=. python3 scripts/prepare_data.py` first.")

    train_script = REPO_ROOT / "scripts" / "train.py"
    out_dir = tempfile.mkdtemp(prefix="final_check_train_")
    try:
        t0 = time.time()
        # Override warmup_steps=0 because the preview only runs 50 steps.
        # Honoring a student's 200-1000 warmup would leave lr near zero for
        # the entire preview and produce a flat loss curve unrelated to
        # implementation correctness. The preview is a smoke test of the
        # training pipeline, not a faithful replica of staff retraining.
        result = subprocess.run(
            [sys.executable, str(train_script),
             "--max_steps", "50",
             "--eval_interval", "25",
             "--warmup_steps", "0",
             "--data_dir", str(REPO_ROOT / "data"),
             "--out_dir", out_dir,
             "--device", "cpu"],
            cwd=str(workdir),
            env={**os.environ, "PYTHONPATH": str(workdir)},
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            tail = "\n  ".join(
                (result.stdout + "\n" + result.stderr).strip().splitlines()[-12:]
            )
            print(f"  {tail}")
            _fail(6)
        # Match the train.py eval-line format precisely so we don't pick up
        # `val_loss`, `validation loss`, or any pipe-delimited text from a
        # generated sample. Format is: "step ... | train loss N | val loss N | ..."
        val_losses: list[float] = []
        for line in result.stdout.splitlines():
            if "| val loss " not in line:
                continue
            for part in line.split("|"):
                part = part.strip()
                if part.startswith("val loss "):
                    try:
                        val_losses.append(float(part.split()[2]))
                    except (ValueError, IndexError):
                        pass
        if len(val_losses) < 2:
            _fail(6, "could not parse two val-loss readings from training "
                     "output; the training script may have changed format. "
                     f"stdout tail:\n  " + "\n  ".join(
                         result.stdout.strip().splitlines()[-8:]))
        first, last = val_losses[0], val_losses[-1]
        if last >= first:
            _fail(6,
                  f"val loss did not decrease ({first:.3f} -> {last:.3f}). "
                  "Check optimizer setup, learning rate, or get_batch.")
        _result(True, f"val loss {first:.3f} -> {last:.3f} in {elapsed:.1f}s")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --- Step 7: State-dict integrity ---------------------------------------

_STATEDICT_SCRIPT = '''
import io, sys, torch, yaml

# Snapshot base-class state_dict / load_state_dict BEFORE importing student
# code so a custom override in transformer_lm/* cannot mask incompatibility
# with the staff-canonical strict load. Mirrors the hardening in
# scripts/train.py (the staff training script).
_BASE_STATE_DICT = torch.nn.Module.state_dict
_BASE_LOAD_STATE_DICT = torch.nn.Module.load_state_dict

with open("config.yaml") as f:
    cfg = yaml.safe_load(f) or {}

def _num(key, default):
    v = cfg.get(key, default)
    return float(v) if isinstance(v, str) else v

VOCAB, CTX = 512, 256
d_model = int(_num("d_model", 64))
n_layers = int(_num("n_layers", 5))
n_heads = int(_num("n_heads", 4))
d_ff = int(_num("d_ff", 256))

from transformer_lm.model import TransformerLM

m1 = TransformerLM(VOCAB, CTX, d_model, n_layers, n_heads, d_ff)
# Use `is` (same Python object), not just data_ptr equality. The assignment
# requires that lm_head.weight BE token_emb.weight (same Parameter), which
# is stricter than sharing storage (storage equality permits two distinct
# Parameter objects that point at the same memory and would produce
# duplicate optimizer entries).
if m1.lm_head.weight is not m1.token_emb.weight:
    print("FAIL: lm_head.weight is not the same tensor as token_emb.weight at init"); sys.exit(1)

buf = io.BytesIO()
torch.save(_BASE_STATE_DICT(m1), buf)
buf.seek(0)
sd = torch.load(buf, weights_only=True)

m2 = TransformerLM(VOCAB, CTX, d_model, n_layers, n_heads, d_ff)
try:
    _BASE_LOAD_STATE_DICT(m2, sd, strict=True)
except Exception as e:
    print(f"FAIL: strict state_dict load failed: {e}"); sys.exit(1)

if m2.lm_head.weight is not m2.token_emb.weight:
    print("FAIL: weight tying broken after state_dict round-trip"); sys.exit(1)

print("OK")
'''


def check_statedict(workdir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _STATEDICT_SCRIPT],
        cwd=str(workdir),
        env={**os.environ, "PYTHONPATH": str(workdir)},
        capture_output=True, text=True, timeout=120,
    )
    out = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not out.endswith("OK"):
        last = out.splitlines()[-1] if out else "unknown error"
        _fail(7, last)
    _result(True, "weight tying preserved through round-trip")


# --- Main ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final pre-submission check (orchestrates all validation)"
    )
    parser.add_argument(
        "zip_path", nargs="?", default=None,
        help="Path to submission ZIP (optional; defaults to working directory)",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip the 50-step training preview (~30s faster, less coverage)",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path) if args.zip_path else None
    print("=== Submission Final Check ===")
    print(f"Source: {zip_path if zip_path else 'current working directory'}")

    cleanup = zip_path is not None
    workdir = None
    try:
        _header(1, "File structure")
        workdir = check_structure(zip_path)

        _header(2, "config.yaml schema")
        check_config(workdir)

        _header(3, "Python imports")
        check_imports(workdir)

        _header(4, "Unit tests (pytest)")
        check_unit_tests(workdir)

        _header(5, "Sanity overfit check")
        check_sanity(workdir)

        if args.skip_training:
            print(f"\n[6/{TOTAL_STEPS}] Training preview (50 steps)")
            print("  -> SKIPPED (--skip-training)")
        else:
            _header(6, "Training preview (50 steps)")
            check_training_preview(workdir)

        _header(7, "State-dict integrity")
        check_statedict(workdir)

        print("\n" + "=" * 48)
        print("ALL CHECKS PASSED. Ready to submit.")
        print("=" * 48)
    finally:
        if cleanup and workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
