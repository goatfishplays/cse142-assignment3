"""Validate a submission ZIP before uploading to Canvas.

Usage:
    PYTHONPATH=. python3 scripts/validate_submission.py submission.zip
    PYTHONPATH=. python3 scripts/validate_submission.py submission.zip --run-tests
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REQUIRED_FILES = [
    "transformer_lm/model.py",
    "transformer_lm/nn_utils.py",
    "transformer_lm/training_utils.py",
    "transformer_lm/tokenizer.py",
    "transformer_lm/lr_schedule.py",
    "config.yaml",
]


def find_file_in_zip(zf: zipfile.ZipFile, target: str) -> str | None:
    """Find a required file in the ZIP, handling nested directory prefixes."""
    names = zf.namelist()
    # Exact match
    if target in names:
        return target
    # Match with a single leading directory prefix (e.g., transformer_assignment/...)
    for name in names:
        parts = name.split("/", 1)
        if len(parts) == 2 and parts[1] == target:
            return name
    return None


def validate_zip(zip_path: Path) -> tuple[bool, list[str]]:
    """Validate ZIP contents. Returns (ok, errors)."""
    errors: list[str] = []

    if not zip_path.exists():
        return False, [f"File not found: {zip_path}"]

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        return False, [f"Not a valid ZIP file: {zip_path}"]

    with zf:
        for req in REQUIRED_FILES:
            match = find_file_in_zip(zf, req)
            if match is None:
                errors.append(f"Missing: {req}")

    return len(errors) == 0, errors


def validate_config(config_path: Path) -> list[str]:
    """Validate config.yaml schema. Errors prefixed with 'WARN:' are
    soft warnings the caller may downgrade."""
    errors: list[str] = []
    try:
        import yaml
    except ImportError:
        return ["WARN: pyyaml not installed — install it (`pip install pyyaml`) "
                "to enable config.yaml validation"]

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        return [f"config.yaml parse error: {e}"]

    if not isinstance(raw, dict):
        return [f"config.yaml must be a YAML mapping, got {type(raw).__name__}"]

    allowed = {
        "d_model", "n_layers", "n_heads", "d_ff", "dropout",
        "batch_size", "learning_rate", "min_lr", "warmup_steps",
        "weight_decay", "gradient_clip_norm", "beta1", "beta2",
    }
    for key in raw:
        if key not in allowed:
            errors.append(f"config.yaml: unknown key '{key}'")
        elif isinstance(raw[key], (int, float)):
            pass  # valid
        elif isinstance(raw[key], str):
            # Handle scientific notation (e.g., 1e-3) which YAML parses as string
            try:
                float(raw[key])
            except ValueError:
                errors.append(f"config.yaml: '{key}' must be numeric, got '{raw[key]}'")
        else:
            errors.append(f"config.yaml: '{key}' must be numeric, got {type(raw[key]).__name__}")

    # Helper: coerce YAML value to numeric
    def to_num(val, default):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return default
        return default

    # Architecture constraint checks
    d_model = int(to_num(raw.get("d_model", 64), 64))
    n_heads = int(to_num(raw.get("n_heads", 4), 4))
    d_ff = int(to_num(raw.get("d_ff", 256), 256))
    n_layers = int(to_num(raw.get("n_layers", 5), 5))

    if d_model <= 0:
        errors.append(f"d_model must be positive (got {d_model})")
    if n_heads <= 0:
        errors.append(f"n_heads must be positive (got {n_heads})")
    if d_ff <= 0:
        errors.append(f"d_ff must be positive (got {d_ff})")
    if n_layers <= 0:
        errors.append(f"n_layers must be positive (got {n_layers})")

    if n_heads > 0 and d_model > 0 and d_model % n_heads != 0:
        errors.append(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if n_heads > 0 and d_model > 0 and d_model % n_heads == 0 \
            and (d_model // n_heads) % 2 != 0:
        errors.append(f"d_head ({d_model // n_heads}) must be even (RoPE requirement)")

    # Parameter budget check (only if dimensions are positive)
    if d_model > 0 and d_ff > 0 and n_layers > 0:
        params = 512 * d_model + n_layers * (4 * d_model * d_model + 3 * d_model * d_ff)
        if params > 500_000:
            errors.append(f"Estimated {params:,} params exceeds 500,000 limit")

    # Training-hyperparameter range checks
    def _num(key, default):
        val = raw.get(key, default)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return float(default)
        return float(default)

    batch_size = int(to_num(raw.get("batch_size", 32), 32))
    if batch_size <= 0:
        errors.append(f"batch_size must be positive (got {batch_size})")
    learning_rate = _num("learning_rate", 1e-3)
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        errors.append(f"learning_rate must be positive and finite (got {learning_rate})")
    min_lr = _num("min_lr", 1e-5)
    if min_lr < 0 or not math.isfinite(min_lr):
        errors.append(f"min_lr must be nonnegative and finite (got {min_lr})")
    warmup = int(to_num(raw.get("warmup_steps", 200), 200))
    if warmup < 0:
        errors.append(f"warmup_steps must be nonnegative (got {warmup})")
    if warmup >= 5000:
        errors.append(
            f"warmup_steps ({warmup}) must be less than max_steps (5000); "
            "the cosine decay phase would never run"
        )
    weight_decay = _num("weight_decay", 0.1)
    if weight_decay < 0 or not math.isfinite(weight_decay):
        errors.append(f"weight_decay must be nonnegative and finite (got {weight_decay})")
    clip_norm = _num("gradient_clip_norm", 1.0)
    if clip_norm <= 0 or not math.isfinite(clip_norm):
        errors.append(
            f"gradient_clip_norm must be positive and finite (got {clip_norm})"
        )
    for beta_key in ("beta1", "beta2"):
        if beta_key in raw:
            b = _num(beta_key, 0.9)
            if not (0.0 <= b < 1.0):
                errors.append(f"{beta_key} must be in [0, 1) (got {b})")
    dropout = _num("dropout", 0.0)
    if not (0.0 <= dropout < 1.0):
        errors.append(f"dropout must be in [0, 1) (got {dropout})")

    return errors


def validate_imports(work_dir: Path) -> list[str]:
    """Check that student Python files import without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import transformer_lm.model; import transformer_lm.nn_utils; "
         "import transformer_lm.training_utils; import transformer_lm.tokenizer; "
         "import transformer_lm.lr_schedule"],
        cwd=str(work_dir),
        env={**__import__("os").environ, "PYTHONPATH": str(work_dir)},
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        last_line = lines[-1] if lines else "unknown import error (no stderr)"
        return [f"Import error: {last_line}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission ZIP")
    parser.add_argument("zip_path", type=str, help="Path to submission ZIP")
    # Tests run by default so a "PASSED" report actually means the
    # implementation works. Use --no-tests for a quick ZIP-only sanity check
    # when you just want to confirm packaging.
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--run-tests", dest="run_tests",
                            action="store_true", default=True,
                            help="Run pytest after structural validation (default)")
    test_group.add_argument("--no-tests", dest="run_tests",
                            action="store_false",
                            help="Skip pytest; only check the ZIP and config")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    print(f"Validating: {zip_path}")
    all_errors: list[str] = []

    # Step 1: Check ZIP contents
    ok, errors = validate_zip(zip_path)
    all_errors.extend(errors)
    if not ok:
        for e in all_errors:
            print(f"  FAIL: {e}")
        print("\nValidation FAILED.")
        sys.exit(1)
    print("  OK: All 6 required files found")

    # Step 2: Extract to temp dir and validate
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract files into proper structure with size caps (zip-bomb defense).
        # Matches autograder/grade_batch.py caps so local validation catches
        # what the staff grader catches.
        _MAX_MEMBER = 5 * 1024 * 1024
        _MAX_TOTAL = 10 * 1024 * 1024
        total_bytes = 0
        with zipfile.ZipFile(zip_path) as zf:
            for req in REQUIRED_FILES:
                match = find_file_in_zip(zf, req)
                info = zf.getinfo(match)
                if info.file_size > _MAX_MEMBER:
                    print(f"  FAIL: {req} is too large ({info.file_size:,} bytes); cap is {_MAX_MEMBER:,}")
                    sys.exit(1)
                dest = tmp / req
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
                            print(f"  FAIL: {req} exceeded per-file cap during read")
                            sys.exit(1)
                        if total_bytes > _MAX_TOTAL:
                            print(f"  FAIL: submission total exceeds {_MAX_TOTAL:,} bytes (zip-bomb?)")
                            sys.exit(1)
                        dst.write(chunk)

        # Ensure __init__.py exists
        init_py = tmp / "transformer_lm" / "__init__.py"
        if not init_py.exists():
            init_py.touch()

        # Step 3: Validate config.yaml. Errors that begin with "WARN:" are
        # soft warnings; everything else (including unknown keys) is a hard
        # FAIL — staff grading rejects unknown keys to keep submissions
        # reproducible.
        config_errors = validate_config(tmp / "config.yaml")
        all_errors.extend(config_errors)
        if config_errors:
            for e in config_errors:
                if e.startswith("WARN:"):
                    print(f"  WARN: {e[5:].lstrip()}")
                else:
                    print(f"  FAIL: {e}")
        else:
            print("  OK: config.yaml is valid")

        # Step 4: Validate imports
        import_errors = validate_imports(tmp)
        all_errors.extend(import_errors)
        if import_errors:
            for e in import_errors:
                print(f"  FAIL: {e}")
        else:
            print("  OK: All Python files import successfully")

        # Step 5: Optionally run tests
        if args.run_tests and not import_errors:
            print("\nRunning tests...")
            # Copy tests/ into temp dir for testing
            import shutil
            src_tests = Path("tests")
            if src_tests.exists():
                shutil.copytree(src_tests, tmp / "tests")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
                cwd=str(tmp),
                env={**__import__("os").environ, "PYTHONPATH": str(tmp)},
                capture_output=True, text=True, timeout=300,
            )
            print(result.stdout)
            if result.returncode != 0:
                all_errors.append("Some tests failed")

    if all_errors:
        # Distinguish warnings from errors. Unknown config keys are now
        # hard errors (matches the training script's behavior).
        hard_errors = [
            e for e in all_errors
            if not e.startswith("WARN:")
        ]
        if hard_errors:
            print(f"\nValidation FAILED ({len(hard_errors)} error(s)).")
            sys.exit(1)
        else:
            print(f"\nValidation PASSED with {len(all_errors)} warning(s).")
    else:
        print("\nValidation PASSED.")


if __name__ == "__main__":
    main()
