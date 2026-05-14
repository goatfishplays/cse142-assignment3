# Building a Transformer LM from Scratch

A from-scratch decoder-only transformer language model trained on
[TinyStories](https://arxiv.org/abs/2305.07759). You will implement the
tokenizer, the model architecture, the loss, and the training utilities,
then tune your design to minimize loss under a strict parameter budget.

> **Read [`assignment.pdf`](assignment.pdf) first.** It is the
> authoritative spec — overview, hard constraints, what you may and may
> not use, allowed config keys, grading rubric, and submission format.
> This README is just the quickstart with commands and project layout.
>
> The PDF is long (~20 pages), but don't let the length put you off.
> Most of it is guidance meant to make the work go smoother:
> implementation hints, common bugs, suggested coding order, debugging
> tips, exploration ideas. The hard requirements — constraints, grading,
> what to submit — are only a few pages. Read those carefully; skim the
> rest and come back to it when something breaks.

## Schedule

- **Released:** 5/14, after 3 PM.
- **Due:** 6/5 (last day of the quarter), submitted to Canvas.

The assignment is paced to the three lectures that cover its material.
You can start the prerequisite reading and Part 1 right away; the rest
unlocks as the lectures arrive.

| Date | Lecture | What you can work on |
|---|---|---|
| 5/14 | Language models | PyTorch prerequisite reading (§2 of `assignment.pdf`) and Part 1 (BPE tokenizer). Part 1 doesn't depend on the model or the later lectures. |
| 5/19 | Transformers | Most of the assignment opens up: Part 2 (model) and Part 3 (primitives and loss). |
| 5/21 | Training | The rest: Part 4 (`get_batch`, `generate`), the LR schedule, and the Part 5 training-quality gate. |

You have about three weeks. Try to finish Parts 1–4 with at least a
week to spare so you can tune hyperparameters and architecture for the
bonus tiers.

## Setup

Python 3.10+ is required. If you go with Option A, install `uv` first
— see the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Option A — uv (recommended; creates .venv from pyproject.toml)
uv sync
source .venv/bin/activate

# Option B — pip + venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

After activating the venv, `python` and `pytest` resolve to the venv's
interpreter. (If you prefer not to activate, prefix each command with
`uv run`, e.g. `uv run python scripts/train.py`.)

Then download and tokenize the dataset (once, ~1 min):

```bash
PYTHONPATH=. python scripts/prepare_data.py
```

This downloads a fixed TinyStories subset to `data/` and creates
`train.bin` / `val.bin` using the staff reference tokenizer. Training
always uses this tokenized data — your Part 1 tokenizer is graded
separately by unit tests.

## First steps

1. Read `assignment.pdf` start to finish (≈15 min).
2. Work through §2 (*Required PyTorch Background*) — four official-doc
   links you should be comfortable with before touching Part 2. Skip
   only if you already know `nn.Module`, broadcasting, and autograd.
3. Skim §10.1 (*Suggested Implementation Order*) — this is the fastest
   path to passing tests. Start with `nn_utils.py` (`softmax`, `silu`),
   then build up `model.py` bottom-first.
4. Run the matching tests after each stub you fill in (e.g.
   `PYTHONPATH=. python3 -m pytest tests/test_nn_utils.py::test_softmax_correctness -v`
   after softmax, `tests/test_model.py::TestLinear` after `Linear`).
   §10.2 lists all the targeted invocations.

All commands below assume the repo root with `PYTHONPATH=.` so Python can
find the `transformer_lm` package. After activating the venv, `python`
and `python3` both resolve to the venv interpreter.

## Workflow

```bash
# Unit tests — run the whole suite, or one part at a time
PYTHONPATH=. python3 -m pytest tests/ -x -q
PYTHONPATH=. python3 -m pytest tests/test_tokenizer.py -x -q   # Part 1
PYTHONPATH=. python3 -m pytest tests/test_model.py     -x -q   # Part 2
PYTHONPATH=. python3 -m pytest tests/test_nn_utils.py  -x -q   # Part 3
PYTHONPATH=. python3 -m pytest tests/test_training.py  -x -q   # Part 4

# Sanity check (after Parts 2–3) — overfits a tiny batch in ~10s
PYTHONPATH=. python3 scripts/sanity_check.py
```

### Verifying your work as you go

The tests are the source of truth for Parts 1–4 implementation
correctness. Passing them is necessary for base credit, but the full
95% also requires a valid submission ZIP and a staff retrain that
clears the sanity gate (`val_loss < 3.0`). The suite is organized so
you can target a specific component you just finished:

```bash
# Run a single test class for one model component
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestLinear     -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestEmbedding  -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestRMSNorm    -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestScaledDotProductAttention   -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestCausalMultiHeadSelfAttention -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestFeedForward    -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestTransformerBlock -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestTransformerLM    -v

# Run a single test function (Parts 1, 3, 4 use function-level tests)
PYTHONPATH=. python3 -m pytest tests/test_nn_utils.py::test_softmax_correctness    -v
PYTHONPATH=. python3 -m pytest tests/test_training.py::test_get_batch_correctness  -v

# Or use -k to match by name keyword
PYTHONPATH=. python3 -m pytest tests/ -k "softmax or silu" -v
PYTHONPATH=. python3 -m pytest tests/ -k "tokenizer"       -v
```

Tip: drop the `-x` to see all failures at once, or add `-vv` to print
full assertion diffs. Tests in `TestAntiCheat` (in `test_model.py`)
verify you haven't called banned shortcut APIs — keep them green.

### Train and validate

```bash
# Train using config.yaml (auto-detects CUDA > MPS > CPU)
PYTHONPATH=. python3 scripts/train.py

# Quick architecture screen with CLI overrides
PYTHONPATH=. python3 scripts/train.py --max_steps 500 \
    --d_model 40 --n_layers 4 --n_heads 5 --d_ff 160

# Recommended pre-submission gate. Seven steps in order, stops at the
# first failure: file structure → config → imports → pytest → sanity
# overfit → 50-step training preview on real data → state-dict
# round-trip. Full description in §14.2 of assignment.pdf.
PYTHONPATH=. python3 scripts/final_check.py submission.zip

# Quick alternative: just file structure + config + imports.
PYTHONPATH=. python3 scripts/validate_submission.py submission.zip
```

Train on CUDA when you can — bonus tiers were calibrated under bf16 on
CUDA, so other devices may give slightly different numbers
(see §8 of `assignment.pdf`).

If you don't have a local CUDA GPU, Google Colab works well. The free
student plan is no longer offered, but the free tier with a single T4
GPU is more than enough for this assignment. One catch: Colab's local
storage is ephemeral and gets wiped when the runtime disconnects, so
mount Google Drive and save your repo, checkpoints, and tokenized data
there — otherwise you'll lose your work between sessions.

## Project structure

A quickstart subset — the full tree is in §15 of `assignment.pdf`.

```
transformer_lm/          # Your code goes here (5 .py files you edit)
  tokenizer.py           # Part 1: BPE tokenizer
  model.py               # Part 2: transformer model components
  nn_utils.py            # Parts 2–3: softmax, SiLU, cross-entropy
  training_utils.py      # Part 4: get_batch, generate
  lr_schedule.py         # LR schedule (customizable)
config.yaml              # Hyperparameter config (6th file you edit)
assignment.pdf           # Full specification — read this first
tests/                   # Test suite (do not modify)
scripts/                 # prepare_data, train, sanity_check,
                         # validate_submission, final_check
```


## Submission

Submit a single ZIP to Canvas containing exactly these 6 files:

- `transformer_lm/model.py`
- `transformer_lm/nn_utils.py`
- `transformer_lm/training_utils.py`
- `transformer_lm/tokenizer.py`
- `transformer_lm/lr_schedule.py`
- `config.yaml`

```bash
zip -r submission.zip \
    transformer_lm/model.py transformer_lm/nn_utils.py \
    transformer_lm/training_utils.py transformer_lm/tokenizer.py \
    transformer_lm/lr_schedule.py config.yaml

# Recommended pre-submission gate (runs the seven checks listed above)
PYTHONPATH=. python3 scripts/final_check.py submission.zip
```

You also upload a **1–2 page experiment report** to Canvas separately.
Two sections: what you tried and why you chose your final config, and
which LLMs or coding assistants you used and on which parts (required
even if you used none — see §11 of `assignment.pdf`). Full submission
requirements in §14.
