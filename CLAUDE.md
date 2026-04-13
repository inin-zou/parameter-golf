# Parameter Golf — Nemotron-H Inspired Mamba-3 Hybrid

## Project Overview

OpenAI Parameter Golf competition: train the best LM that fits in 16MB, under 10 min on 8xH100.
Deadline: April 30, 2026. Metric: val_bpb (bits per byte) — lower is better.

- Fork: https://github.com/inin-zou/parameter-golf
- Upstream: https://github.com/openai/parameter-golf
- Current SOTA: 1.0810 bpb (pure Transformer, PR #1493)
- Baseline: 1.2244 bpb

## Our Approach

Nemotron-H inspired hybrid: mostly Mamba-3 SSM layers + few Attention layers.
Based on PR #1355 (best SSM submission, 1.1526 bpb).

### Key Files

- `train_nemotron_hybrid.py` — our main training script (forked from PR #1355)
- `train_gpt_hybrid.py` — earlier Mamba-2 experiment (deprecated, kept for reference)
- `train_gpt.py` — competition baseline (pure Transformer)
- `reference_pr1355.py` — PR #1355 original script for reference
- `modal_train.py` — Modal cloud GPU deployment (H100)
- `.claude/docs/competition-strategy.md` — full strategy document
- `.claude/docs/submission-guide.md` — how to submit to the competition
- `.claude/docs/compute-grant-application.md` — OpenAI compute grant application text

### Current Results (2026-04-13)

| Config | Steps | val_bpb | ms/step | GPU |
|--------|-------|---------|---------|-----|
| Baseline (pure Transformer) | 200 | 2.02 | 333 | 1xH100 |
| Nemotron Hybrid (7 Mamba3 + 1 Attn) | 200 | 2.09 | 524 | 1xH100 |
| **Nemotron Hybrid** | **2000** | **1.292** | **470** | **1xH100** |
| Baseline full (8xH100, 13780 steps) | 13780 | 1.224 | 43 | 8xH100 |

2000-step result (1.292) is within 0.067 of baseline with only 1/7 the steps. Still improving.

## Unexplored Directions (No PRs exist for these)

These are our competitive advantages — nobody in the competition has tried them:

1. **Mamba depth recurrence** — reuse SSM layers (all existing depth recurrence PRs use Transformer only)
2. **Mamba-aware quantization (Q-Mamba/DSQ)** — specialized quantization for SSM outlier patterns
3. **Drop RoPE on Mamba layers** — Mamba-3 complex-valued states encode position natively
4. **Knowledge distillation to Mamba** — train Transformer teacher, distill to Mamba hybrid student
5. **Ternary Mamba** — 1.58-bit quantization on SSM (all ternary PRs are Transformer/BitNet)

## Environment Setup

### Modal (cloud GPU)

```bash
# Quick import test
modal run modal_train.py --mode test

# 200-step smoke test (skip GPTQ, ~2 min)
modal run modal_train.py --mode nemotron

# 2000-step medium run (~20 min)
modal run modal_train.py --mode nemotron-medium
```

Modal image: PyTorch 2.9.1 + Triton 3.5 + mamba-ssm v2.3.1 (pre-built wheel) + Mamba-3 files from `mamba3-release` branch. Image is cached.

### Key env vars for train_nemotron_hybrid.py

```bash
NUM_LAYERS=8              # total layers
NUM_ATTN_LAYERS=1         # how many are attention (rest are Mamba-3)
ATTN_PLACEMENT=even       # even/first/last/nemotron/custom (e.g. "3,7")
MAMBA3_D_STATE=64         # SSM state dimension (Nemotron-H uses 128)
MAMBA3_NGROUPS=1          # Mamba groups (Nemotron-H uses 8)
ROPE_FRACTION=1.0         # partial RoPE on attention layers (0.5 = half dims)
SWEEP_MODE=1              # skip GPTQ/serialize/eval for fast iteration
EVAL_STRIDE=0             # disable sliding eval (too slow for smoke tests)
TTT_ENABLED=0             # disable test-time training
```

## Reference PRs

| PR | Score | Key Technique | Status |
|----|-------|---------------|--------|
| #1355 | 1.1526 | Mamba-3 Hybrid + GPTQ (best SSM) | Our base |
| #852 | 1.1189 | Hymba parallel Attn+SSM | Best hybrid |
| #1013 | 1.1682 | S4D-Lin pure PyTorch SSM (torch.compile compatible) | Research |
| #1493 | 1.0810 | Current overall SOTA (pure Transformer) | Target to beat |

## Conventions

- Use `uv` for local dependency management
- Use Modal for all GPU experiments (no local GPU)
- Use `SWEEP_MODE=1` for fast iteration, disable for final submission runs
- Commit frequently with experiment results in commit messages
