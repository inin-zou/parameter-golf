# Parameter Golf — Nemotron-H Inspired Mamba-3 Hybrid

## Project Overview

OpenAI Parameter Golf competition: train the best LM that fits in 16MB, under 10 min on 8xH100.
**Deadline: April 30, 2026 (TOMORROW).** Metric: val_bpb (bits per byte) — lower is better.

- Fork: https://github.com/inin-zou/parameter-golf
- Upstream: https://github.com/openai/parameter-golf
- **PR submitted**: https://github.com/openai/parameter-golf/pull/1607
- Current SOTA: 1.0810 bpb (pure Transformer, PR #1493)
- Best SSM: 1.1526 bpb (PR #1355)
- Baseline: 1.2244 bpb

## Current Status

**Phase 1 COMPLETE. Phase 2 in progress. No OpenAI compute credits received.**

### Branches
- `main` — all experiment files + docs
- `experiments` — full experiment archive
- `submission/nemotron-h-mamba3-depth-recurrence` — clean PR branch (only submission folder)

### Best Config
```bash
NUM_LAYERS=8 NUM_ATTN_LAYERS=1 ATTN_PLACEMENT=even MAMBA3_D_STATE=64
RECUR_LAYERS=3,4 RECUR_REPEATS=2 RECUR_MODE=block RECUR_START_FRAC=0.35
ROPE_FRACTION=1.0  # DO NOT remove — hurts at small scale
```

### Results

| Config | Steps | val_bpb | post-quant | ms/step | GPU |
|--------|-------|---------|-----------|---------|-----|
| Baseline (pure Transformer) | 200 | 2.02 | 2.02 | 333 | 1xH100 |
| Nemotron Hybrid (7M3+1Attn) | 2000 | 1.292 | — | 470 | 1xH100 |
| + depth recur 3,4 ×2 (best arch) | 2000 | **1.282** | — | 618 | 1xH100 |
| Best config + GPTQ | 1000 | 1.395 | **1.477** | 624 | 1xH100 |

### 1000-Step Benchmark (for cheap iteration)

PR #1355 reference: 1000 steps = 1.279 bpb → final 1.153.
Scaling rule: **1000-step × 0.90 ≈ final score.**

| 1000-step bpb | Predicted final | Meaning |
|---------------|----------------|---------|
| 1.33 (current) | ~1.20 | Near baseline |
| **1.29 (target)** | **~1.16** | **Near best SSM → worth 8xH100 run** |
| 1.20 | ~1.08 | Near SOTA (unlikely) |

**Gate: if 1000-step ≤ 1.29, commit $5 for 8xH100 full run.**

## Phase 1 Findings (COMPLETE)

### What Works
- Mamba-3 SISO hybrid (7 SSM + 1 Attn) — viable architecture
- Depth recurrence on Mamba layers — **first in competition**, -0.009 bpb
- Hinge-point focused recurrence > spread recurrence
- Standard GPTQ int6 — 0.082 quant loss, good enough

### What Doesn't Work (at 26M params)
- Ternary Mamba: +0.397 worse (needs ≥1.3B params)
- Q-Mamba DSQ: +0.066 worse than standard GPTQ
- Remove RoPE: +0.072 worse (small model needs explicit position)
- Dual attention at hinge: -0.0017 only, not worth extra Attn cost

## Phase 2 Plan (IN PROGRESS)

See `.claude/docs/phase2-plan.md` for full details (67+ papers researched).

### Phase 2A — Zero/low cost (do first)
1. **SP8192 tokenizer** — change env var + download data, +0.04 bpb
2. **Enable TTT + EMA** — change env var, +0.01 bpb
3. **DeepNorm residual scaling** — 30 min code, +0.005 bpb

### Phase 2B — Needs compute
4. **JEPA auxiliary loss** — +0.005 bpb (PR #1243 proved it works)
5. **Hadamard rotation during training** — +0.03 post-quant improvement
6. **LAuReL-RW residual** — +0.005 bpb

### Phase 2C — Stretch
7. **TOP auxiliary prediction** — +0.005 bpb
8. **8xH100 full 10-min run** — needs compute

## Key Files

| File | Purpose |
|------|---------|
| `train_nemotron_hybrid.py` | Main training script (forked from PR #1355 + our additions) |
| `train_ternary_mamba.py` | Ternary experiment (ruled out) |
| `train_qmamba.py` | Q-Mamba DSQ experiment (ruled out) |
| `train_gpt_hybrid.py` | Earlier Mamba-2 experiment (deprecated) |
| `modal_train.py` | Modal cloud GPU deployment |
| `reference_pr1355.py` | PR #1355 original for reference |
| `.claude/docs/experiment-results.md` | All experiment data |
| `.claude/docs/phase2-plan.md` | Phase 2 plan with 67+ paper citations |
| `.claude/docs/competition-strategy.md` | Overall strategy |

## Environment

### Modal (cloud GPU)
```bash
modal run modal_train.py --mode test           # verify Mamba-3 imports
modal run modal_train.py --mode nemotron       # 200-step smoke (~2 min, $0.20)
modal run modal_train.py --mode nemotron-medium # 2000-step (~20 min, $0.80)
modal run modal_train.py --mode baseline-gptq  # 1000-step + full GPTQ ($1.50)
modal run modal_train.py --mode recur-ablation # 3 depth recurrence configs parallel
modal run modal_train.py --mode hinge-ablation # 3 hinge point configs parallel
modal run modal_train.py --mode qmamba-ablation # 3 Q-Mamba configs parallel
```

Image: PyTorch 2.9.1 + Triton 3.5 + mamba-ssm v2.3.1 + Mamba-3 from mamba3-release branch. Cached.

**Remaining Modal credits: ~$4. Enough for ~5 × 1000-step runs or 1 × 8xH100 full run.**

### Key env vars
```bash
# Architecture
NUM_LAYERS=8  NUM_ATTN_LAYERS=1  ATTN_PLACEMENT=even  MAMBA3_D_STATE=64

# Depth recurrence (best config)
RECUR_LAYERS=3,4  RECUR_REPEATS=2  RECUR_MODE=block  RECUR_START_FRAC=0.35

# Fast iteration
SWEEP_MODE=1  EVAL_STRIDE=0  TTT_ENABLED=0

# Full pipeline (for submission)
SWEEP_MODE=0  EVAL_STRIDE=32  TTT_ENABLED=1  EMA_ENABLED=1
```

## Reference PRs

| PR | Score | Key Technique |
|----|-------|---------------|
| #1493 | 1.0810 | SOTA: SP8192 + 3-layer recur + parallel residuals + TTT |
| #1355 | 1.1526 | Best SSM: Mamba-3 + GPTQ + Late QAT + MuonEq-R |
| #1243 | 1.1230 | Best JEPA: JEPA + Leader-Stack |
| #852 | 1.1189 | Best hybrid: Hymba parallel Attn+SSM |
| #1241 | 0.9901 | MDLM Diffusion (21h training, not 10-min compliant) |
| **#1607** | **1.4765** | **Ours: Nemotron Mamba-3 + hinge depth recurrence** |

## Conventions

- Use `uv` for local dependency management
- Use Modal for all GPU experiments (Mac only, no local GPU)
- `SWEEP_MODE=1` for fast iteration, `0` for submission runs
- 1000-step benchmark for cheap config comparison ($0.80/run)
- Commit frequently with experiment results in messages
- User communicates in Chinese and English
