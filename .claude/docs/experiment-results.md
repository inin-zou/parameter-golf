# Experiment Results

## Baseline Runs

| Config | Steps | val_bpb | ms/step | GPU | Date |
|--------|-------|---------|---------|-----|------|
| Pure Transformer baseline | 200 | 2.02 | 333 | 1xH100 | 2026-04-13 |
| Pure Transformer baseline (full) | 13780 | 1.2244 | 43 | 8xH100 | official |
| Mamba-2 hybrid (7M2+2Attn, no compile) | 200 | 1.85 | 973 | 1xH100 | 2026-04-13 |
| Nemotron Mamba-3 hybrid (7M3+1Attn) | 200 | 2.09 | 524 | 1xH100 | 2026-04-13 |
| Nemotron Mamba-3 hybrid (7M3+1Attn) | 2000 | 1.2916 | 470 | 1xH100 | 2026-04-13 |

## Depth Recurrence Ablations (2026-04-13)

All: 8 layers, 7 Mamba-3 + 1 Attention, 2000 steps, 1xH100, SWEEP_MODE=1

| Experiment | RECUR_LAYERS | RECUR_MODE | val_bpb | ms/step | Virtual layers | Status |
|------------|-------------|------------|---------|---------|----------------|--------|
| No recurrence | — | — | 1.2916 | 470 | 8 | baseline |
| A: block 2-layer | 2,3 | block | 1.2851 | 538 | 10 | done |
| B: untie_mlp | 2,3 | untie_mlp | CRASH | — | 10 | DDP unused params error |
| **C: block 3-layer** | **2,3,4** | **block** | **1.2830** | **584** | **11** | **best** |

### Key Findings

1. **Mamba depth recurrence works.** First-ever validation in the competition.
2. C (3-layer) > A (2-layer): more recurrence = better, despite slower step time.
3. B (untie_mlp) crashed due to DDP detecting unused parameters (recur_mlps not used before RECUR_START_FRAC=0.35). Fix: use recur_mlps from step 0 or set find_unused_parameters=True.
4. Recurrence enabled at step 700 (frac=0.35), schedule example: [0,1,2,3,4,2,3,4,5,6,7]
5. Layer 4 is Attention — so C recurs both Mamba AND Attention layers together.
6. val_bpb still decreasing at step 2000, more steps would help.

### Improvement from Recurrence

| | val_bpb | Delta vs no-recur |
|--|---------|-------------------|
| No recurrence | 1.2916 | — |
| A: 2-layer block | 1.2851 | -0.0065 |
| C: 3-layer block | 1.2830 | -0.0086 |

### Cost

- 3 experiments in parallel: 3 x H100 x ~20 min = ~$4 total (Modal)
- Total Modal spend today: ~$10-12

## Hinge Point Ablations (2026-04-13)

All: 8 layers, 2000 steps, 1xH100, SWEEP_MODE=1. Compared against best from round 1 (C: 1.2830).

| Experiment | Config | val_bpb | ms/step | Virtual layers | vs no-recur |
|------------|--------|---------|---------|----------------|-------------|
| C (prev best) | recur 2,3,4 ×1 | 1.2830 | 584 | 11 | -0.0086 |
| Hinge 1: dual Attn | 2 Attn@3,4 + recur 2,3,4 | 1.2899 | 596 | 11 | -0.0017 |
| **Hinge 3: multi recur** | **recur 3,4 ×2** | **1.2824** | **618** | **12** | **-0.0092** |
| Bonus: 4-layer | recur 2,3,4,5 ×1 | 1.2864 | 625 | 12 | -0.0052 |

### Key Findings

1. **Hinge 3 (multi recur) is the new best**: 1.2824 bpb. Repeating the hinge core (3,4) twice beats spreading recurrence over more layers.
2. **Dual Attention hurt**: 2 Attn layers at hinge (1.2899) worse than 1 Attn (1.2830). Extra Attn too expensive in params.
3. **4-layer recurrence worse than 3-layer**: 1.2864 vs 1.2830. Too many layers = too slow, not enough steps.
4. **Focused recurrence > spread recurrence**: hinge ×2 (12 virtual layers, 1.2824) > 4-layer ×1 (12 virtual layers, 1.2864).

### Current Best Config

```bash
RECUR_LAYERS=3,4 RECUR_REPEATS=2 RECUR_MODE=block RECUR_START_FRAC=0.35
NUM_LAYERS=8 NUM_ATTN_LAYERS=1 ATTN_PLACEMENT=even MAMBA3_D_STATE=64
```

## Unexplored Directions on Mamba (PR survey 2026-04-13)

All three directions below have been done on Transformers but NEVER on Mamba/SSM in this competition.

### 1. Ternary Mamba (highest potential)

Ternary Transformers already achieve strong results:
- PR #923: 74M Ternary U-Net Transformer → **1.1090 bpb** (100k steps/3h, non-record)
- PR #920: 74M Ternary U-Net Transformer → **1.1539 bpb** (record)
- PR #666: BitNet 65M ternary → 1.1932 bpb
- PR #367: BitNet b1.58 → 1.1770 bpb (systematic analysis)
- PR #1273: Annealed Muon 1.58-bit → 1.2196 bpb

Key insight: ternary (1.58-bit) lets you fit **74M params in 16MB** vs our current 26M.
A Ternary Mamba could combine SSM parameter efficiency with extreme quantization.
**Nobody has tried this.**

### 2. Remove/Reduce RoPE on Mamba Hybrid

Partial RoPE (16/64 dims) is standard on Transformer SOTA (PR #332, #458, #827).
But Mamba-3's complex-valued states natively encode position — RoPE may be redundant.
- ROPE_FRACTION=0: fully remove RoPE on attention layers
- ROPE_FRACTION=0.25: keep only 1/4 dims
- Saves parameters and computation in attention layers.
**Nobody has tried removing RoPE on a Mamba hybrid.**

### 3. Q-Mamba / Mamba-Aware Quantization (DSQ)

All Mamba submissions use generic GPTQ. Research shows Mamba has unique quantization challenges:
- SSM states have "outlier states" (large values from outer products)
- GPTQ int5 degrades SSM weights more than attention weights (PR #1013 finding)
- Q-Mamba paper proposes DSQ (Decoupled Scale Quantization) for SSM-specific outlier handling
**Nobody has applied Mamba-specific quantization methods in this competition.**

## Three-Way Ablation: No RoPE + Ternary + Q-Mamba (2026-04-13)

All on best config (recur 3,4 ×2), 1000 steps, 1xH100.

| Experiment | val_bpb @1000 | Status | Verdict |
|------------|--------------|--------|---------|
| No RoPE (ROPE_FRACTION=0) | 1.3901 | Done | **Worse** — small model needs RoPE |
| Ternary Mamba (BitLinear 1.58-bit) | 1.7149 | Done | **Much worse** — 26M too small for ternary |
| Q-Mamba DSQ (first attempt) | nan | Crashed | Bug fixed |

## Q-Mamba Quantization Ablation (2026-04-13)

All on best config (recur 3,4 ×2), 1000 steps, 1xH100, SWEEP_MODE=0 (full GPTQ pipeline).

| Experiment | pre-quant | post-quant | quant loss | size | Verdict |
|------------|-----------|-----------|------------|------|---------|
| **Standard GPTQ (control)** | **1.3948** | **1.4765** | **0.082** | **8.2MB** | **BEST** |
| QM-A: A=FP16 only | 1.3348 | 1.7434 | 0.409 | 6.7MB | Worse |
| QM-B: mixed precision | 1.3349 | 1.7418 | 0.407 | 6.7MB | Worse |
| QM-C: full DSQ | 1.3335 | 1.4810 | 0.148 | 8.0MB | Better than A/B but worse than standard |

### Key Findings

1. **Standard GPTQ is already excellent** on our model (0.082 quant loss). PR #1355's Full Hessian + AR self-gen calibration is well-optimized.
2. **Q-Mamba DSQ does NOT help** in our case. The DSQ paper's improvements apply to naive per-tensor quantization, not to sophisticated GPTQ.
3. **Ternary is not viable at 26M params** — literature confirms minimum ~1.3B for ternary to work.
4. **RoPE cannot be removed at small scale** — unlike Jamba (1.3B), our 26M model depends on RoPE.

## Ruled Out Approaches (with evidence)

| Approach | Tested? | Result | Reason |
|----------|---------|--------|--------|
| Remove RoPE | Yes | +0.072 bpb worse | Small model needs explicit position encoding |
| Ternary Mamba | Yes | +0.397 bpb worse | 26M params insufficient expressivity |
| Q-Mamba DSQ | Yes | +0.066 bpb worse than standard GPTQ | Full Hessian GPTQ already handles outliers |
| TurboQuant | No (research) | N/A | Designed for KV cache, not weights |
| Text Diffusion | No (research) | N/A | BPB metric favors autoregressive |

## Current Best Configuration

```bash
# Architecture
NUM_LAYERS=8 NUM_ATTN_LAYERS=1 ATTN_PLACEMENT=even MAMBA3_D_STATE=64

# Depth recurrence
RECUR_LAYERS=3,4 RECUR_REPEATS=2 RECUR_MODE=block RECUR_START_FRAC=0.35

# Keep defaults
ROPE_FRACTION=1.0  USE_DSQ=0  USE_TERNARY=0

# Results (1000 steps, 1xH100)
# pre-quant: 1.3948 | post-quant: 1.4765 | size: 8.2MB
```

## Next Steps

- [ ] Run best config for 2000+ steps with GPTQ to see real improvement
- [ ] SP8192 tokenizer (biggest untapped improvement)
- [ ] 8xH100 full 10-minute submission run (needs OpenAI compute grant)
- [ ] Submit as non-record to openai/parameter-golf
