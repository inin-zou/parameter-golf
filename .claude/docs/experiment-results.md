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

## Next Experiments TODO

- [ ] Fix B (untie_mlp) and re-run
- [ ] Try 4-layer recurrence (2,3,4,5) — even more aggressive
- [ ] Try recurrence only on Mamba layers (skip Attention layer 4)
- [ ] Run best config (C) for full 8xH100 10-min to get real submission bpb
- [ ] Add GPTQ int6 + LZMA to check compressed size
- [ ] Nemotron-H ablations: 2 attn layers, d_state=128, ngroups=8
