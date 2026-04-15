# Phase 2 Plan: Advanced Optimizations for Nemotron Mamba-3 Hybrid

## Starting Point (Phase 1 Results)

```
Best config: 7 Mamba-3 + 1 Attention, 8 layers
Depth recurrence: RECUR_LAYERS=3,4 RECUR_REPEATS=2 (12 virtual layers)
Post-quant val_bpb: 1.4765 (1000 steps, 1xH100, GPTQ int6+LZMA, 8.2MB)
Pre-quant val_bpb: 1.2824 (2000 steps, best architecture config)
```

### Phase 1 Ruled Out
- Ternary Mamba: not viable at 26M params (needs ≥1.3B)
- Q-Mamba DSQ: standard GPTQ already better (0.082 vs 0.148 quant loss)
- Remove RoPE: hurts at small scale (+0.072 bpb)
- Text Diffusion: too slow for 10-min limit

---

## Phase 2 Directions (Ranked by Priority)

### Direction 1: JEPA Auxiliary Loss (HIGH PRIORITY)

**What:** Add JEPA embedding prediction as auxiliary loss alongside standard next-token prediction.

**Evidence:**
- PR #1243 achieved 1.1230 bpb with JEPA + Leader-Stack (best JEPA in competition)
- LLM-JEPA paper: consistent improvements across Llama3, Gemma2, OLMo (1B-8B scale)
- SSAMBA paper: self-supervised auxiliary loss works on Mamba SSM architecture
- NextLat paper: latent-space prediction improves world modeling and language modeling

**Concerns:**
- Parameter Golf data shows only -0.24% improvement on real text at small scale
- 40-50% training overhead (mitigated by loss dropout at 0.5)
- Needs EMA target encoder (extra memory)

**Implementation Recipe:**
```python
# Total loss = NTP + lambda * JEPA
loss_ntp = cross_entropy(logits, targets)
loss_jepa = 1 - cosine_similarity(predictor_output, target_embedding.detach())
loss_var = max(0, 1 - std(z, dim=0)).mean()  # VICReg anti-collapse
loss = loss_ntp + 0.10 * (loss_jepa + 0.5 * loss_var)
```

- Lambda warmup: 0 for steps 0-500, ramp to 0.10 by step 2000
- EMA target encoder: tau=0.996
- Loss dropout: p=0.5 (compute JEPA loss 50% of steps)
- Predictor: reuse model with K=2 appended [PRED] tokens (no extra network)

**Expected gain:** +0.005 bpb (conservative, based on PR #1243)
**Implementation effort:** Medium (1-2 hours)
**Compute cost:** ~50% more per step, but can offset with loss dropout

### Direction 2: Hadamard Rotation During Training (HIGH PRIORITY)

**What:** Insert fixed Hadamard rotations in forward/backward pass to eliminate activation outliers, making the model inherently quantization-friendly.

**Evidence:**
- PolarQuant: "Hadamard rotation alone accounts for 98% of quality improvement"
- HALO paper: achieves 1.27-1.41x SPEEDUP (not slowdown) by enabling 8-bit compute
- MambaQuant: plain Hadamard insufficient for Mamba — KLT-enhanced rotation needed
- Quamba: massive outliers in SSM output activations eliminated by Hadamard
- RoSTE: learned rotations during fine-tuning consistently beat post-training rotation

**Implementation Recipe:**
```python
# Insert at 5 points in each Mamba block:
# 1. Before gate projection (biggest outlier source)
# 2. After selective scan output (massive outliers per Quamba)
# 3. After BCNorm on B/C matrices
# For attention layers:
# 4. Replace output projection with Hadamard + diagonal (saves 25% attention params)
# 5. Pre-QK rotation for FP8 attention compatibility

from fast_hadamard_transform import hadamard_transform
x = hadamard_transform(x)  # O(d log d), fixed matrix, zero params
```

- Use HadaCore kernel for GPU acceleration
- Start with fixed random-sign Hadamard; optionally learn rotations later (SpinQuant/Cayley SGD)
- Expected: model produces quantization-friendly distributions → GPTQ quant loss drops from 0.082 to <0.03

**Expected gain:** -0.03-0.05 bpb improvement in post-quant score
**Implementation effort:** Medium (2-3 hours)
**Compute cost:** ~8% overhead per step, offset by potential 8-bit training

### Direction 3: Hyperconnection / Advanced Residual (MEDIUM PRIORITY)

**What:** Replace standard residual connections with learned scaling to improve depth recurrence effectiveness.

**Evidence:**
- Hyper-Connections (ByteDance, ICLR 2025): 0.034 val loss reduction on OLMo-1B
- DeepSeek mHC: BBH 43.8→51.0 on 27B model
- LAuReL (Google, ICML 2025): +4-20% downstream at 1B-4B, only 0.012% params
- Transponder: 9-15% perplexity reduction at 60-250M params with <1% overhead
- DeepNorm: zero-cost scaling constants proven for very deep (200+ layer) networks

**Best choices for our Mamba-3 + depth recurrence:**

| Method | Overhead | Compatible with recurrence? | Recommended? |
|--------|----------|---------------------------|-------------|
| DeepNorm scaling | 0% | Excellent | **Yes — do first** |
| LAuReL-RW (2 scalars/layer) | 0.012% | Yes | **Yes** |
| Transponder (contextual modulation) | <1% | Yes | Yes |
| Vanilla HC/mHC | 6.7% | No (multi-stream conflicts) | No |
| DCA/MUDDFormer | 0.2% | No (needs all layer outputs) | No |

**Implementation Recipe:**
```python
# DeepNorm (zero cost, add to existing residual):
alpha = (2 * num_virtual_layers) ** 0.25  # depth-dependent constant
x = alpha * x + block(x)  # scale residual up

# LAuReL-RW (2 learnable scalars per block):
self.res_alpha = nn.Parameter(torch.ones(1))
self.res_beta = nn.Parameter(torch.ones(1))
x = self.res_alpha * x + self.res_beta * block(x)
```

**Expected gain:** +0.005-0.01 bpb (especially for depth recurrence)
**Implementation effort:** Low (30 minutes for DeepNorm, 1 hour for LAuReL)
**Compute cost:** 0% (DeepNorm) to <1% (LAuReL)

### Direction 4: Multi-Token / Auxiliary Token Prediction (MEDIUM PRIORITY)

**What:** Add auxiliary prediction heads to improve representation quality during training.

**Evidence:**
- Token Order Prediction (TOP): gains at 340M params, single unembedding layer, zero inference overhead
- DeepSeek-V3 MTP: lambda=0.3 → 0.1 schedule, 1.8x speculative decoding
- MuToR registers: architecture-agnostic, negligible parameters
- Pre-Training Curriculum (ACL 2025): solves "MTP too hard for small models" via gradual ramp
- Meta MTP: benefits mainly at scale (>1B), not helpful for small models with independent heads

**Best choice for our 26M model:**

**TOP (Token Order Prediction)** — ranks upcoming tokens by proximity instead of predicting exact tokens.
- Single extra unembedding layer (~0.5% params)
- Zero inference overhead (removed at eval)
- Proven at 340M (closest to our scale)
- Architecture-agnostic

**Implementation Recipe:**
```python
# TOP auxiliary loss (from arXiv 2508.19228):
# Instead of predicting exact next-k tokens, rank them by proximity
future_embeddings = get_future_token_embeddings(targets, k=4)
scores = model_hidden @ future_embeddings.T  # similarity scores
loss_top = listwise_ranking_loss(scores, proximity_labels)
loss = loss_ntp + 0.1 * loss_top
```

**Expected gain:** +0.003-0.01 bpb
**Implementation effort:** Medium (1-2 hours)
**Compute cost:** <5% per step

### Direction 5: SP8192 Tokenizer (HIGH PRIORITY, LOW EFFORT)

**What:** Switch from SP1024 to SP8192 vocabulary.

**Evidence:**
- Biggest single improvement in SOTA history (~0.03-0.05 bpb)
- All top leaderboard entries use SP8192
- Just need to download new dataset variant

**Implementation:**
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10
# Then: VOCAB_SIZE=8192 TOKENIZER_PATH=...fineweb_8192_bpe.model
```

**Expected gain:** +0.03-0.05 bpb
**Implementation effort:** Very low (change env vars + download data)
**Compute cost:** 0%

### Direction 6: Enable TTT + EMA (HIGH PRIORITY, ZERO EFFORT)

**What:** Turn on already-implemented features in the PR #1355 pipeline.

```bash
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3  # test-time training
EMA_ENABLED=1 EMA_DECAY=0.997              # exponential moving average
```

**Expected gain:** +0.007 (TTT) + 0.003 (EMA) = +0.01 bpb
**Implementation effort:** Zero (change env vars)
**Compute cost:** TTT uses eval budget (10 min separate from training)

---

## Recommended Execution Order

```
Phase 2A (immediate, zero/low cost):
  1. SP8192 tokenizer           — change env var, +0.04 bpb
  2. Enable TTT + EMA           — change env var, +0.01 bpb
  3. DeepNorm residual scaling  — 30 min code, +0.005 bpb

Phase 2B (when compute available):
  4. JEPA auxiliary loss         — 2 hours code, +0.005 bpb
  5. Hadamard rotation training  — 3 hours code, +0.03 post-quant
  6. LAuReL-RW residual         — 1 hour code, +0.005 bpb

Phase 2C (stretch):
  7. TOP auxiliary prediction   — 2 hours code, +0.005 bpb
  8. 8xH100 full 10-min run    — needs OpenAI compute grant
```

**Estimated total improvement from Phase 2:** +0.10-0.15 bpb
**Projected post-quant val_bpb:** ~1.33-1.38 (from current 1.4765)
**Projected with 8xH100 full run:** ~1.18-1.25

---

## Research Sources Summary

### JEPA (18 papers)
Key: LLM-JEPA, I-JEPA, VICReg, C-JEPA, SSAMBA, NextLat, Parameter Golf PR #1243/#1480/#1581

### Hadamard Rotation (18 papers)
Key: HALO, SpinQuant, QuaRot, PolarQuant, MambaQuant, Quamba, RoSTE, HadaCore, GSR

### Hyperconnection (16 papers)
Key: Hyper-Connections, mHC, DeepCrossAttention, LAuReL, Transponder, DeepNorm, SpanNorm, Retrofitted Recurrence

### Multi-Token Prediction (15 papers)
Key: Meta MTP, DeepSeek-V3, TOP, MuToR, FSP, L-MTP, FastMTP, GenRM, RLP
