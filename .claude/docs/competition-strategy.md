# Parameter Golf Competition Strategy

## Competition Overview

- **Goal**: Train the best language model that fits in 16MB, trained in under 10 min on 8xH100
- **Metric**: val_bpb (bits per byte) on FineWeb validation set — lower is better
- **Baseline**: 1.2244 bpb | **Current SOTA**: 1.0810 bpb
- **Deadline**: April 30, 2026

## Current SOTA Architecture (1.0810 bpb)

```
SP8192 vocab (8192 BPE tokens)
├── 11-layer Transformer, 512 dim, 8 heads (4 KV heads, GQA)
├── MLP: 4x expansion (2048 hidden), LeakyReLU(0.5)^2
├── Partial RoPE (16/64 dims), LayerNorm scale per layer
├── Depth recurrence: layers 3-5 reused → 17 virtual layers from 11 physical
├── Parallel residuals: layer 7+ attention & MLP in parallel (GPT-J style)
├── GPTQ int6 (weights, k=12.85σ) + int8 (embeddings, k=20.0σ)
├── Brotli-11 compression + byte-shuffle
├── MuonEq-R optimizer + EMA (decay=0.9965)
├── Score-First TTT: SGD adaptation (lr=0.005, 3 epochs) on already-scored tokens
└── Total artifact: 15.99MB
```

- **Parameters**: ~36M (compressed from 135MB to 16MB)
- **Training**: 4550 steps in 588s on 8xH100
- **Eval**: Sliding window + TTT, ~500s eval budget

## Potential Approaches

### Direction A: More Aggressive Quantization

**Idea**: Push below int6 to int4/int3 with GPTQ or alternative methods, fitting a larger model in 16MB.

**Techniques**:
- GPTQ int4 with group quantization (group_size=32/64/128)
- QuIP# (incoherence processing for sub-4-bit)
- AQLM (additive quantization for 2-bit)
- Mixed precision: int3 for less sensitive layers, int6 for critical ones

**Pros**:
- Same 16MB budget → bigger model → lower loss
- GPTQ int4 has mature PyTorch tooling
- Can be combined with existing SOTA stack

**Cons**:
- Below int4, quality degrades fast for small models
- Calibration becomes harder at lower bits
- Diminishing returns if model is already well-compressed

**Risk**: Medium | **Potential gain**: 0.01-0.03 bpb

### Direction B: Nemotron-H Inspired Hybrid Mamba-Transformer (PRIMARY)

**Idea**: Inspired by NVIDIA's Nemotron-H architecture, build a hybrid model where
most layers are Mamba (SSM) blocks with only a few attention layers for long-range
dependencies. This maximizes parameter efficiency under the 16MB constraint.

**Inspiration — Nemotron-H-8B architecture**:
- 52 layers total: 24 Mamba-2 + 24 MLP + 4 Attention (~8% attention)
- Mamba-2: head_dim=64, expansion=2, conv_window=4
- Attention: GQA with 8 KV heads, Squared ReLU MLP
- Attention layers evenly dispersed, always preceding an MLP layer
- First layer is always Mamba, last layer is always MLP

**Also informed by Mamba-3 (March 2026)**:
- Complex-valued state spaces for rotational position encoding (like RoPE)
- MIMO architecture: 4x parallel ops per step via matrix-mult state updates
- ~4% better language modeling than Transformers at same parameter count
- 7x faster at long sequences

**Proposed design for Parameter Golf (512 dim, 16MB budget)**:

```
Hybrid Model (~46M params, targeting ~15MB after int6 GPTQ):
├── Embedding: SP8192 tied embeddings (~4.2M params)
├── Layer 0-3:   Mamba-2/3 blocks (~0.8M each = 3.2M)
├── Layer 4:     Attention (GQA, 4KV) + MLP (~3.4M)
├── Layer 5-8:   Mamba-2/3 blocks (~0.8M each = 3.2M)
├── Layer 9:     Attention (GQA, 4KV) + MLP (~3.4M)
├── Layer 10-15: Mamba-2/3 blocks (~0.8M each = 4.8M)
├── All layers:  MLP 4x expansion (~2.1M × 16 = 33.6M shared w/ above)
└── Depth recurrence on Mamba layers (reuse layers 2-5)
```

**Parameter comparison**:

| Component | Per-layer params (512 dim) |
|-----------|---------------------------|
| Attention (GQA, 4KV) | ~1.3M (Q/K/V projections + output) |
| Mamba-2 (expansion=2) | ~0.8M (no Q/K/V, just SSM params) |
| MLP (4x) | ~2.1M (same for both) |
| **Attention + MLP** | **~3.4M** |
| **Mamba + MLP** | **~2.9M** |

Mamba layers save ~15% params per layer vs attention. With 14 Mamba layers + 2 attention
layers, we get a 16-layer model (~46M params) vs SOTA's 11-layer model (~36M params).
That's 28% more parameters and 45% more layers in the same 16MB budget.

**Why this could win**:
1. More layers = more representational depth = lower loss
2. Mamba handles local patterns (majority of text) cheaply
3. Few attention layers capture long-range dependencies where needed
4. Depth recurrence on Mamba layers is natural (reuse SSM blocks)
5. Mamba's O(n) inference allows longer eval context within 10min budget
6. Mamba-3's complex-valued states provide position info without RoPE overhead

**Key risks**:
- Mamba kernel integration with existing GPTQ pipeline
- Small-model Mamba behavior is less studied (most results are 1B+)
- Training stability with mixed Mamba/Attention may need tuning
- `mamba_ssm` package dependency must work on H100 + PyTorch

**Implementation plan**:
1. Get `mamba_ssm` (or Mamba-3 from `state-spaces/mamba`) working in isolation
2. Modify `train_gpt.py` to support hybrid layers
3. Smoke test on Modal 1xH100 (200 steps)
4. Compare train_loss curves: hybrid vs pure-transformer at same param count
5. If promising, optimize quantization and scale to 8xH100

**Risk**: High | **Potential gain**: 0.02-0.05 bpb (if it works, could be a big jump)

**References**:
- Nemotron-H paper: https://arxiv.org/abs/2504.03624
- Mamba-3 blog: https://tridao.me/blog/2026/mamba3-part1/
- Mamba GitHub: https://github.com/state-spaces/mamba

### Direction C: Incremental SOTA Stack Improvements

**Idea**: Build on the current SOTA stack with targeted improvements.

**Possible improvements**:
- Extend depth recurrence to more layers (currently 3-5, try 2-6 or 3-7)
- Tune QK-Gain beyond 5.25
- Optimize warmdown schedule and learning rate
- Better TTT strategy (more epochs, different LR schedule)
- Try MLP expansion 5x or 6x instead of 4x
- Experiment with different vocab sizes (SP16384?)

**Pros**:
- Lowest risk — proven stack, incremental changes
- Easy to validate with A/B experiments
- Most likely to produce a valid submission

**Cons**:
- Hard to beat SOTA by the required 0.005 nats
- Diminishing returns on a heavily-optimized stack
- Less interesting / novel

**Risk**: Low | **Potential gain**: 0.002-0.01 bpb

### Direction D: Evaluation-Side Optimization

**Idea**: Keep the model the same, improve how we evaluate (TTT, sliding window, context length).

**Possible improvements**:
- Longer eval context (current is 1024, try 2048/4096)
- Better TTT: more sophisticated adaptation (Adam instead of SGD, per-layer LR)
- Multi-pass scoring strategies
- Sliding window stride optimization

**Pros**:
- Doesn't require retraining — fast to iterate
- TTT alone contributed ~0.007 bpb improvement historically
- Can be combined with any model

**Cons**:
- Eval is capped at 10 minutes
- Must stay within legal TTT rules (score-first)
- Limited headroom

**Risk**: Low | **Potential gain**: 0.002-0.005 bpb

### Direction E: JEPA for Language Modeling (Exploratory)

**Idea**: Apply Joint Embedding Predictive Architecture to language model pretraining.
Predict continuous embeddings instead of discrete tokens.

**Recent research (2025-2026)**:
- **LLM-JEPA** (arXiv 2509.14252): JEPA applied to LLM pretrain/finetune, significantly
  outperforms standard training objectives on Llama3, Gemma2, Olmo families
- **VL-JEPA** (arXiv 2512.10942): Vision-language JEPA achieves stronger performance
  with 50% fewer trainable parameters than standard token-space VLM training
- Key insight: embedding-space prediction is far superior to input-space prediction

**Why it's interesting for Parameter Golf**:
- 50% fewer trainable parameters = much more headroom in 16MB budget
- Unclaimed on competition wishlist — guaranteed novelty
- Could be a strong non-record submission even if it doesn't beat SOTA

**Challenges**:
- val_bpb evaluation requires per-token log probabilities; JEPA predicts embeddings,
  not token distributions — needs an additional decoding head or adapter
- No existing small-scale JEPA language model to reference
- High engineering effort to integrate with existing pipeline

**Status**: Exploratory / future direction. Not primary focus.

**References**:
- LLM-JEPA: https://arxiv.org/abs/2509.14252
- VL-JEPA: https://arxiv.org/abs/2512.10942

## Ruled Out Approaches

### Text Diffusion
- Evaluation metric (bits per byte) naturally favors autoregressive models
- Diffusion models haven't consistently beaten AR models on perplexity/bpb
- High engineering effort with uncertain payoff
- **Verdict**: Not suitable given the bpb evaluation metric

## Ruled Out Approaches

### TurboQuant
- Designed for KV cache compression, not weight quantization
- No PyTorch implementation (only llama.cpp/GGML)
- Weight-compression derivatives are ~4-5 BPW, not better than current int6
- **Verdict**: Not suitable for this competition

## Recommended Strategy: Nemotron-H Architecture + Mamba-3 Kernel + PR #1355 Pipeline

Combine three sources:
- **Nemotron-H** (architecture): alternating SSM/MLP layout, evenly distributed attention, no RoPE
- **Mamba-3** (SSM kernel): complex-valued states, MIMO, Triton kernels
- **PR #1355** (training pipeline): GPTQ int6, MuonEq-R, Late QAT, LZMA, torch.compile(fullgraph=False)

### Concrete Architecture Plan

```
8 layers, dim=512, seq_len=4096:
├── Layer 0: Mamba-3 + MLP
├── Layer 1: Mamba-3 + MLP
├── Layer 2: Mamba-3 + MLP
├── Layer 3: Attention (GQA 8h/4kv) + MLP  ← evenly placed
├── Layer 4: Mamba-3 + MLP
├── Layer 5: Mamba-3 + MLP
├── Layer 6: Mamba-3 + MLP
├── Layer 7: Mamba-3 + MLP
├── U-Net skip connections (encoder/decoder halves)
├── Mamba-3: SISO, d_state=64, expand=2, headdim=64, chunk_size=64
├── No RoPE on Mamba layers (Mamba-3 complex states encode position)
├── RoPE only on Attention layer (partial, 16/64 dims)
└── MLP: LeakyReLU(0.5)^2, 4x expansion
```

### Training Pipeline
- torch.compile(fullgraph=False) — proven at 115ms/step in PR #1355
- MuonEq-R optimizer + EMA
- Late QAT (activate at lr_mul < 0.15)
- Full Hessian GPTQ int6 with AR self-gen calibration data
- LZMA compression
- Target: ~15.8MB artifact

### Key Differences from PR #1355
1. Nemotron-H inspired layer placement (evenly spaced attention)
2. Potential: try separated Mamba/MLP layers (Nemotron-H style) vs combined blocks
3. Potential: try d_state=128 (Nemotron-H uses this)
4. Potential: drop RoPE entirely (Mamba-3 complex states provide position info)

### Experiment Plan
1. First replicate PR #1355 baseline (1.1526 bpb) on 8xH100
2. Then ablate Nemotron-H inspired changes one at a time
3. Submit best result as non-record submission

### Reference PRs
- PR #1355: Best SSM (1.1526 bpb) — https://github.com/openai/parameter-golf/pull/1355
- PR #852: Best Hymba (1.1189 bpb) — https://github.com/openai/parameter-golf/pull/852
- PR #1013: S4D-Lin (torch.compile fix) — https://github.com/openai/parameter-golf/pull/1013

**Fallback**: If hybrid doesn't work, fall back to incremental SOTA improvements (Direction C)

## Infrastructure

- **Local (Mac)**: MLX smoke tests, code iteration
- **Modal 1xH100**: Medium experiments (2000 steps, ~$0.80)
- **Modal 8xH100**: Full runs (10 min, ~$5.30)
- **$30 Modal free credits** available
- **OpenAI compute grant**: Applied/pending
