# Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA

## Score: mean val_bpb = 1.1483 (3 seeds: 1.1488, 1.1485, 1.1476)

Trained on 8×H100 SXM in 600 seconds. 15.92MB artifact (int6+zstd-22).

## Approach

Seven techniques stacked on the baseline 9-layer, 512-dim GPT:

### 1. Per-Row Int6 Quantization + zstd-22 Compression
MLP and attention weight matrices are quantized to int6 ([-32, 31]) with per-row scaling. Tied embeddings remain in fp16 (quantization-sensitive). The last transformer layer's key projection is also kept in fp16 to reduce the quantization penalty on late-layer attention. zstd at level 22 provides ~5% better compression than zlib-9 on int6 data, freeing additional bytes for parameters.

### 2. 3× MLP Expansion
MLP hidden dimension increased from 1024 (2×) to 1536 (3×), enabled by the byte savings from int6 quantization. This is the single largest contributor to the improvement over int8-based submissions.

### 3. SmearGate
A learned gate that blends each token's embedding with the previous token's embedding, providing lightweight bigram-level context at the embedding layer. Adds ~512 parameters. Helps the model capture local dependencies without increasing sequence modeling cost.

### 4. BigramHash Embedding
A 4096-bucket hash table (dim=128, projected to 512) that maps adjacent token pairs to learned embeddings. The hash function `(prev_token * 31 + curr_token) % 4096` provides collision-resistant coverage of the 1M possible bigram pairs. Adds ~524K parameters. Complements SmearGate by providing an additive bigram signal rather than a multiplicative gate.

### 5. Orthogonal Weight Initialization
All large weight matrices initialized with `torch.nn.init.orthogonal_(gain=1.0)`. Output projections (attention proj, MLP proj) are additionally scaled by `1/sqrt(2 * num_layers)` following muP conventions. This accelerates early convergence by starting from a well-conditioned point, giving Muon a head start.

### 6. Muon Optimizer with Weight Decay
Muon optimizer with decoupled weight decay (WD=0.02) applied after the Newton-Schulz gradient update. Momentum warmup from 0.92 to 0.99 over 1500 steps. AdamW (WD=0.01) for embedding and scalar parameters. Weight decay regularizes weight magnitudes, directly improving int6 quantization quality.

### 7. Stochastic Weight Averaging (SWA)
SWA enabled over the last 50% of training, averaging checkpoints every 200 steps. Produces smoother weight distributions that quantize better, reducing the int6 quantization penalty.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 9 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 (warmup from 0.92) |
| grad_clip_norm | 0.3 |
| weight_decay | 0.01 (AdamW) / 0.02 (Muon) |
| eval_stride | 64 |
| bigram_vocab_size | 4096 |
| bigram_dim | 128 |
| compressor | zstd (level 22) |

## Key Metrics

- **Mean val_bpb: 1.1483** (seeds 1337, 42, 7)
- Pre-quant val_bpb: 1.1640
- Quantization penalty: 0.016 bpb (int6 vs fp16)
- Training: 7,373 steps in 600s (81.4 ms/step)
- Model params: ~22M
- Artifact size: 15.92MB (int6+zstd-22)

## Reproducibility

Three independent training runs with different random seeds, all other settings identical:

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.93978 | 1.14885 |
| 42 | 1.93923 | 1.14852 |
| 7 | 1.93762 | 1.14757 |
| **Mean** | **1.93888** | **1.14831** |
| **Std** | **0.00111** | **0.00066** |

Improvement over current SOTA (1.1748): **-0.0265 bpb / -0.0459 nats** (p < 0.001 by one-sample t-test against 1.1748 with 3 samples).
