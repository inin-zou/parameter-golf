# Nemotron-H Inspired Mamba-3 Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Mamba-3 hybrid training script that combines PR #1355's proven pipeline with Nemotron-H architectural insights, targeting < 1.15 bpb on 8xH100.

**Architecture:** Fork PR #1355's `train_mamba3_hybrid.py` (1812 lines, proven at 1.1526 bpb, 115ms/step). Add Nemotron-H inspired modifications: configurable attention placement strategy, optional d_state=128, and env-var toggles for ablation. Keep the script self-contained and runnable via `torchrun`.

**Tech Stack:** PyTorch 2.6+, mamba-ssm v2.3.1 (Mamba3 module), GPTQ int6, LZMA, Modal for cloud GPU.

**Reference files:**
- `/Users/yongkangzou/Desktop/Hackathons/Parameter Golf/reference_pr1355.py` — PR #1355 training script (proven baseline)
- `/Users/yongkangzou/Desktop/Hackathons/Parameter Golf/train_gpt.py` — competition baseline
- `/Users/yongkangzou/Desktop/Hackathons/Parameter Golf/modal_train.py` — Modal deployment script

---

### Task 1: Create our training script from PR #1355

**Files:**
- Create: `train_nemotron_hybrid.py` (copy from `reference_pr1355.py`, rename)

This is our starting point — a working, proven script at 1.1526 bpb.

- [ ] **Step 1: Copy PR #1355 script as our base**

```bash
cp reference_pr1355.py train_nemotron_hybrid.py
```

- [ ] **Step 2: Update the docstring**

Change the top docstring in `train_nemotron_hybrid.py` to:

```python
"""Nemotron-H inspired Mamba-3 Hybrid for Parameter Golf.

Based on PR #1355 (best SSM, 1.1526 bpb) with Nemotron-H architectural insights:
- Configurable attention placement strategy (evenly spaced, first-layers, last-layers)
- Configurable d_state (64 or 128, Nemotron-H uses 128 for 8B)
- Configurable ngroups (1 or 8, Nemotron-H uses 8)
- Ablation-friendly env-var driven config
"""
```

- [ ] **Step 3: Commit**

```bash
git add train_nemotron_hybrid.py
git commit -m "feat: fork PR #1355 as train_nemotron_hybrid.py base"
```

---

### Task 2: Add Nemotron-H attention placement strategies

**Files:**
- Modify: `train_nemotron_hybrid.py` — `Hyperparameters` class and `GPT.__init__`

PR #1355 uses evenly-spaced attention (`round(i * num_layers / (num_attn_layers + 1))`). Nemotron-H also uses evenly-spaced but with specific constraints (first layer = Mamba, last layer = MLP). Add configurable placement strategies for ablation.

- [ ] **Step 1: Add ATTN_PLACEMENT env var to Hyperparameters**

In the `Hyperparameters` class, after `num_attn_layers`, add:

```python
    # Attention placement strategy: "even" (PR #1355 default), "first" (low layers),
    # "last" (high layers), "nemotron" (even, but first=Mamba last=MLP)
    attn_placement = os.environ.get("ATTN_PLACEMENT", "even")
```

- [ ] **Step 2: Add placement logic to GPT.__init__**

Replace the attention index computation block in `GPT.__init__` (the `# Compute evenly spaced attention layer indices` section) with:

```python
        # Compute attention layer indices based on placement strategy
        attn_placement = os.environ.get("ATTN_PLACEMENT", "even")
        attn_indices = set()
        if num_attn_layers > 0:
            if attn_placement == "even":
                # PR #1355 default: evenly spaced
                for i in range(1, num_attn_layers + 1):
                    attn_indices.add(round(i * num_layers / (num_attn_layers + 1)))
            elif attn_placement == "first":
                # Attention in first N layers
                attn_indices = set(range(num_attn_layers))
            elif attn_placement == "last":
                # Attention in last N layers
                attn_indices = set(range(num_layers - num_attn_layers, num_layers))
            elif attn_placement == "nemotron":
                # Nemotron-H: evenly spaced but layer 0 is always Mamba
                for i in range(1, num_attn_layers + 1):
                    idx = round(i * num_layers / (num_attn_layers + 1))
                    if idx == 0:
                        idx = 1  # push away from first layer
                    attn_indices.add(idx)
            else:
                # Custom: comma-separated indices e.g. "3,7"
                attn_indices = {int(x.strip()) for x in attn_placement.split(",")}
        self.attn_indices = sorted(attn_indices)
        if rank == 0:
            print(f"attn_placement:{attn_placement} attn_indices:{self.attn_indices}")
```

Note: `rank` is not available inside GPT.__init__. Instead, just use a plain print — it will only show on rank 0 due to torchrun output handling, or guard with `int(os.environ.get("RANK", 0)) == 0`.

- [ ] **Step 3: Commit**

```bash
git add train_nemotron_hybrid.py
git commit -m "feat: add configurable attention placement strategies"
```

---

### Task 3: Add Nemotron-H inspired hyperparameter options

**Files:**
- Modify: `train_nemotron_hybrid.py` — `Hyperparameters` class

Add env vars for Nemotron-H specific config options that differ from PR #1355 defaults.

- [ ] **Step 1: Add new env vars after existing mamba3 config**

In `Hyperparameters`, ensure these env vars exist (some may already be there from PR #1355 — only add what's missing):

```python
    # Nemotron-H ablation options
    mamba3_ngroups = int(os.environ.get("MAMBA3_NGROUPS", 1))  # Nemotron-H uses 8
    num_attn_layers_2 = int(os.environ.get("NUM_ATTN_LAYERS", 1))  # try 2 for Nemotron-H style
    rope_fraction = float(os.environ.get("ROPE_FRACTION", 1.0))  # partial RoPE: 0.5 = half dims
```

- [ ] **Step 2: Wire ngroups into Mamba3Layer**

In `Mamba3Layer.__init__`, add `ngroups` parameter and pass it to `Mamba3`:

```python
class Mamba3Layer(nn.Module):
    """Pure Mamba-3 SISO layer. Uses the Mamba3 module directly."""
    def __init__(self, dim: int, d_state: int = 64, expand: int = 2,
                 headdim: int = 64, chunk_size: int = 64, ngroups: int = 1):
        super().__init__()
        from mamba_ssm.modules.mamba3 import Mamba3
        self.mamba3 = Mamba3(
            d_model=dim, d_state=d_state, expand=expand,
            headdim=headdim, is_mimo=False, chunk_size=chunk_size,
            ngroups=ngroups,
        )
```

- [ ] **Step 3: Pass ngroups through Block and GPT**

In `Block.__init__`, add `mamba3_ngroups` parameter and forward to `Mamba3Layer`.
In `GPT.__init__`, read from args and pass through.

- [ ] **Step 4: Wire rope_fraction into AttentionLayer**

In `AttentionLayer.__init__`, apply partial RoPE using `rope_fraction`:

```python
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, rope_fraction: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = max(2, int(self.head_dim * rope_fraction) // 2 * 2)  # must be even
        kv_dim = num_kv_heads * self.head_dim
        # ... rest unchanged ...
        self.rotary = Rotary(self.rope_dims, base=rope_base)
```

And in the forward, apply RoPE only to the first `rope_dims` dimensions of q and k.

- [ ] **Step 5: Commit**

```bash
git add train_nemotron_hybrid.py
git commit -m "feat: add Nemotron-H config options (ngroups, rope_fraction)"
```

---

### Task 4: Update Modal script for the new training script

**Files:**
- Modify: `modal_train.py`

Add a `train_nemotron` function that runs our new script with various configs.

- [ ] **Step 1: Add train_nemotron_hybrid.py to the hybrid_image**

In `modal_train.py`, update the `hybrid_image` to also include our new script:

```python
    .add_local_file("train_nemotron_hybrid.py", "/app/train_nemotron_hybrid.py")
```

- [ ] **Step 2: Add nemotron smoke test function**

```python
@app.function(
    image=hybrid_image,
    gpu="H100",
    timeout=1800,
    volumes={"/vol": data_vol},
)
def train_nemotron_smoke():
    """1xH100, 200 steps, Nemotron-H inspired Mamba-3 hybrid"""
    _ensure_data("sp1024", train_shards=1)
    return _run_training(1, {
        "RUN_ID": "nemotron_smoke",
        "ITERATIONS": "200",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
    }, script="train_nemotron_hybrid.py")
```

- [ ] **Step 3: Add nemotron medium test function**

```python
@app.function(
    image=hybrid_image,
    gpu="H100",
    timeout=3600,
    volumes={"/vol": data_vol},
)
def train_nemotron_medium():
    """1xH100, 2000 steps, Nemotron-H Mamba-3 hybrid"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "nemotron_medium",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
    }, script="train_nemotron_hybrid.py")
```

- [ ] **Step 4: Update local_entrypoint to include new modes**

```python
@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode == "smoke":
        result = train_smoke.remote()
    elif mode == "medium":
        result = train_medium.remote()
    elif mode == "full":
        result = train_full.remote()
    elif mode == "hybrid":
        result = train_hybrid_smoke.remote()
    elif mode == "nemotron":
        result = train_nemotron_smoke.remote()
    elif mode == "nemotron-medium":
        result = train_nemotron_medium.remote()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use smoke/medium/full/hybrid/nemotron/nemotron-medium")
```

- [ ] **Step 5: Commit**

```bash
git add modal_train.py
git commit -m "feat: add nemotron smoke and medium modes to Modal"
```

---

### Task 5: Run smoke test to validate the script works

**Files:** None (execution only)

- [ ] **Step 1: Run nemotron smoke test on Modal**

```bash
cd "/Users/yongkangzou/Desktop/Hackathons/Parameter Golf"
modal run modal_train.py --mode nemotron
```

Expected: Training completes 200 steps, prints `val_bpb`, step time should be ~115-120ms/step (with torch.compile fullgraph=False), similar to PR #1355.

- [ ] **Step 2: Verify key metrics**

Check output for:
- `hybrid_arch` or `attn_indices` printed correctly
- `model_params` close to ~27M (PR #1355 baseline)
- `step_avg` around 115-130ms (not 900ms+)
- `val_bpb` printed at end
- No OOM errors

- [ ] **Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: resolve smoke test issues"
```

---

### Task 6: Run Nemotron-H ablation experiments

**Files:** None (execution only, record results)

Run three ablations comparing Nemotron-H ideas vs PR #1355 defaults. Use medium mode (2000 steps) for meaningful signal.

- [ ] **Step 1: Baseline — replicate PR #1355 defaults**

```bash
# 8 layers, 1 attn (even), d_state=64, ngroups=1
modal run modal_train.py --mode nemotron-medium
```

Record: train_loss at step 2000, val_bpb, step_avg

- [ ] **Step 2: Ablation A — 2 attention layers (Nemotron-H ratio)**

```bash
# Modify modal_train.py nemotron_medium to use NUM_ATTN_LAYERS=2
```

Record and compare.

- [ ] **Step 3: Ablation B — d_state=128 (Nemotron-H 8B config)**

```bash
# Modify to use MAMBA3_D_STATE=128
```

Record and compare. Watch for OOM.

- [ ] **Step 4: Ablation C — attention in first layers (opposite of Nemotron-H)**

```bash
# Modify to use ATTN_PLACEMENT=first, NUM_ATTN_LAYERS=2
```

PR #1013 found attention is better in low layers at dim=512. Test this.

- [ ] **Step 5: Document results in strategy doc**

Update `.claude/docs/competition-strategy.md` with ablation results table.

- [ ] **Step 6: Commit**

```bash
git add .claude/docs/competition-strategy.md
git commit -m "docs: add Nemotron-H ablation results"
```

---

### Task 7: Commit final version and push

**Files:**
- All modified files

- [ ] **Step 1: Final commit and push**

```bash
git add train_nemotron_hybrid.py modal_train.py reference_pr1355.py
git push origin main
```

- [ ] **Step 2: Update memory with results**

Update `.claude/projects/.../memory/project_hybrid_mamba.md` with final experiment results and next steps.

---
