"""
Modal script for Parameter Golf training.
Usage:
  # Smoke test (200 steps, 1xH100, ~1 min, ~$0.07)
  modal run modal_train.py --mode smoke

  # Medium experiment (2000 steps, 1xH100, ~12 min, ~$0.80)
  modal run modal_train.py --mode medium

  # Full baseline (10 min wallclock, 8xH100, ~$5.30)
  modal run modal_train.py --mode full
"""
import modal

app = modal.App("parameter-golf")

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "tqdm", "torch", "huggingface-hub", "kernels",
        "setuptools", "typing-extensions==4.15.0", "datasets",
        "tiktoken", "sentencepiece",
    )
    .add_local_file("train_gpt.py", "/app/train_gpt.py")
    .add_local_dir("data", "/app/data", ignore=["datasets", "tokenizers"])
)

# Hybrid image: PyTorch 2.9 + Triton 3.5 (same as PR #1355 RunPod environment)
# Pre-built wheels for mamba-ssm + causal-conv1d, then copy Mamba-3 files from mamba3-release
_conv1d_whl = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
_mamba_whl = "https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
hybrid_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.*", "numpy", "tqdm", "huggingface-hub", "kernels",
        "setuptools", "typing-extensions==4.15.0", "datasets",
        "tiktoken", "sentencepiece", "einops",
    )
    .pip_install(_conv1d_whl, _mamba_whl)
    .run_commands(
        # Copy Mamba-3 modules from mamba3-release (not in v2.3.1 wheel)
        "git clone --depth 1 --branch mamba3-release https://github.com/state-spaces/mamba.git /tmp/mamba3src",
        "PKG=$(python -c 'import mamba_ssm,os; print(os.path.dirname(mamba_ssm.__file__))') && "
        "cp /tmp/mamba3src/mamba_ssm/modules/mamba3.py $PKG/modules/ && "
        "cp -r /tmp/mamba3src/mamba_ssm/ops/triton/mamba3 $PKG/ops/triton/ && "
        "cp /tmp/mamba3src/mamba_ssm/ops/triton/angle_cumsum.py $PKG/ops/triton/ && "
        "cp -r /tmp/mamba3src/mamba_ssm/ops/cute $PKG/ops/ 2>/dev/null || true && "
        "cp -r /tmp/mamba3src/mamba_ssm/ops/tilelang $PKG/ops/ 2>/dev/null || true && "
        "ls $PKG/modules/mamba3.py && echo 'mamba3 files OK' && "
        "rm -rf /tmp/mamba3src",
    )
    .add_local_file("train_gpt_hybrid.py", "/app/train_gpt_hybrid.py")
    .add_local_file("train_nemotron_hybrid.py", "/app/train_nemotron_hybrid.py")
    .add_local_dir("data", "/app/data", ignore=["datasets", "tokenizers"])
)

# Persistent volume to cache downloaded datasets across runs
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


def _ensure_data(variant: str, train_shards: int):
    """Download dataset. Uses /vol as persistent cache, symlinks into /app/data/."""
    import os
    import subprocess

    vol_ds = f"/vol/datasets/fineweb10B_{variant}"
    vol_tok = "/vol/tokenizers"
    app_ds_parent = "/app/data/datasets"
    app_tok = "/app/data/tokenizers"

    if os.path.isdir(vol_ds) and len(os.listdir(vol_ds)) > 1 and os.path.isdir(vol_tok):
        print(f"Dataset cached in volume at {vol_ds}")
    else:
        print(f"Downloading {variant} with {train_shards} shards...")
        subprocess.run(
            ["python3", "/app/data/cached_challenge_fineweb.py",
             "--variant", variant, "--train-shards", str(train_shards)],
            check=True,
        )
        os.makedirs(vol_ds, exist_ok=True)
        os.makedirs(vol_tok, exist_ok=True)
        subprocess.run(["cp", "-rn", f"/app/data/datasets/fineweb10B_{variant}/.", vol_ds], check=True)
        subprocess.run(["cp", "-rn", "/app/data/tokenizers/.", vol_tok], check=True)
        print("Cached to volume.")

    # Symlink from volume into expected paths
    os.makedirs(app_ds_parent, exist_ok=True)
    app_ds = f"{app_ds_parent}/fineweb10B_{variant}"
    if not os.path.exists(app_ds):
        os.symlink(vol_ds, app_ds)
    if not os.path.exists(app_tok):
        os.symlink(vol_tok, app_tok)


def _run_training(gpus: int, env_overrides: dict, script: str = "train_gpt.py"):
    """Run a training script with torchrun. Streams output in real-time."""
    import os
    import subprocess
    import sys

    env = {**os.environ}
    env.update({
        "DATA_PATH": "/app/data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "/app/data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "PYTHONUNBUFFERED": "1",  # force unbuffered output
    })
    env.update(env_overrides)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}",
        f"/app/{script}",
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {env_overrides}", flush=True)
    # Stream stdout/stderr in real-time instead of buffering
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
    return f"exit_code={result.returncode}"


@app.function(
    image=base_image,
    gpu="H100",
    timeout=1800,
    volumes={"/vol": data_vol},
)
def train_smoke():
    """1xH100, 200 steps, ~1 min, ~$0.07"""
    _ensure_data("sp1024", train_shards=1)
    return _run_training(1, {
        "RUN_ID": "modal_smoke",
        "ITERATIONS": "200",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
    })


@app.function(
    image=base_image,
    gpu="H100",
    timeout=3600,
    volumes={"/vol": data_vol},
)
def train_medium():
    """1xH100, 2000 steps, ~12 min, ~$0.80"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "modal_medium",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
    })


@app.function(
    image=base_image,
    gpu="H100:8",
    timeout=3600,
    volumes={"/vol": data_vol},
)
def train_full():
    """8xH100, 10 min wallclock, competition conditions, ~$5.30"""
    _ensure_data("sp1024", train_shards=80)
    return _run_training(8, {
        "RUN_ID": "modal_full_baseline",
        "MAX_WALLCLOCK_SECONDS": "600",
        "VAL_LOSS_EVERY": "200",
    })


@app.function(
    image=hybrid_image,
    gpu="H100",
    timeout=1800,
    volumes={"/vol": data_vol},
)
def train_hybrid_smoke():
    """1xH100, 200 steps, hybrid Mamba-Transformer, ~$0.07
    9 layers: layers 0,1,2,3,5,6,7 are Mamba, layers 4,8 are Attention
    (~78% Mamba, ~22% Attention, inspired by Nemotron-H's ~92/8 ratio)
    """
    _ensure_data("sp1024", train_shards=1)
    return _run_training(1, {
        "RUN_ID": "modal_hybrid_smoke",
        "ITERATIONS": "200",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "MAMBA_LAYERS": "0,1,2,3,5,6,7",
    }, script="train_gpt_hybrid.py")


@app.function(
    image=hybrid_image,
    gpu="H100",
    timeout=3600,
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
        "EVAL_STRIDE": "0",       # disable sliding eval (too slow for smoke test)
        "TTT_ENABLED": "0",       # disable TTT
        "USE_GPTQ": "0",          # skip GPTQ (saves ~5 min)
        "SWEEP_MODE": "1",        # skip all post-training (quant, serialize, eval)
    }, script="train_nemotron_hybrid.py")


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
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",        # skip GPTQ/serialize for faster iteration
    }, script="train_nemotron_hybrid.py")


@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_recur_block():
    """Ablation A: depth recurrence on entire blocks (layers 2,3)"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "recur_block",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "2,3",
        "RECUR_MODE": "block",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_recur_untie():
    """Ablation B: depth recurrence with untied MLPs (layers 2,3)"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "recur_untie_mlp",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "2,3",
        "RECUR_MODE": "untie_mlp",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_recur_deep():
    """Ablation C: depth recurrence on 3 layers (2,3,4) — more aggressive"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "recur_deep",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "2,3,4",
        "RECUR_MODE": "block",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


# --- Hinge point ablations ---

@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_hinge_dual_attn():
    """Hinge test 1: 2 Attention layers at hinge (layers 3,4), recur 2,3,4"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "hinge_dual_attn",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "2",
        "ATTN_PLACEMENT": "3,4",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "2,3,4",
        "RECUR_MODE": "block",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_hinge_multi_recur():
    """Hinge test 3: recur hinge (3,4) twice = 12 virtual layers"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "hinge_multi_recur",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "3,4",
        "RECUR_MODE": "block",
        "RECUR_REPEATS": "2",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


@app.function(image=hybrid_image, gpu="H100", timeout=3600, volumes={"/vol": data_vol})
def train_hinge_4layer():
    """Hinge test bonus: 4-layer recurrence (2,3,4,5)"""
    _ensure_data("sp1024", train_shards=10)
    return _run_training(1, {
        "RUN_ID": "hinge_4layer",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_LOSS_EVERY": "500",
        "MAX_WALLCLOCK_SECONDS": "0",
        "NUM_LAYERS": "8",
        "NUM_ATTN_LAYERS": "1",
        "ATTN_PLACEMENT": "even",
        "MAMBA3_D_STATE": "64",
        "RECUR_LAYERS": "2,3,4,5",
        "RECUR_MODE": "block",
        "RECUR_START_FRAC": "0.35",
        "EVAL_STRIDE": "0",
        "TTT_ENABLED": "0",
        "SWEEP_MODE": "1",
    }, script="train_nemotron_hybrid.py")


@app.function(
    image=hybrid_image,
    gpu="H100",  # need H100 for Triton kernels (sm_90)
    timeout=120,
)
def test_mamba3_import():
    """Quick import test — validates the image has Mamba-3 modules."""
    import os
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}")

    # Test 1: mamba_ssm base + list files
    import mamba_ssm
    mod_dir = os.path.dirname(mamba_ssm.__file__)
    modules = os.listdir(os.path.join(mod_dir, "modules"))
    print(f"mamba_ssm: {mamba_ssm.__version__}")
    print(f"modules/: {sorted(modules)}")
    has_mamba3 = "mamba3.py" in modules
    print(f"mamba3.py present: {has_mamba3}")

    if has_mamba3:
        triton_contents = os.listdir(os.path.join(mod_dir, "ops", "triton"))
        print(f"ops/triton/: {sorted(triton_contents)}")

        # Test 2: Mamba3 import and forward pass
        from mamba_ssm.modules.mamba3 import Mamba3
        m3 = Mamba3(d_model=64, d_state=16, headdim=16, is_mimo=False, chunk_size=16).cuda().bfloat16()
        x = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16)
        y3 = m3(x)
        print(f"Mamba3 forward OK: {x.shape} -> {y3.shape}")
    else:
        print("FAIL: mamba3.py not found!")

    return "ALL TESTS PASSED" if has_mamba3 else "FAIL: mamba3 not installed"


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
    elif mode == "test":
        result = test_mamba3_import.remote()
        print(result)
    elif mode == "recur-ablation":
        # Run all 3 recurrence ablations in parallel!
        print("Launching 3 recurrence ablations in parallel...")
        h1 = train_recur_block.spawn()
        h2 = train_recur_untie.spawn()
        h3 = train_recur_deep.spawn()
        r1 = h1.get()
        print("=== Ablation A (block recurrence) done ===")
        r2 = h2.get()
        print("=== Ablation B (untie MLP) done ===")
        r3 = h3.get()
        print("=== Ablation C (3-layer deep) done ===")
    elif mode == "hinge-ablation":
        # Run 3 hinge point experiments in parallel
        print("Launching 3 hinge ablations in parallel...")
        h1 = train_hinge_dual_attn.spawn()
        h2 = train_hinge_multi_recur.spawn()
        h3 = train_hinge_4layer.spawn()
        r1 = h1.get()
        print("=== Hinge 1 (dual attn) done ===")
        r2 = h2.get()
        print("=== Hinge 3 (multi recur) done ===")
        r3 = h3.get()
        print("=== Hinge bonus (4-layer) done ===")
    elif mode == "recur-block":
        result = train_recur_block.remote()
    elif mode == "recur-untie":
        result = train_recur_untie.remote()
    elif mode == "recur-deep":
        result = train_recur_deep.remote()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use smoke/medium/full/hybrid/nemotron/nemotron-medium/test/recur-ablation/recur-block/recur-untie/recur-deep")
