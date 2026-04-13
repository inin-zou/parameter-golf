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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "tqdm", "torch", "huggingface-hub", "kernels",
        "setuptools", "typing-extensions==4.15.0", "datasets",
        "tiktoken", "sentencepiece",
    )
    .add_local_file("train_gpt.py", "/app/train_gpt.py")
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


def _run_training(gpus: int, env_overrides: dict):
    """Run train_gpt.py with torchrun."""
    import os
    import subprocess

    env = {**os.environ}
    env.update({
        "DATA_PATH": "/app/data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "/app/data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    })
    env.update(env_overrides)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}",
        "/app/train_gpt.py",
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {env_overrides}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print("STDERR (last 3000 chars):", result.stderr[-3000:])
    return result.stdout


@app.function(
    image=image,
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
    image=image,
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
    image=image,
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


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode == "smoke":
        result = train_smoke.remote()
    elif mode == "medium":
        result = train_medium.remote()
    elif mode == "full":
        result = train_full.remote()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use smoke/medium/full")
