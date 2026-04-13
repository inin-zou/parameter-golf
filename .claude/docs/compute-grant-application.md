# OpenAI Compute Grant Application

## Form Settings

- **Country of residence**: France
- **Current role**: Industry
- **Level**: Development grant (~$500 / ~160 compute hours)

## Brief description of your approach (max 1,500 chars)

Hybrid Mamba-Transformer architecture inspired by NVIDIA's Nemotron-H. Instead of pure Transformer layers, I replace ~78% of attention layers with Mamba-2 SSM blocks while keeping 2 attention layers for long-range dependencies. This is the first SSM-based submission to the challenge (listed on the wishlist as unclaimed).

Early results are promising: at 200 steps on 1xH100, the hybrid model (7 Mamba + 2 Attention layers) achieves val_bpb=1.85 vs the pure Transformer baseline's val_bpb=2.02 — a 0.17 bpb improvement at the same step count. The hybrid model compresses to 9.7MB (well under 16MB), with 22M params vs baseline's 17M.

Current bottleneck: Mamba's custom CUDA kernels break torch.compile(fullgraph=True), causing 3x slower step times (973ms vs 333ms). My next step is implementing a pure-PyTorch Mamba layer compatible with torch.compile to recover training speed. If step time matches the baseline, the hybrid architecture should train significantly more steps within 10 minutes and achieve a much lower final bpb.

I also plan to combine this with techniques from the current SOTA stack: GPTQ int6 quantization, depth recurrence on Mamba layers, and score-first TTT for evaluation. Need compute to run full 8xH100 experiments and iterate on architecture/quantization.

## What have you tried so far? (max 255 chars)

Built hybrid Mamba-Transformer on Modal 1xH100. 200-step smoke test: val_bpb=1.85 vs baseline 2.02. Identified torch.compile bottleneck (3x slower). Next: pure-PyTorch Mamba for compile compat + 8xH100 full runs.

## Link(s) to your PR submission

https://github.com/inin-zou/parameter-golf
