# Parameter Golf Submission Guide

## Overview

All submissions are made via Pull Request to [openai/parameter-golf](https://github.com/openai/parameter-golf). Each PR adds a single new folder to the appropriate `records/` subfolder.

## Tracks

- **SOTA Record**: `records/track_10min_16mb/` — must beat current best
- **Non-record / Creative**: `records/track_non_record_16mb/` — interesting approaches that don't beat SOTA
- **Unlimited Compute**: also under `track_non_record_16mb/`, note it in your README

## Folder Naming

```
records/track_10min_16mb/YYYY-MM-DD_ShortMethodName/
```

## Required Files

Every submission folder must contain exactly these 4 files:

### 1. `README.md`
Explain your approach, configuration, key metrics, and training command.

### 2. `submission.json`
```json
{
  "author": "Your Name",
  "github_id": "your-github-username",
  "name": "Method Name",
  "blurb": "One-line description of your approach",
  "date": "YYYY-MM-DDTHH:MM:SSZ",
  "val_loss": 2.0727,
  "val_bpb": 1.2244,
  "bytes_total": 15863489,
  "bytes_code": 47642
}
```

### 3. `train_gpt.py`
Your training script. Must compile and run successfully from within the records folder.

### 4. `train.log`
The training log produced by your script.

## SOTA Record Requirements

1. Beat existing SOTA by at least **0.005 nats**
2. Statistical significance at **p < 0.01** (typically 3 runs averaged)
3. Reproducibly runs in **under 10 minutes on 8xH100 SXM**
4. Total artifact size (code + compressed model) < **16,000,000 bytes** (decimal, not MiB)
5. No network calls or training data access during evaluation — artifact must be self-contained
6. If you changed the tokenizer, prove val_bpb is correctly calculated

## Evaluation Rules

- Evaluation gets an additional 10 minutes (separate from training)
- Any sequence length is allowed for evaluation
- You CANNOT access validation data during training
- Test-time training is allowed only on validation tokens you've already evaluated on
- No "paid prefix" — you can't compress validation data into your 16MB artifact

## Submission Workflow

```bash
# 1. Create a branch
git checkout -b my-submission

# 2. Create your submission folder
mkdir -p records/track_10min_16mb/YYYY-MM-DD_MyMethod

# 3. Add required files (README.md, submission.json, train_gpt.py, train.log)

# 4. Commit and push
git add records/track_10min_16mb/YYYY-MM-DD_MyMethod/
git commit -m "Add submission: MyMethod"
git push origin my-submission

# 5. Open PR to openai/parameter-golf main branch
```

## Key Metrics to Report

From the training log, include:
- `val_loss` and `val_bpb` (post-quantization roundtrip)
- Training time and step count
- Serialized model size (int8+zlib bytes)
- Code size (bytes)
- Total submission size
- Peak GPU memory

## External Packages

You can import any package (e.g., FlashAttention) as long as it doesn't violate rules. Include a `requirements.txt` in your records folder if needed. Library bytes don't count toward 16MB, but you can't sneak in extra compute via custom libraries.

## Non-record Submissions

Accepted for unique/creative approaches even if they don't beat SOTA. Justify your ideas and results in detail. Same file format requirements apply.

## Support

- Discord: [OpenAI Discord](https://discord.com/invite/openai) — channels `#parameter-golf-discussions` and `#parameter-golf-announcements`
- Compute grants: [Request form](https://openai.com/index/parameter-golf/#credit-form) ($1M total pool)
- Deadline: **April 30, 2026**
