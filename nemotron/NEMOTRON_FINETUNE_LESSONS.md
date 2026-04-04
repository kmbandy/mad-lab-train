# Nemotron-4B Fine-Tune: Lessons Learned

**Date:** 2026-03-28
**Model:** nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
**Dataset:** kurtisbandy/quant-stack-finetune-data (Kaggle)
**Goal:** Domain-adapted stock analyst for quant strategy bot

---

## What We Were Trying to Do

Fine-tune Nemotron-4B on financial Q&A data to serve as the senior analyst brain in a
multi-tier trading architecture. The fine-tuned model replaces the base GGUF on the GTX 1070
and runs via llama.cpp as the Tier-2 decision engine.

---

## The Broken First Run (~24 hours wasted)

### What went wrong

LoRA `target_modules` only covered attention and FFN layers:

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

**Nemotron-H is a Mamba/SSM hybrid.** Only ~8 of 32 layers are attention — the rest are
Mamba SSM layers (`in_proj`, `x_proj`, `dt_proj`, `out_proj`). Targeting only attention/FFN
meant ~75% of the model's compute was frozen and untouched by training.

### Symptoms
- Training loss plateau'd almost immediately (never descended meaningfully)
- Eval accuracy collapsed to near-zero
- 24 hours of A10G time with essentially nothing learned

### Root cause
Never assume a model's architecture from its name or size. Nemotron-H is documented as a
"hybrid SSM-Transformer" — the SSM layers dominate and must be included in LoRA targets.

---

## The Fixed Run

### LoRA target_modules (quality fix)

```python
target_modules=[
    # Attention layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    # FFN layers
    "gate_proj", "up_proj", "down_proj",
    # SSM/Mamba layers — THE MISSING PIECE
    "in_proj", "x_proj", "dt_proj", "out_proj",
]
```

### Training config (efficiency fixes)

| Parameter | Broken | Fixed | Reason |
|-----------|--------|-------|--------|
| `per_device_train_batch_size` | 2 | 8 | A10G 24GB has headroom |
| `per_device_eval_batch_size` | 2 | 8 | same |
| `gradient_accumulation_steps` | 4 | 2 | effective batch stays 16 |
| `learning_rate` | 2e-4 | 1e-4 | SSM layers more sensitive |
| `packing` | False | True | ~1.7x fewer steps |
| `eval_steps` / `save_steps` | 200 | 500 | ~11 evals over ~5800 steps |

### Expected runtime improvement
~33 hours → ~5.5 hours on A10G (G5 spot)

---

## Infrastructure Notes

### EC2 Setup
- **Instance:** G5.xlarge (A10G, 24GB VRAM) spot
- **AMI:** AWS Deep Learning AMI with PyTorch (Python 3.13 at `/opt/pytorch/`)
- **Key:** `~/.ssh/mad-lab-key.pem`
- **CRITICAL:** Always run with `/opt/pytorch/bin/python3`, NOT `/usr/bin/python3`
  - System Python 3.12 has no pip in this AMI
  - The PyTorch env has everything pre-installed

### Launch command
```bash
scp -i ~/.ssh/mad-lab-key.pem ~/kaggle-finetune/ec2_train.py ubuntu@<ip>:~/ec2_train.py
ssh -i ~/.ssh/mad-lab-key.pem ubuntu@<ip>
KAGGLE_USERNAME=kurtisbandy KAGGLE_KEY=<key> nohup /opt/pytorch/bin/python3 ec2_train.py > train.log 2>&1 &
tail -f train.log
```

### Retrieval
```bash
# Grab the adapter before shutdown (script auto-shuts the instance when done)
scp -i ~/.ssh/mad-lab-key.pem -r ubuntu@<ip>:~/nemotron-4b-bf16-lora-r32 ./
# Or set S3_BUCKET env var — script uploads automatically
```

---

## Nemotron-H Architecture Quirks

1. **Mostly Mamba, not attention** — ~75% of layers are SSM. KV cache is tiny (~257MB at 32k ctx).
2. **mamba-ssm required** — must install from git (PyPI version has `bare_metal_version` bug with CUDA 13). The ec2_train.py script handles this.
3. **No gradient checkpointing** — `gradient_checkpointing=False` required; SSM doesn't support it natively.
4. **BF16 only** (no 4-bit quantization) — mamba-ssm CUDA kernels incompatible with bitsandbytes 4-bit. BF16 = ~8GB on A10G, plenty of room.
5. **transformers patches required** — transformers 5.x has bugs in `configuration_nemotron_h.py` and `modeling_nemotron_h.py`. The ec2_train.py script patches both automatically.
6. **chunk_size** — cap at 256 for best SSM quality on A10G.

---

## General SSM/Hybrid LoRA Rule

**Any time you fine-tune a hybrid SSM model, always include SSM layer projections in target_modules:**

```python
# SSM layers (Mamba, DeltaNet, etc.)
"in_proj", "x_proj", "dt_proj", "out_proj"
```

Without these, you're fine-tuning the decorative attention layers while the actual
computation stays frozen. The model will appear to train (loss ticks down slightly) but
won't generalize to the domain.

---

## Output

- **Adapter location:** `~/nemotron-4b-bf16-lora-r32/` (on EC2) or S3
- **training_summary.json** written at end with full hyperparams
- **Next step:** Merge adapter + quantize to Q8_0 GGUF → deploy to `~/models/` on mad-lab → update `llama-server-quant.service`
