#!/usr/bin/env python3
"""Nemotron-4B QLoRA fine-tune — EC2 A10G (24GB VRAM).

Usage:
    KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx \
    S3_BUCKET=your-bucket         # optional — uploads adapter at end
    python3 ec2_train.py

Requires: Python 3.10+, CUDA 12.x, ~20GB disk for model cache.
"""

import subprocess, sys, os, json, importlib
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# nvcc is bundled with the pytorch package — add to PATH for mamba-ssm compilation
os.environ["PATH"] = (
    "/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin:"
    "/opt/pytorch/bin:" +
    os.environ.get("PATH", "")
)

# ── Deps ───────────────────────────────────────────────────────────────────────
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

print("Installing dependencies...")
pip("transformers==5.3.0", "peft>=0.10.0", "trl>=0.8.6",
    "datasets>=2.18.0", "accelerate>=0.29.0", "bitsandbytes>=0.46.1")

# mamba-ssm: install from git (PyPI version has bare_metal_version bug with CUDA 13)
print("Installing SSM kernels (mamba-ssm, causal-conv1d)...")
_ssm_ok = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "git+https://github.com/state-spaces/mamba.git",
     "causal-conv1d"],
    env=os.environ,
).returncode == 0
if not _ssm_ok:
    print("  [warn] mamba-ssm install failed — will use pure-PyTorch fallback")

import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"device: {torch.cuda.get_device_name(0)}, "
      f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ── Dataset download ───────────────────────────────────────────────────────────
DATA_DIR = Path("./data/quant-stack-finetune-data")
if not (DATA_DIR / "quant_train.jsonl").exists():
    print("Downloading dataset from Kaggle...")
    pip("kaggle")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "kurtisbandy/quant-stack-finetune-data",
         "--unzip", "-p", str(DATA_DIR)],
        check=True,
    )
    print(f"Dataset downloaded to {DATA_DIR}")
else:
    print(f"Dataset already present at {DATA_DIR}")

OUTPUT_DIR = Path("./nemotron-4b-bf16-lora-r32")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Nemotron-H patches (transformers source) ───────────────────────────────────
import transformers
_tf_root    = Path(transformers.__file__).parent
_cfg_file   = _tf_root / "models/nemotron_h/configuration_nemotron_h.py"
_model_file = _tf_root / "models/nemotron_h/modeling_nemotron_h.py"

if not _cfg_file.exists():
    print(f"ERROR: nemotron_h not found in transformers {transformers.__version__}")
    sys.exit(1)

def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    src = path.read_text()
    for old, new in replacements:
        if new in src:
            print(f"  [skip] already patched: {old[:50]!r}")
            continue
        if old not in src:
            print(f"  [warn] patch target not found: {old[:60]!r}")
            continue
        src = src.replace(old, new, 1)
        print(f"  [ok]   patched: {old[:50]!r}")
    path.write_text(src)

print("\nPatching configuration_nemotron_h.py...")
_patch_file(_cfg_file, [
    ('"*": "attention",\n',
     '"*": "attention",\n            "-": "mlp",\n'),
    ('valid_types = {"mamba", "attention", "moe"}',
     'valid_types = {"mamba", "attention", "moe", "mlp"}'),
])

print("Patching modeling_nemotron_h.py...")
_patch_file(_model_file, [
    ('"moe": NemotronHMoE,\n}',
     '"moe": NemotronHMoE,\n    "mlp": NemotronHMLP,\n}'),
    ('class NemotronHMLP(nn.Module):\n    def __init__(self, config):',
     'class NemotronHMLP(nn.Module):\n    def __init__(self, config, **kwargs):'),
    ('"moe": None,\n',
     '"moe": None,\n                "mlp": None,\n'),
])

import transformers.models.nemotron_h.configuration_nemotron_h as _cfg_mod
import transformers.models.nemotron_h.modeling_nemotron_h as _model_mod
importlib.reload(_cfg_mod)
importlib.reload(_model_mod)
print("Modules reloaded.\n")

# ── Remote modeling patch (rmsnorm fallback) ───────────────────────────────────
# Safety net: if the HF-hosted modeling_nemotron_h.py hard-fails on ImportError,
# replace it with a pure-PyTorch rmsnorm_fn. With mamba-ssm installed this should
# be a no-op, but keeps things from crashing if there's a CUDA kernel mismatch.
_RMSNORM_TARGET = 'except ImportError:\n    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'
_RMSNORM_FALLBACK = '''except ImportError:
    pass  # mamba_ssm unavailable — pure-PyTorch rmsnorm_fn defined below

import torch as _torch
def rmsnorm_fn(x, weight, bias=None, z=None, residual=None, prenorm=False,
               residual_in_fp32=False, eps=1e-6, is_rms_norm=True, **kwargs):
    orig_dtype = x.dtype
    x = x.float()
    if residual is not None:
        x = x + residual.float()
    stored = x
    if is_rms_norm:
        x = x * _torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    else:
        x = (x - x.mean(-1, keepdim=True)) * _torch.rsqrt(
            x.var(-1, keepdim=True, unbiased=False) + eps)
    x = x.to(orig_dtype) * weight
    if bias is not None:
        x = x + bias
    if prenorm:
        return x, stored.to(orig_dtype)
    return x
'''

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

def _patch_remote_modeling():
    _patched = False
    for _cache_root in [
        Path.home() / ".cache/huggingface/modules/transformers_modules",
        Path.home() / ".cache/huggingface/hub",
    ]:
        for _f in _cache_root.rglob("modeling_nemotron_h.py"):
            _src = _f.read_text()
            if _RMSNORM_TARGET in _src:
                _f.write_text(_src.replace(_RMSNORM_TARGET, _RMSNORM_FALLBACK, 1))
                print(f"  [ok] patched remote code: {_f.parent.name}/{_f.name}")
                _patched = True
    if not _patched:
        print("  [info] remote modeling already patched or mamba-ssm OK")
    for key in list(sys.modules.keys()):
        if "nemotron" in key.lower() or "transformers_modules" in key:
            del sys.modules[key]
    importlib.invalidate_caches()

# ── Tokenizer ──────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_cached_module_file as _get_cached_module_file

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Pre-seeding modules cache with remote modeling file...")
try:
    _get_cached_module_file(MODEL_ID, "modeling_nemotron_h.py", trust_remote_code=True)
    print("  [ok] modeling_nemotron_h.py pre-cached")
except Exception as _e:
    print(f"  [warn] pre-cache failed: {_e}")

print("Patching HuggingFace remote modeling code...")
_patch_remote_modeling()

# ── Model (BF16 — mamba-ssm CUDA kernels incompatible with bitsandbytes 4-bit) ─
# L4 has 24GB; Nemotron-4B in BF16 = ~8GB, plenty of room without quantization.
print("Loading model in BF16 (no quantization)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# A10G has 24GB — use full chunk_size=256 for best SSM quality
if hasattr(model.config, "chunk_size") and model.config.chunk_size > 256:
    print(f"chunk_size: {model.config.chunk_size} → 256")
    model.config.chunk_size = 256
for module in model.modules():
    if hasattr(module, "chunk_size"):
        setattr(module, "chunk_size", 256)

model.config.use_cache = False

# ── LoRA ───────────────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model

model.enable_input_require_grads()

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        # Attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # FFN layers
        "gate_proj", "up_proj", "down_proj",
        # SSM/Mamba layers — core compute of Nemotron-H, previously missing
        "in_proj", "x_proj", "dt_proj", "out_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Log VRAM after model load
_used = torch.cuda.memory_allocated() / 1e9
_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM after model load: {_used:.1f}/{_total:.1f}GB")

# ── Dataset ────────────────────────────────────────────────────────────────────
from datasets import load_dataset

print("\nLoading dataset...")
dataset = load_dataset("json", data_files={
    "train":      str(DATA_DIR / "quant_train.jsonl"),
    "validation": str(DATA_DIR / "quant_eval.jsonl"),
})
print(f"  Train: {len(dataset['train'])} | Eval: {len(dataset['validation'])}")

def format_sample(example):
    convos = example.get("conversations", [])
    messages = []
    for turn in convos:
        role = turn.get("from", "")
        content = turn.get("value", "")
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
    if not messages:
        return {"text": ""}
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)
    return {"text": text}

dataset = dataset.map(format_sample, remove_columns=dataset["train"].column_names)

# ── Training ───────────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer

print("\nStarting training...")
sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=4,        # SSM activations dominate — no grad checkpointing support
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,         # effective batch=16
    learning_rate=1e-4,                    # halved — SSM layers more sensitive than attention
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    fp16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=False,          # Nemotron-H SSM doesn't support it natively
    max_length=1024,
    packing=True,                          # pack multiple samples per 1024-token chunk — ~1.7x fewer steps
    dataset_text_field="text",
    dataloader_num_workers=4,
    optim="adamw_bnb_8bit",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

import traceback
try:
    trainer.train()
except Exception as e:
    traceback.print_exc()
    print("\nTraining interrupted — saving current checkpoint...")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

summary = {
    "model_id": MODEL_ID,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_targets": "attention+ffn+ssm",
    "quantization": "bf16",
    "max_length": 1024,
    "packing": True,
    "chunk_size": 256,
    "num_epochs": 2,
    "batch_size": 8,
    "grad_accum": 2,
    "effective_batch": 16,
    "learning_rate": 1e-4,
    "train_samples": len(dataset["train"]),
    "eval_samples":  len(dataset["validation"]),
}
(OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
print("Done!\n" + json.dumps(summary, indent=2))

# ── Optional S3 upload ─────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET", "")
if S3_BUCKET:
    print(f"\nUploading adapter to s3://{S3_BUCKET}/nemotron-4b-quant-lora/...")
    subprocess.run(
        ["aws", "s3", "sync", str(OUTPUT_DIR),
         f"s3://{S3_BUCKET}/nemotron-4b-quant-lora/", "--no-progress"],
        check=True,
    )
    print("Upload complete.")
else:
    print("\nNo S3_BUCKET set — adapter saved locally. Download with:")
    print(f"  scp -r ubuntu@<ip>:{OUTPUT_DIR.resolve()} ./nemotron-adapter/")

# Shut down the instance when done to stop billing
print("\nShutting down instance...")
subprocess.run(["sudo", "shutdown", "-h", "now"])
