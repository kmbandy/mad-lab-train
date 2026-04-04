#!/usr/bin/env python3
"""Merge LoRA adapters into base model for MOEification.

Run before mergekit-moe:
  python3 merge_adapters.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-1.7B"
MODELS_DIR   = Path("/mnt/hdd/models")
ADAPTERS_DIR = Path("/mnt/mainpc/models")

for domain in ["technical", "sentiment"]:
    adapter_path = ADAPTERS_DIR / f"qwen3-1.7b-{domain}-lora"
    output_path  = MODELS_DIR / f"qwen3-1.7b-{domain}-merged"

    if output_path.exists():
        print(f"Skipping {domain} — {output_path} already exists")
        continue

    print(f"Merging {domain}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(str(output_path))

    del model
    print(f"Saved to {output_path}")

print("Done. Run mergekit-moe next.")
