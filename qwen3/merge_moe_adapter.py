#!/usr/bin/env python3
"""Merge the MOE alignment LoRA adapter into the base MOE model.

Run after moe_finetune.py completes:
  python3 merge_moe_adapter.py

Then quantize:
  ~/llama.cpp/build-rocm/bin/llama-quantize \
    /mnt/hdd/models/qwen3-1.7b-moe-merged/model.safetensors \
    /mnt/hdd/models/Qwen3-1.7B-MOE-Q8_0.gguf Q8_0
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL   = "/mnt/hdd/models/qwen3-1.7b-moe"
ADAPTER_PATH = "/mnt/hdd/models/qwen3-1.7b-moe-lora"
OUTPUT_PATH  = "/mnt/hdd/models/qwen3-1.7b-moe-merged"

if Path(OUTPUT_PATH).exists():
    print(f"Already exists: {OUTPUT_PATH}")
else:
    print("Loading MOE model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print("Applying adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.save_pretrained(OUTPUT_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

print("Done. Next: quantize with llama-quantize.")
