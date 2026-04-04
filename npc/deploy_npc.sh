#!/bin/bash
# Deploy SmolLM3-3B NPC fine-tune → GGUF (Q8_0) → npc-1/npc-2
# Run from: ~/mad-lab-dnd/training/
# Usage: bash deploy_npc.sh

set -e

ADAPTER_DIR="$HOME/mad-lab-dnd/training/output/smollm3-npc"
MERGED_DIR="$HOME/mad-lab-dnd/training/output/smollm3-npc-merged"
GGUF_OUT="$HOME/models/smollm3-npc-q8_0.gguf"
BASE_MODEL="$HOME/models/SmolLM3-3B-Base"
LLAMA_CPP="$HOME/llama.cpp"
VENV="$HOME/axolotl-env"

echo "=== Step 1: Merge LoRA adapter into base model ==="
$VENV/bin/python3 - <<EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import gc

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "$BASE_MODEL",
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base, "$ADAPTER_DIR")

print("Merging...")
merged = model.merge_and_unload()

# Free base + peft wrapper before saving to avoid holding two full copies
del base, model
gc.collect()

print("Saving merged model (sharded)...")
merged.save_pretrained("$MERGED_DIR", max_shard_size="1GB", safe_serialization=True)
del merged
gc.collect()

print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained("$ADAPTER_DIR")
tok.save_pretrained("$MERGED_DIR")

print("Merge done → $MERGED_DIR")
EOF

echo ""
echo "=== Step 2: Convert to GGUF (Q4_K_M) ==="
python3 $LLAMA_CPP/convert_hf_to_gguf.py \
    "$MERGED_DIR" \
    --outfile "$GGUF_OUT" \
    --outtype q8_0

echo ""
echo "=== Step 3: Test GGUF loads ==="
$LLAMA_CPP/build/bin/llama-cli \
    --model "$GGUF_OUT" \
    --prompt "[CHARACTER: Elara Dawnwhisper] [SCENE: dimly lit tavern] [MOOD: guarded] [PLAYER ACTION: You ask about the road north]" \
    --n-predict 120 \
    --temp 0.8 \
    -ngl 99 \
    --no-warmup 2>/dev/null

echo ""
echo "=== Done! ==="
echo "GGUF saved to: $GGUF_OUT (~3.2GB, Q8_0 — preserves fine-tune fidelity)"
echo ""
echo "Next: update llama-server-npc-1 and llama-server-npc-2 service files"
echo "to point at $GGUF_OUT, then restart."
