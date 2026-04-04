#!/usr/bin/env python3
"""
QLoRA fine-tune of SmolLM3-3B-Base for D&D NPC roleplay.

Usage:
    ~/axolotl-env/bin/python3 train.py [--dry-run]

GTX 1070 8GB — fp16, 4-bit quantization, gradient checkpointing.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

# torch 2.4.x is missing nn.Module.set_submodule (added in 2.5.0).
# transformers 5.x bitsandbytes integration calls it — patch it in.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.rsplit(".", 1)
        if len(parts) == 1:
            self.add_module(target, module)
        else:
            parent = self.get_submodule(parts[0])
            parent.add_module(parts[1], module)
    nn.Module.set_submodule = _set_submodule

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH   = "/home/kmbandy/models/SmolLM3-3B-Base"
TRAIN_DATA   = "/home/kmbandy/mad-lab-dnd/training/data/train.jsonl"
EVAL_DATA    = "/home/kmbandy/mad-lab-dnd/training/data/eval.jsonl"
OUTPUT_DIR   = "/home/kmbandy/mad-lab-dnd/training/output/smollm3-npc"

SEQUENCE_LEN = 2048
BATCH_SIZE   = 1
GRAD_ACCUM   = 16   # effective batch = 16
EPOCHS       = 3
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# ---------------------------------------------------------------------------
# Chat template  (ChatML)
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(sample: dict) -> dict:
    """Convert ShareGPT from/value format to role/content messages."""
    messages = [
        {"role": ROLE_MAP[turn["from"]], "content": turn["value"]}
        for turn in sample["conversations"]
        if turn["from"] in ROLE_MAP
    ]
    return {"messages": messages}


def format_sample(sample: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model + tokenize 10 samples, then exit")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Dataset ----
    print("Loading dataset...")
    train_raw = load_jsonl(TRAIN_DATA)
    eval_raw  = load_jsonl(EVAL_DATA)

    train_msgs = [to_messages(s) for s in train_raw]
    eval_msgs  = [to_messages(s) for s in eval_raw]

    train_fmt = [format_sample(s, tokenizer) for s in train_msgs]
    eval_fmt  = [format_sample(s, tokenizer) for s in eval_msgs]

    train_dataset = Dataset.from_list(train_fmt)
    eval_dataset  = Dataset.from_list(eval_fmt)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval:  {len(eval_dataset)} samples")
    print(f"  Sample preview:\n{train_fmt[0]['text'][:400]}\n...")

    if args.dry_run:
        # Tokenize a few samples to catch template / token issues
        for i, s in enumerate(train_fmt[:10]):
            ids = tokenizer(s["text"], truncation=True, max_length=SEQUENCE_LEN)
            print(f"  [{i}] tokens: {len(ids['input_ids'])}")
        print("\nDry run complete — tokenizer OK.")
        return

    # ---- Model (4-bit QLoRA) ----
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16, not bf16
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Training config (SFTConfig = TrainingArguments + SFT-specific params) ----
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
        # SFT-specific
        max_length=SEQUENCE_LEN,
        dataset_text_field="text",
        packing=False,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.model.print_trainable_parameters()

    # LoRA params may be initialized as bfloat16 (from model config).
    # AMP grad scaler requires fp32 trainable params — cast any bf16 up to fp32.
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    print("\nStarting training...")
    trainer.train()

    print("\nSaving final adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
