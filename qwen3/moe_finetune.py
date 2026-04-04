#!/usr/bin/env python3
"""QLoRA fine-tune for the Qwen3-1.7B MOE model.

Aligns the router after mergekit-moe MOEification.
Run on EC2 g6.xlarge (L4 24GB) after ec2_setup.sh.

Usage:
  python3 moe_finetune.py

Output: models/qwen3-1.7b-moe-lora/
"""

import sys
from pathlib import Path
import torch

BASE_MODEL = str(Path(__file__).parent / "models" / "qwen3-1.7b-moe")
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

# Attention projections + MoE expert FFN layers (gate_proj matches all experts)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",  # matches all experts
]

TRAIN_CFG = {
    "num_train_epochs":              1,
    "per_device_train_batch_size":   8,
    "gradient_accumulation_steps":   2,   # effective batch = 16
    "learning_rate":                 5e-5,
    "warmup_ratio":                  0.05,
    "lr_scheduler_type":             "cosine",
    "eval_strategy":                 "steps",
    "logging_steps":                 25,
    "eval_steps":                    200,
    "save_steps":                    200,
    "save_total_limit":              2,
    "bf16":                          True,
    "fp16":                          False,
    "optim":                         "adamw_bnb_8bit",
    "gradient_checkpointing":        True,
    "dataloader_num_workers":        2,
    "report_to":                     "none",
    "max_seq_length":                448,
    "packing":                       False,
}


def main():
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    train_file = DATA_DIR / "moe_train.jsonl"
    eval_file  = DATA_DIR / "moe_eval.jsonl"
    output_dir = MODELS_DIR / "qwen3-1.7b-moe-lora"

    MODELS_DIR.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq = TRAIN_CFG["max_seq_length"]

    print(f"Base model: {BASE_MODEL}")
    print(f"Train data: {train_file} ({sum(1 for _ in open(train_file))} samples)")
    print(f"Device:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading MOE model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def _format(example):
        convs = example["conversations"]
        parts = []
        for turn in convs:
            role = turn["from"]
            content = turn["value"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "human":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "gpt":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        return {"text": "\n".join(parts)}

    raw = load_dataset("json", data_files={
        "train": str(train_file),
        "eval":  str(eval_file),
    })
    train_ds = raw["train"].map(_format, remove_columns=raw["train"].column_names)
    eval_ds  = raw["eval"].map(_format, remove_columns=raw["eval"].column_names)

    cfg = dict(TRAIN_CFG)
    cfg.pop("max_seq_length")   # already applied via tokenizer.model_max_length
    packing = cfg.pop("packing")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        **cfg,
        packing=packing,
        eval_packing=False,
        dataset_text_field="text",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("\nStarting MOE router alignment fine-tune...")
    trainer.train()

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved to {final_path}")
    print("Next: merge adapter into MOE model, then quantize to GGUF Q8_0")


if __name__ == "__main__":
    main()
