#!/usr/bin/env python3
"""QLoRA fine-tune for Qwen3-1.7B on GTX 1070 (8GB CUDA).

Run twice — once per domain:
  python3 qwen3_finetune.py --domain technical   # RSI/MACD/EMA signals
  python3 qwen3_finetune.py --domain sentiment    # news/filing signals

Each run:
  - QLoRA 4-bit (bitsandbytes NF4 + double quant)
  - LoRA r=32, alpha=64 on attention + MLP layers
  - ~1.5–2h on GTX 1070 (1.2-1.5 t/s)
  - Outputs adapter to models/qwen3-1.7b-{domain}-lora/

MOEification (run after both fine-tunes complete):
  mergekit-moe  models/qwen3-1.7b-technical-lora  \\
                models/qwen3-1.7b-sentiment-lora   \\
                --config moe_config.yaml --out models/qwen3-1.7b-moe
"""

import argparse
import sys
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

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-1.7B"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR   = Path(__file__).parent / "data"

LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

# Qwen3 attention + MLP projection names
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_CFG = {
    "num_train_epochs":              3,
    "per_device_train_batch_size":   8,
    "gradient_accumulation_steps":   2,   # effective batch = 16
    "learning_rate":                 2e-4,
    "warmup_ratio":                  0.05,
    "lr_scheduler_type":             "cosine",
    "logging_steps":                 25,
    "eval_strategy":                 "steps",
    "eval_steps":                    200,
    "save_steps":                    200,
    "save_total_limit":              2,
    "bf16":                          True,   # L4 = Ada Lovelace, BF16 native
    "fp16":                          False,
    "optim":                         "adamw_bnb_8bit",
    "gradient_checkpointing":        True,
    "dataloader_num_workers":        2,
    "report_to":                     "none",
    "max_seq_length":                448,
    "packing":                       False,
}


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-1.7B")
    p.add_argument("--domain", choices=["technical", "sentiment"], required=True,
                   help="Which expert to train: technical or sentiment")
    p.add_argument("--base-model", default=BASE_MODEL,
                   help="HF model ID or local path (default: Qwen/Qwen3-1.7B)")
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["num_train_epochs"],
                   help="Training epochs (default: 3)")
    p.add_argument("--lr", type=float, default=TRAIN_CFG["learning_rate"],
                   help=f"Learning rate (default: {TRAIN_CFG['learning_rate']})")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint if available")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
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
        print("Install: pip install transformers peft trl bitsandbytes datasets accelerate")
        sys.exit(1)

    domain = args.domain
    train_file = DATA_DIR / f"{domain}_train.jsonl"
    eval_file  = DATA_DIR / f"{domain}_eval.jsonl"
    output_dir = MODELS_DIR / f"qwen3-1.7b-{domain}-lora"

    if not train_file.exists():
        print(f"ERROR: {train_file} not found — run generate_{domain}_data.py first")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain:     {domain}")
    print(f"Base model: {args.base_model}")
    print(f"Train data: {train_file}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── 4-bit quantisation ─────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    max_seq = TRAIN_CFG["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq

    # ── LoRA ───────────────────────────────────────────────────────────────────
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

    # ── Dataset ────────────────────────────────────────────────────────────────
    def _format(example):
        """Convert conversations list → ChatML string."""
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

    data_files = {"train": str(train_file)}
    if eval_file.exists():
        data_files["eval"] = str(eval_file)

    raw = load_dataset("json", data_files=data_files)
    train_ds = raw["train"].map(_format, remove_columns=raw["train"].column_names)
    eval_ds  = raw["eval"].map(_format, remove_columns=raw["eval"].column_names) \
               if "eval" in raw else None

    # ── Training args ─────────────────────────────────────────────────────────
    cfg = dict(TRAIN_CFG)
    cfg["num_train_epochs"] = args.epochs
    cfg["learning_rate"] = args.lr
    max_seq = cfg.pop("max_seq_length")
    packing = cfg.pop("packing")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        **cfg,
        packing=packing,
        eval_packing=False,
        dataset_text_field="text",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds else None,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Resume from checkpoint if requested
    resume_from = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"Resuming from {resume_from}")

    print(f"\nStarting {domain} fine-tune...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ──────────────────────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved to {final_path}")
    print(f"\nDone. Next step:")
    if domain == "technical":
        print("  python3 qwen3_finetune.py --domain sentiment")
        print("  Then MOEify both adapters with mergekit.")
    else:
        print("  Both fine-tunes complete — run MOEify step.")
        print("  See moe_config.yaml in this directory.")


if __name__ == "__main__":
    main()
