#!/usr/bin/env python3
"""
Generalized QLoRA fine-tuner.

Reads training hyperparams from a finetune config yaml.
Auto-detects hardware (CUDA/ROCm) via pipeline/hardware.py.
Dataset format and chat template are specified in the theme.

Usage:
    ~/axolotl-env/bin/python3 pipeline/train.py \
        --config run.yaml \
        --theme themes/dnd_npc \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
import torch.nn as nn
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from hardware import detect as detect_hardware
from generate import Theme

# ---------------------------------------------------------------------------
# Torch 2.4.x compat patch — set_submodule added in 2.5.0
# transformers 5.x bitsandbytes integration calls it
# ---------------------------------------------------------------------------
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.rsplit(".", 1)
        if len(parts) == 1:
            self.add_module(target, module)
        else:
            parent = self.get_submodule(parts[0])
            parent.add_module(parts[1], module)
    nn.Module.set_submodule = _set_submodule


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

CHAT_TEMPLATES = {
    "chatml": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
    "llama3": (
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    ),
    "mistral": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>"
        "{% endif %}{% endfor %}"
    ),
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(sample: dict, role_map: dict) -> dict:
    """Convert ShareGPT from/value format to role/content messages."""
    messages = [
        {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in sample["conversations"]
        if turn["from"] in role_map
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="Run config yaml (paths, endpoints)")
    parser.add_argument("--theme",   required=True, help="Path to theme directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tokenize samples and exit without loading model")
    args = parser.parse_args()

    # ---- Load configs ----
    theme_dir = Path(args.theme)
    if not theme_dir.is_absolute():
        theme_dir = Path(__file__).parent.parent / args.theme
    theme = Theme(theme_dir)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / args.config
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)

    # Fine-tune config: prefer theme-local finetune.yaml, fall back to run config
    ft_path = theme_dir / "finetune.yaml"
    if ft_path.exists():
        with open(ft_path) as f:
            ft_cfg = yaml.safe_load(f)
    else:
        ft_cfg = run_cfg.get("finetune", {})

    # Dispatch to Kaggle backend if requested
    if ft_cfg.get("training_env") == "kaggle":
        sys.path.insert(0, str(Path(__file__).parent))
        from kaggle_train import run as kaggle_run
        train_data = ft_cfg.get("train_data") or str(Path(run_cfg["output_dir"]) / "train.jsonl")
        eval_data  = ft_cfg.get("eval_data")  or str(Path(run_cfg["output_dir"]) / "eval.jsonl")
        kaggle_run(ft_cfg, run_cfg, train_data, eval_data, theme_dir)
        return

    # Resolve paths
    model_path  = ft_cfg.get("base_model") or run_cfg.get("base_model")
    train_data  = ft_cfg.get("train_data")  or run_cfg.get("train_data",
                    str(Path(run_cfg["output_dir"]) / "train.jsonl"))
    eval_data   = ft_cfg.get("eval_data")   or run_cfg.get("eval_data",
                    str(Path(run_cfg["output_dir"]) / "eval.jsonl"))
    output_dir  = ft_cfg.get("output_dir")  or str(Path(run_cfg["output_dir"]) / "adapter")

    if not model_path:
        print("Error: base_model not set in finetune config or run config")
        sys.exit(1)

    # Training hyperparams (with sensible defaults)
    sequence_len = ft_cfg.get("sequence_len", 2048)
    batch_size   = ft_cfg.get("micro_batch_size", 1)
    grad_accum   = ft_cfg.get("gradient_accumulation_steps", 16)
    epochs       = ft_cfg.get("num_epochs", 3)
    lr           = ft_cfg.get("learning_rate", 2e-4)
    lora_r       = ft_cfg.get("lora_r", 16)
    lora_alpha   = ft_cfg.get("lora_alpha", 32)
    lora_dropout = ft_cfg.get("lora_dropout", 0.05)
    warmup_steps = ft_cfg.get("warmup_steps", 20)

    # Dataset format from theme
    out_fmt  = theme.cfg.get("output", {})
    role_map = {
        out_fmt.get("system_role",    "system"):    "system",
        out_fmt.get("user_role",      "human"):     "user",
        out_fmt.get("assistant_role", "gpt"):       "assistant",
    }
    chat_template_name = out_fmt.get("chat_template", "chatml")
    chat_template = CHAT_TEMPLATES.get(chat_template_name, CHAT_TEMPLATES["chatml"])

    # ---- Hardware ----
    hw = detect_hardware(verbose=True)

    # ---- Tokenizer ----
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Dataset ----
    print("Loading dataset...")
    from datasets import Dataset

    train_raw = load_jsonl(train_data)
    eval_raw  = load_jsonl(eval_data)

    train_fmt = [format_sample(to_messages(s, role_map), tokenizer) for s in train_raw]
    eval_fmt  = [format_sample(to_messages(s, role_map), tokenizer) for s in eval_raw]

    train_dataset = Dataset.from_list(train_fmt)
    eval_dataset  = Dataset.from_list(eval_fmt)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval:  {len(eval_dataset)} samples")
    print(f"  Sample preview:\n{train_fmt[0]['text'][:300]}\n...")

    if args.dry_run:
        for i, s in enumerate(train_fmt[:10]):
            ids = tokenizer(s["text"], truncation=True, max_length=sequence_len)
            print(f"  [{i}] tokens: {len(ids['input_ids'])}")
        print("\nDry run complete — tokenizer OK.")
        return

    # ---- Model (4-bit QLoRA) ----
    if not hw.supports_4bit:
        print("Warning: bitsandbytes not available — loading in full precision (slow/high VRAM)")

    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training

    compute_dtype = torch.float16 if hw.compute_dtype == "float16" else torch.bfloat16

    if hw.supports_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=hw.device,
            dtype=compute_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=hw.device,
            dtype=compute_dtype,
        )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Training config ----
    from trl import SFTTrainer, SFTConfig

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        fp16=hw.fp16,
        bf16=hw.bf16,
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
        max_length=sequence_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Cast any bf16 LoRA params to fp32 (AMP scaler requires fp32 trainable params)
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    trainer.model.print_trainable_parameters()
    print("\nStarting training...")
    trainer.train()

    print("\nSaving adapter...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
