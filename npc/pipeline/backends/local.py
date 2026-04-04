"""Local GPU training backend (CUDA / ROCm / CPU)."""

from __future__ import annotations

import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

from .base import TrainingBackend

# Torch 2.4.x compat patch — set_submodule added in 2.5.0
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.rsplit(".", 1)
        if len(parts) == 1:
            self.add_module(target, module)
        else:
            parent = self.get_submodule(parts[0])
            parent.add_module(parts[1], module)
    nn.Module.set_submodule = _set_submodule

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


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _to_messages(sample: dict, role_map: dict) -> dict:
    messages = [
        {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in sample["conversations"]
        if turn["from"] in role_map
    ]
    return {"messages": messages}


def _format_sample(sample: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}


def _sft_args(ft_cfg, hw, output_dir: str, sequence_len: int):
    from trl import SFTConfig
    if ft_cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", ft_cfg.wandb_project or "mad-lab-train")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=ft_cfg.num_epochs,
        per_device_train_batch_size=ft_cfg.micro_batch_size,
        per_device_eval_batch_size=ft_cfg.micro_batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        lr_scheduler_type=ft_cfg.lr_scheduler_type,
        warmup_steps=ft_cfg.warmup_steps,
        weight_decay=ft_cfg.weight_decay,
        max_grad_norm=ft_cfg.max_grad_norm,
        gradient_checkpointing=ft_cfg.gradient_checkpointing,
        fp16=hw.fp16,
        bf16=hw.bf16,
        logging_steps=ft_cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=ft_cfg.eval_steps,
        save_strategy="steps",
        save_steps=ft_cfg.save_steps,
        save_total_limit=ft_cfg.save_total_limit,
        load_best_model_at_end=ft_cfg.load_best_model_at_end,
        metric_for_best_model=ft_cfg.metric_for_best_model,
        report_to=ft_cfg.report_to,
        dataloader_num_workers=ft_cfg.dataloader_num_workers,
        seed=ft_cfg.seed,
        max_length=sequence_len,
        dataset_text_field="text",
        packing=ft_cfg.packing,
    )


def _lora_config(ft_cfg):
    from peft import LoraConfig
    return LoraConfig(
        r=ft_cfg.lora_r,
        lora_alpha=ft_cfg.lora_alpha,
        lora_dropout=ft_cfg.lora_dropout,
        target_modules=ft_cfg.lora_target_modules,
        bias=ft_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )


class LocalBackend(TrainingBackend):
    """Train on the local GPU (CUDA / ROCm / CPU)."""

    def run(self, train_data: str, eval_data: str) -> None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from hardware import detect as detect_hardware
        from generate import Theme
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        from trl import SFTTrainer

        ft_cfg   = self.ft_cfg
        theme    = Theme(self.theme_dir)

        out_fmt  = theme.cfg.get("output", {})
        role_map = {
            out_fmt.get("system_role",    "system"): "system",
            out_fmt.get("user_role",      "human"):  "user",
            out_fmt.get("assistant_role", "gpt"):    "assistant",
        }
        chat_template = CHAT_TEMPLATES.get(
            out_fmt.get("chat_template", "chatml"), CHAT_TEMPLATES["chatml"]
        )

        output_dir   = ft_cfg.output_dir
        sequence_len = ft_cfg.sequence_len
        model_path   = ft_cfg.base_model

        hw = detect_hardware(verbose=True)

        # Tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.chat_template = chat_template
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Dataset
        print("Loading dataset...")
        train_raw = _load_jsonl(train_data)
        eval_raw  = _load_jsonl(eval_data)
        train_fmt = [_format_sample(_to_messages(s, role_map), tokenizer) for s in train_raw]
        eval_fmt  = [_format_sample(_to_messages(s, role_map), tokenizer) for s in eval_raw]
        train_dataset = Dataset.from_list(train_fmt)
        eval_dataset  = Dataset.from_list(eval_fmt)
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Eval:  {len(eval_dataset)} samples")

        # Model
        if not hw.supports_4bit:
            print("Warning: bitsandbytes not available — loading in full precision")
        print("\nLoading model...")
        compute_dtype = torch.float16 if hw.compute_dtype == "float16" else torch.bfloat16
        if hw.supports_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config,
                device_map=hw.device, dtype=compute_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=hw.device, dtype=compute_dtype,
            )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=ft_cfg.gradient_checkpointing
        )

        lora_cfg      = _lora_config(ft_cfg)
        training_args = _sft_args(ft_cfg, hw, output_dir, sequence_len)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=lora_cfg,
        )

        # Cast bf16 LoRA params to fp32 (AMP scaler requirement)
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
