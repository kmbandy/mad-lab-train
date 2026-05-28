"""torchrun entry point for multi-GPU pretraining with DeepSpeed ZeRO-2.

Launched by PretrainExecutor._run_multi_gpu() via:
    torchrun --nproc_per_node N -m pipeline.executors._pretrain_worker --config /path/to/cfg.json

Emits JSONL events to stdout; the parent executor reads and forwards them to SSE.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def _emit(event: str, data: dict) -> None:
    print(json.dumps({"event": event, "data": data}), flush=True)


def _deepspeed_config(zero_stage: int, bf16: bool, fp16: bool) -> dict:
    return {
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": fp16, "auto_cast": False},
        "zero_force_ds_cpu_optimizer": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerFast,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    out_dir = Path(cfg["_out_dir"])
    run_datasets_dir = Path(cfg["_run_datasets_dir"])
    train_cfg = cfg.get("training", {})
    zero_stage = int(cfg.get("deepspeed_zero_stage", 2))

    # ── Precision ──────────────────────────────────────────────────────────────
    # In multi-GPU mode, BF16 is strongly preferred; fall back to FP16
    bf16 = bool(train_cfg.get("bf16", torch.cuda.is_bf16_supported()))
    fp16 = bool(train_cfg.get("fp16", not bf16))

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer_model = cfg.get("tokenizer_model")
    tokenizer_save = out_dir / "tokenizer"

    if tokenizer_model:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    elif tokenizer_save.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_save))
    else:
        # Tokenizer must be pre-trained in single-GPU warm-up pass; worker shouldn't train it
        _emit("error", {"message": "tokenizer_model required for multi-GPU pretrain"})
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model from architecture ────────────────────────────────────────────────
    arch_path = cfg["architecture"]
    path = Path(arch_path)
    if path.suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            arch_dict = yaml.safe_load(f)
    else:
        with open(path) as f:
            arch_dict = json.load(f)

    model_type = arch_dict.pop("model_type", "llama")
    if model_type == "mlambaformer":
        import mlambaformer  # noqa: F401
    arch_dict.setdefault("vocab_size", len(tokenizer))
    model_config = AutoConfig.for_model(model_type, **arch_dict)
    model = AutoModelForCausalLM.from_config(model_config)

    param_count = sum(p.numel() for p in model.parameters())
    _emit("corpus_loaded", {"param_count": param_count})

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_path = run_datasets_dir / "train.jsonl"
    if not train_path.exists():
        train_path = run_datasets_dir / "training.jsonl"
    eval_path = run_datasets_dir / "eval.jsonl"
    has_eval = eval_path.exists()

    data_files: dict = {"train": str(train_path)}
    if has_eval:
        data_files["validation"] = str(eval_path)
    raw = load_dataset("json", data_files=data_files)
    train_ds = raw["train"]
    eval_ds = raw.get("validation")

    def _format(examples: dict) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            if isinstance(msgs, list)
            else str(msgs)
            for msgs in examples["messages"]
        ]

    # ── Callback (rank-0 only to avoid duplicate events) ──────────────────────
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    class _WorkerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not is_main or not logs:
                return
            loss = logs.get("loss") or logs.get("train_loss")
            if loss is not None:
                _emit("step", {
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    "epoch": int(state.epoch or 0),
                    "total_epochs": args.num_train_epochs,
                    "loss": float(loss),
                    "lr": float(logs.get("learning_rate", 0.0)),
                    "grad_norm": float(logs.get("grad_norm", 0.0)),
                })

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not is_main or not metrics:
                return
            eval_loss = metrics.get("eval_loss")
            try:
                ppl = round(math.exp(float(eval_loss)), 4) if eval_loss else 0.0
            except (OverflowError, ValueError):
                ppl = float("inf")
            _emit("eval", {
                "step": state.global_step,
                "eval_loss": float(eval_loss) if eval_loss is not None else 0.0,
                "perplexity": ppl,
            })

        def on_epoch_end(self, args, state, control, **kwargs):
            if not is_main:
                return
            history = state.log_history or []
            train_loss = next((h["loss"] for h in reversed(history) if "loss" in h), 0.0)
            eval_loss = next((h["eval_loss"] for h in reversed(history) if "eval_loss" in h), 0.0)
            _emit("epoch_end", {
                "epoch": int(state.epoch or 0),
                "train_loss": float(train_loss),
                "eval_loss": float(eval_loss),
            })

        def on_save(self, args, state, control, **kwargs):
            if not is_main:
                return
            _emit("checkpoint", {
                "checkpoint_id": str(state.global_step),
                "sequence": state.global_step,
                "metadata": {"epoch": state.epoch, "step": state.global_step},
            })

    # ── SFTConfig with DeepSpeed ───────────────────────────────────────────────
    ds_config = _deepspeed_config(zero_stage, bf16, fp16)

    data_format = cfg.get("data_format", "messages")
    common = dict(
        output_dir=str(out_dir),
        num_train_epochs=int(train_cfg.get("epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=int(train_cfg.get("warmup_steps", 20)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        bf16=bf16,
        fp16=fp16,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(train_cfg.get("eval_steps", 100)) if has_eval else None,
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        report_to="none",
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        deepspeed=ds_config,
    )
    if data_format == "text":
        # raw-text CLM pretraining: pack the `text` field, no chat template
        sft_config = SFTConfig(**common, dataset_text_field="text", packing=True)
        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer, callbacks=[_WorkerCallback()],
        )
    else:
        sft_config = SFTConfig(**common, dataset_text_field=None)
        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=train_ds, eval_dataset=eval_ds,
            formatting_func=_format, processing_class=tokenizer,
            callbacks=[_WorkerCallback()],
        )

    from pipeline.executors.finetune import _find_latest_checkpoint
    resume_from = _find_latest_checkpoint(out_dir)
    trainer.train(resume_from_checkpoint=resume_from)

    if is_main:
        trainer.save_model(str(out_dir))


if __name__ == "__main__":
    main()
