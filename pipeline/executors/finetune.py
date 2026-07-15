"""Finetune executor — QLoRA via TRL SFTTrainer + PEFT. Always single-GPU."""
import asyncio
import math
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class FinetuneExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        mode = cfg.get("mode", "standard")
        base_model = cfg["base_model"]
        gpu_target = cfg.get("gpu_target", "auto")
        is_healing = mode == "healing"

        lora_cfg = cfg.get("lora", {})
        train_cfg = cfg.get("training", {})

        out_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent
            / "datasets"
            / str(self.run_id)
            / "finetune"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        run_datasets_dir = out_dir.parent
        train_path, gen_path, eval_path = _resolve_datasets(cfg, run_datasets_dir)

        await self.emit_event("stage_started", {
            "stage_type": "finetune",
            "mode": mode,
            "base_model": base_model,
        }, stage_type="finetune")

        loop = asyncio.get_event_loop()

        def _emit_sync(event_type: str, data: dict) -> None:
            asyncio.run_coroutine_threadsafe(
                self.emit_event(event_type, data, stage_type="finetune"),
                loop,
            )

        pause_requested = self._pause_requested  # captured ref; callback writes via executor
        executor_ref = self  # for callback to check flags

        def _train() -> None:
            import torch
            from datasets import concatenate_datasets, load_dataset
            from peft import LoraConfig, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainerCallback,
                TrainerControl,
                TrainerState,
                TrainingArguments,
            )
            from trl import SFTConfig, SFTTrainer

            class _PipelineCallback(TrainerCallback):
                def on_log(
                    self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    logs: dict | None = None,
                    **kwargs,
                ) -> None:
                    if not logs:
                        return
                    loss = logs.get("loss") or logs.get("train_loss")
                    if loss is not None:
                        _emit_sync("step", {
                            "step": state.global_step,
                            "total_steps": state.max_steps,
                            "epoch": int(state.epoch or 0),
                            "total_epochs": args.num_train_epochs,
                            "loss": float(loss),
                            "lr": float(logs.get("learning_rate", 0.0)),
                            "grad_norm": float(logs.get("grad_norm", 0.0)),
                        })
                    # Emit VRAM on every log tick
                    try:
                        if torch.cuda.is_available():
                            dev_idx = torch.cuda.current_device()
                            used = torch.cuda.memory_allocated(dev_idx) / 1e9
                            total = torch.cuda.get_device_properties(dev_idx).total_memory / 1e9
                            _emit_sync("vram", {
                                "used_gb": round(used, 2),
                                "total_gb": round(total, 2),
                            })
                    except Exception:
                        pass

                def on_evaluate(
                    self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    metrics: dict | None = None,
                    **kwargs,
                ) -> None:
                    if not metrics:
                        return
                    eval_loss = metrics.get("eval_loss")
                    _emit_sync("eval", {
                        "step": state.global_step,
                        "eval_loss": float(eval_loss) if eval_loss is not None else 0.0,
                        "perplexity": _perplexity(eval_loss),
                    })

                def on_epoch_end(
                    self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    **kwargs,
                ) -> None:
                    history = state.log_history or []
                    train_loss = next(
                        (h["loss"] for h in reversed(history) if "loss" in h), 0.0
                    )
                    eval_loss = next(
                        (h["eval_loss"] for h in reversed(history) if "eval_loss" in h), 0.0
                    )
                    _emit_sync("epoch_end", {
                        "epoch": int(state.epoch or 0),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss),
                    })

                def on_save(
                    self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    **kwargs,
                ) -> None:
                    _emit_sync("checkpoint", {
                        "checkpoint_id": str(state.global_step),
                        "sequence": state.global_step,
                        "metadata": {
                            "epoch": state.epoch,
                            "step": state.global_step,
                        },
                    })
                    asyncio.run_coroutine_threadsafe(
                        executor_ref.record_checkpoint(
                            state.global_step,
                            str(out_dir / f"checkpoint-{state.global_step}"),
                            {"epoch": state.epoch, "step": state.global_step},
                        ),
                        loop,
                    )
                    if executor_ref._pause_requested:
                        control.should_training_stop = True

                def on_step_end(
                    self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    **kwargs,
                ) -> None:
                    if executor_ref._force_pause:
                        control.should_training_stop = True

            # ── Device selection ───────────────────────────────────────────────
            device = _pick_device(gpu_target)
            bf16, fp16 = _precision_flags(device, train_cfg)

            # ── Tokenizer ──────────────────────────────────────────────────────
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ── Model (4-bit QLoRA) ────────────────────────────────────────────
            compute_dtype = torch.bfloat16 if bf16 else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
            )

            # ── LoRA adapter config ────────────────────────────────────────────
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", "all-linear"),
                bias="none",
            )

            # ── SFTConfig ──────────────────────────────────────────────────────
            has_eval = eval_path is not None and eval_path.exists()
            sft_config = SFTConfig(
                output_dir=str(out_dir),
                num_train_epochs=int(train_cfg.get("epochs", 1 if is_healing else 3)),
                per_device_train_batch_size=int(train_cfg.get("micro_batch_size", 1)),
                gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
                learning_rate=float(train_cfg.get("learning_rate", 5e-4 if is_healing else 2e-4)),
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
                dataset_text_field=None,
            )

            # ── Datasets ───────────────────────────────────────────────────────
            data_files: dict[str, str | list[str]] = {"train": str(train_path)}
            if gen_path and gen_path.exists():
                data_files["train"] = [str(train_path), str(gen_path)]
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
                    for msgs in examples["messages"]
                ]

            # ── Train ──────────────────────────────────────────────────────────
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=peft_config,
                formatting_func=_format,
                processing_class=tokenizer,
                callbacks=[_PipelineCallback()],
            )

            resume_from = cfg.get("_resume_artifact") or _find_latest_checkpoint(out_dir)
            trainer.train(resume_from_checkpoint=resume_from)
            trainer.save_model(str(out_dir))

        await loop.run_in_executor(None, _train)

        if self._force_pause or self._pause_requested:
            return None

        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_datasets(
    cfg: dict, run_datasets_dir: Path
) -> tuple[Path, Path | None, Path | None]:
    """Return (train_path, gen_path, eval_path). Config explicit paths override auto-wiring."""
    if cfg.get("dataset"):
        train_path = Path(os.path.expanduser(cfg["dataset"]))
    else:
        train_path = run_datasets_dir / "train.jsonl"
        if not train_path.exists():
            train_path = run_datasets_dir / "training.jsonl"

    gen_path = run_datasets_dir / "datagen" / "generated.jsonl"

    if cfg.get("eval_dataset"):
        eval_path = Path(os.path.expanduser(cfg["eval_dataset"]))
    else:
        p = run_datasets_dir / "eval.jsonl"
        eval_path = p if p.exists() else None

    return train_path, gen_path, eval_path


def _pick_device(gpu_target: str) -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        if gpu_target == "auto" or torch.cuda.device_count() == 1:
            return "cuda:0"
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i).lower()
            if gpu_target == "r9700" and "9700" in name:
                return f"cuda:{i}"
            if gpu_target == "6900xt" and "6900" in name:
                return f"cuda:{i}"
        return "cuda:0"
    except Exception:
        return "cpu"


def _supports_bf16(device: str) -> bool:
    try:
        import torch
        if device == "cpu":
            return False
        idx = int(device.split(":")[-1]) if ":" in device else 0
        with torch.cuda.device(idx):
            return torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _precision_flags(device: str, train_cfg: dict) -> tuple[bool, bool]:
    """Return (bf16, fp16). Explicit config overrides auto-detection."""
    if "bf16" in train_cfg:
        bf16 = bool(train_cfg["bf16"])
    else:
        bf16 = _supports_bf16(device)
    if "fp16" in train_cfg:
        fp16 = bool(train_cfg["fp16"])
    else:
        fp16 = (not bf16) and (device != "cpu")
    return bf16, fp16


def _find_latest_checkpoint(out_dir: Path) -> str | None:
    checkpoints = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def _perplexity(loss: float | None) -> float:
    if loss is None:
        return 0.0
    try:
        return round(math.exp(float(loss)), 4)
    except (OverflowError, ValueError):
        return float("inf")
