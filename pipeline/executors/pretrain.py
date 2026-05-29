"""Pretrain executor — full-weight training from scratch.

Single-GPU: runs inline in a thread pool (full precision, no quantization).
Multi-GPU: launches torchrun with DeepSpeed ZeRO-2, reads JSONL events from stdout.
"""
import asyncio
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor
from pipeline.executors.finetune import _perplexity, _pick_device, _precision_flags


class PretrainExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False
        self._worker_proc: asyncio.subprocess.Process | None = None

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        multi_gpu = bool(cfg.get("multi_gpu", False))

        out_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent
            / "datasets"
            / str(self.run_id)
            / "pretrain"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        await self.emit_event("stage_started", {
            "stage_type": "pretrain",
            "multi_gpu": multi_gpu,
            "base_model": cfg.get("architecture", "custom"),
        }, stage_type="pretrain")

        if multi_gpu:
            result = await self._run_multi_gpu(cfg, out_dir)
        else:
            result = await self._run_single_gpu(cfg, out_dir)

        if self._force_pause or self._pause_requested:
            return None

        return result

    async def pause(self) -> None:
        self._pause_requested = True
        if self._worker_proc and self._worker_proc.returncode is None:
            self._worker_proc.terminate()

    async def force_pause(self) -> None:
        self._force_pause = True
        if self._worker_proc and self._worker_proc.returncode is None:
            self._worker_proc.kill()

    # ── Single-GPU ────────────────────────────────────────────────────────────

    async def _run_single_gpu(self, cfg: dict, out_dir: Path) -> str:
        loop = asyncio.get_event_loop()
        executor_ref = self

        def _emit_sync(event_type: str, data: dict) -> None:
            asyncio.run_coroutine_threadsafe(
                executor_ref.emit_event(event_type, data, stage_type="pretrain"),
                loop,
            )

        def _train() -> None:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainerCallback,
                TrainerControl,
                TrainerState,
                TrainingArguments,
            )
            from trl import SFTConfig, SFTTrainer

            gpu_target = cfg.get("gpu_target", "auto")
            device = _pick_device(gpu_target)
            train_cfg = cfg.get("training", {})
            bf16, fp16 = _precision_flags(device, train_cfg)

            # ── Tokenizer ──────────────────────────────────────────────────────
            tokenizer_model = cfg.get("tokenizer_model")
            vocab_size = int(cfg.get("vocab_size", 32000))
            run_datasets_dir = out_dir.parent

            if tokenizer_model:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
            else:
                tokenizer = _train_tokenizer(
                    run_datasets_dir, out_dir / "tokenizer", vocab_size, _emit_sync
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ── Model from architecture ────────────────────────────────────────
            arch_path = cfg["architecture"]
            model_config = _load_architecture(arch_path, len(tokenizer))
            model = AutoModelForCausalLM.from_config(model_config)
            model = model.to(dtype=torch.bfloat16 if bf16 else torch.float16)

            param_count = sum(p.numel() for p in model.parameters())
            _emit_sync("corpus_loaded", {"param_count": param_count})

            # ── Datasets ──────────────────────────────────────────────────────
            from datasets import load_dataset
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

            # ── Callback ──────────────────────────────────────────────────────
            class _PipelineCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
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
                    try:
                        if torch.cuda.is_available():
                            idx = torch.cuda.current_device()
                            used = torch.cuda.memory_allocated(idx) / 1e9
                            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
                            _emit_sync("vram", {"used_gb": round(used, 2), "total_gb": round(total, 2)})
                    except Exception:
                        pass

                def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                    if metrics:
                        eval_loss = metrics.get("eval_loss")
                        _emit_sync("eval", {
                            "step": state.global_step,
                            "eval_loss": float(eval_loss) if eval_loss is not None else 0.0,
                            "perplexity": _perplexity(eval_loss),
                        })

                def on_epoch_end(self, args, state, control, **kwargs):
                    history = state.log_history or []
                    train_loss = next((h["loss"] for h in reversed(history) if "loss" in h), 0.0)
                    eval_loss = next((h["eval_loss"] for h in reversed(history) if "eval_loss" in h), 0.0)
                    _emit_sync("epoch_end", {
                        "epoch": int(state.epoch or 0),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss),
                    })

                def on_save(self, args, state, control, **kwargs):
                    _emit_sync("checkpoint", {
                        "checkpoint_id": str(state.global_step),
                        "sequence": state.global_step,
                        "metadata": {"epoch": state.epoch, "step": state.global_step},
                    })
                    if executor_ref._pause_requested:
                        control.should_training_stop = True

                def on_step_end(self, args, state, control, **kwargs):
                    if executor_ref._force_pause:
                        control.should_training_stop = True

            # ── SFTConfig ──────────────────────────────────────────────────────
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
            )
            from pipeline.executors._quant_native_callbacks import mlambaformer_quant_callbacks
            quant_cbs = mlambaformer_quant_callbacks(model)
            if data_format == "text":
                # raw-text CLM pretraining: pack the `text` field, no chat template
                sft_config = SFTConfig(**common, dataset_text_field="text", packing=True)
                trainer = SFTTrainer(
                    model=model, args=sft_config,
                    train_dataset=train_ds, eval_dataset=eval_ds,
                    processing_class=tokenizer,
                    callbacks=[_PipelineCallback(), *quant_cbs],
                )
            else:
                sft_config = SFTConfig(**common, dataset_text_field=None)
                trainer = SFTTrainer(
                    model=model, args=sft_config,
                    train_dataset=train_ds, eval_dataset=eval_ds,
                    formatting_func=_format, processing_class=tokenizer,
                    callbacks=[_PipelineCallback()],
                )

            from pipeline.executors.finetune import _find_latest_checkpoint
            resume_from = _find_latest_checkpoint(out_dir)
            trainer.train(resume_from_checkpoint=resume_from)
            trainer.save_model(str(out_dir))

        await loop.run_in_executor(None, _train)
        return str(out_dir)

    # ── Multi-GPU (DeepSpeed ZeRO-2 via torchrun) ─────────────────────────────

    async def _run_multi_gpu(self, cfg: dict, out_dir: Path) -> str:
        import torch

        gpu_count = torch.cuda.device_count() if hasattr(torch, "cuda") else 2

        # Write full config to a temp file for the worker to read
        worker_cfg = dict(cfg)
        worker_cfg["_out_dir"] = str(out_dir)
        worker_cfg["_run_datasets_dir"] = str(out_dir.parent)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="pretrain_cfg_"
        ) as f:
            json.dump(worker_cfg, f)
            cfg_path = f.name

        cmd = [
            "torchrun",
            "--nproc_per_node", str(gpu_count),
            "--standalone",
            "-m", "pipeline.executors._pretrain_worker",
            "--config", cfg_path,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._worker_proc = proc

            # Stream stdout events (worker prints JSONL)
            assert proc.stdout is not None
            async for line in proc.stdout:
                if self._force_pause:
                    break
                raw = line.decode().strip()
                if not raw:
                    continue
                try:
                    evt = json.loads(raw)
                    await self.emit_event(
                        evt["event"], evt.get("data", {}), stage_type="pretrain"
                    )
                except Exception:
                    pass

            await proc.wait()
            if proc.returncode not in (0, None) and not self._force_pause and not self._pause_requested:
                stderr = await proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(
                    f"torchrun exited {proc.returncode}: {stderr.decode()[:500]}"
                )
        finally:
            try:
                Path(cfg_path).unlink(missing_ok=True)
            except Exception:
                pass

        return str(out_dir)


# ── Architecture / tokenizer helpers ──────────────────────────────────────────

def _load_architecture(arch_path: str, vocab_size: int):
    """Load a HuggingFace-compatible model config from a JSON/YAML architecture file."""
    from transformers import AutoConfig

    path = Path(os.path.expanduser(arch_path))
    if path.suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            arch_dict = yaml.safe_load(f)
    else:
        with open(path) as f:
            arch_dict = json.load(f)

    model_type = arch_dict.pop("model_type", "llama")
    if model_type == "mlambaformer":
        import mlambaformer  # noqa: F401  registers config + model with HF Auto*
    # Ensure vocab_size is consistent with tokenizer
    arch_dict.setdefault("vocab_size", vocab_size)
    return AutoConfig.for_model(model_type, **arch_dict)


def _train_tokenizer(
    corpus_dir: Path,
    save_dir: Path,
    vocab_size: int,
    emit_sync,
) :
    """Train a BPE tokenizer from corpus text, return HF PreTrainedTokenizerFast."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect text files from corpus
    text_files = []
    for pattern in ("*.jsonl", "*.txt"):
        text_files.extend(corpus_dir.glob(pattern))

    def _text_iter():
        for p in text_files:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msgs = obj.get("messages", [])
                        for m in msgs:
                            yield m.get("content", "")
                    except Exception:
                        yield line

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        _text_iter(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
    )
    bpe.save_model(str(save_dir))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.save_pretrained(str(save_dir))

    emit_sync("tokenizer_trained", {"vocab_size": tokenizer.vocab_size})
    return tokenizer
