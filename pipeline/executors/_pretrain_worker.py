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


def iter_packed_domain_rows(raw_ds, tokenizer, seq_len: int):
    """Yield {"input_ids", "labels", "domain_ids"} rows. No `datasets` dependency.

    This is the piece that decides which token gets supervised as which domain, and if it
    is wrong nothing downstream can tell -- the loss curve stays perfectly plausible while
    the gate trains toward a mislabelled target. So it must be testable, and on 2026-08-04
    it was not: the only environments that can import BOTH `datasets` and `mlambaformer`
    are a 40 GB image that is not built on this box, so the test for it could not run
    anywhere. A test that cannot run is not a test.

    Splitting the pure logic out fixes that -- it needs only an object exposing
    `column_names`, `__len__` and iteration, which a plain list-backed stub satisfies.
    `Dataset.from_generator` stays in the thin wrapper below, where there is nothing left
    to get wrong.
    """
    from mlambaformer.packing import pack_domain_documents

    cols = set(raw_ds.column_names)
    missing = {"text", "domain_id"} - cols
    if missing:
        raise ValueError(
            f"data_format='text_domains' needs columns {sorted(missing)} which are absent "
            f"from the dataset (has: {sorted(cols)}). build_corpus.py writes both; a "
            "corpus built before the domain enum, or a JSONL produced by another path, "
            "will not have domain_id. Refusing rather than training an unsupervised gate "
            "while the config claims supervision."
        )

    def _encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; cannot separate packed documents")

    # Dataset.from_generator MATERIALISES an Arrow cache. It is fine for smoke runs and
    # tests and catastrophic for the real corpus: 50B tokens becomes int64 input_ids
    # 0.40 TB + labels 0.40 TB + domain_ids 0.40 TB = 1.2 TB, against 407 GB free on
    # /home and 493 GB on /mnt/hdd. Fail with the arithmetic rather than filling the disk
    # partway through a multi-hour tokenisation.
    _MAX_INLINE_DOCS = 500_000
    if len(raw_ds) > _MAX_INLINE_DOCS:
        raise ValueError(
            f"{len(raw_ds):,} documents is too many to pack inline: Dataset.from_generator "
            "writes an int64 Arrow cache (~24 bytes/token across input_ids, labels and "
            "domain_ids) and will exhaust the disk. Pre-pack instead with "
            "mlambaformer/scripts/pack_corpus.py (uint16 tokens + int8 domains, ~3 "
            "bytes/token) and set `shards_dir` in the run config."
        )

    docs = ((r["text"], int(r["domain_id"])) for r in raw_ds)
    for win in pack_domain_documents(docs, _encode, seq_len, eos_id=eos_id):
        # labels == input_ids: the model shifts internally for CLM. domain_ids is NOT
        # shifted -- domain is a property of the token at position s and the gate reads
        # position s's hidden state (modeling_mlambaformer.py:643-645).
        yield {
            "input_ids": win["input_ids"],
            "labels": list(win["input_ids"]),
            "domain_ids": win["domain_ids"],
        }


def _pack_with_domains(raw_ds, tokenizer, seq_len: int):
    """Materialise iter_packed_domain_rows into a HF Dataset. Thin on purpose."""
    from datasets import Dataset

    return Dataset.from_generator(lambda: iter_packed_domain_rows(raw_ds, tokenizer, seq_len))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    import torch
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerFast,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        default_data_collator,
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
    # MAD-327: MAD-160 runs set cfg["tokenizer_model"] to the pinned 48k slice,
    # i.e. str(mlambaformer.tokenization.get_tokenizer_dir("mad160-48k")) -- the
    # controlled-constant tokenizer shared across all 8 cells and the eval holdout.
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

    from pipeline.executors.pretrain import assert_vocab_matches
    assert_vocab_matches(int(arch_dict["vocab_size"]), len(tokenizer))

    model_config = AutoConfig.for_model(model_type, **arch_dict)
    model = AutoModelForCausalLM.from_config(model_config)

    param_count = sum(p.numel() for p in model.parameters())
    _emit("corpus_loaded", {"param_count": param_count})

    # ── Regional torch.compile (R13) ──────────────────────────────────────────
    # PER-LAYER, not whole-model. This distinction is measured, not stylistic:
    #   whole-model (HF Trainer's torch_compile=true)  0.997x -- and it never actually
    #       compiled; Dynamo skipped the model forward after create_block_mask broke the
    #       graph, so it paid the compile wall and got nothing. That is why the run config
    #       sets torch_compile: false and must keep doing so.
    #   per-layer (this)                                1.143x measured 2026-08-04, from
    #       8,117,090 -> 7,101,345 us/step. It eats 84% of the ELEMENTWISE kernel class.
    # Until now this existed ONLY as a loop inside scripts/bench_train_step.py, so every
    # measured 1.143x lived in the benchmark and no training run ever saw it (R13).
    #
    # FAILS LOUD IF IT CANNOT FIND THE LAYERS. A silently-skipped optimisation is the exact
    # mechanism that kept this gap open for twelve days -- the config asks for it, nothing
    # happens, and the throughput shortfall looks like the model rather than the wiring.
    if bool(train_cfg.get("regional_compile", False)):
        import torch

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError(
                f"regional_compile=true but {model_type!r} exposes no model.model.layers to "
                "compile. Refusing to continue silently: the run asked for the 1.143x and "
                "would otherwise have paid nothing and reported it as achieved."
            )
        for i in range(len(layers)):
            layers[i] = torch.compile(layers[i], dynamic=False)
        _emit("regional_compile", {"layers_compiled": len(layers)})

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
    # D2 guard, at the pipeline end rather than only in the model. A config that asks to
    # supervise the domain gate but selects a data_format carrying no labels would train
    # a frozen gate for the whole run. The model raises on the first step, but catching
    # it here means the failure lands before the GPU is claimed and the checkpoint
    # directory is created, and it names the actual fix.
    _coef = float(getattr(model.config, "moe_domain_gate_loss_coef", 0.0) or 0.0)
    if _coef > 0.0 and data_format != "text_domains":
        raise ValueError(
            f"model config sets moe_domain_gate_loss_coef={_coef} but data_format="
            f"{data_format!r} supplies no per-token domain labels, so the domain-gate "
            "cross-entropy could never fire and the gate would stay a fixed random "
            "projection for the entire run. Use data_format: 'text_domains' (needs a "
            "corpus with a domain_id column), or set moe_domain_gate_loss_coef to 0.0 to "
            "declare an unsupervised gate deliberately."
        )
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
        # R13 (2026-08-06): these four were DECLARED in the run config and read NOWHERE.
        # configs/run/mad160-v1.json has set optim, seed and data_seed since it was written,
        # and none of them reached TrainingArguments -- so the run silently used HF's
        # defaults (adamw_torch, seed 42) while the config of record said otherwise. A
        # config key with no consumer is indistinguishable from a config key that works,
        # right up until you compare the run against what you thought you asked for.
        #   optim      -- the optimizer is a NUMERICS decision, not just a throughput one:
        #                 fp32 Adam m+v is 6.94 GiB at 932.02M params vs 1.74 GiB for a
        #                 low-bit state, and that 5.21 GiB is what OOM'd four benchmarks
        #                 on 2026-08-06. Defaulted to adamw_torch_fused, which is what the
        #                 run config asks for.
        #   seed/data_seed -- a 50B-token run that cannot be reproduced is not a result.
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        seed=int(train_cfg.get("seed", 42)),
        data_seed=int(train_cfg.get("data_seed", train_cfg.get("seed", 42))),
        # Whole-model compile, measured 0.997x and it never actually compiled -- see the
        # regional_compile block above for the mechanism that is worth 1.143x. Wired only so
        # the key is not silently dead; it should stay false.
        torch_compile=bool(train_cfg.get("torch_compile", False)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        deepspeed=ds_config,
    )
    from pipeline.executors._quant_native_callbacks import mlambaformer_quant_callbacks
    quant_cbs = mlambaformer_quant_callbacks(model)
    if data_format == "text_domains":
        # MAD-322 / D2: raw-text CLM pretraining that ALSO supervises the domain gate.
        #
        # We pack here rather than letting TRL do it. TRL's packing=True concatenates
        # documents and keeps only the text column, so a per-DOCUMENT domain_id cannot
        # reach the model as per-TOKEN supervision -- and a packed sequence spans several
        # documents with different domains, so the label genuinely has to be built while
        # packing. Without this the domain gate receives no gradient at all and stays a
        # fixed random projection for the whole run (the D2 defect).
        #
        # MAD-364: domain_ids is a SEPARATE tensor and never enters input_ids. An
        # in-stream domain tag was deliberately cut because under causal attention every
        # token attends to a tag at position 0 and Mamba's state carries it forward, so
        # the gate would learn a trivial lookup and collapse at inference.
        seq_len = int(cfg.get("max_seq_length", 2048))
        shards_dir = cfg.get("shards_dir")
        if shards_dir:
            # PRODUCTION PATH. Pre-packed memmapped shards: uint16 tokens + int8 domain
            # ids in a parallel file. At 50B tokens this is 0.15 TB; routing the same
            # corpus through Dataset.from_generator would write an Arrow cache of int64
            # columns totalling 1.2 TB, against 407 GB free. Pack with
            # mlambaformer/scripts/pack_corpus.py.
            from mlambaformer.data import MlambaformerDataset

            train_ds = MlambaformerDataset(shards_dir, seq_len, require_domains=True)
            eval_shards = cfg.get("eval_shards_dir")
            eval_ds = (
                MlambaformerDataset(eval_shards, seq_len, require_domains=True)
                if eval_shards else None
            )
        else:
            train_ds = _pack_with_domains(train_ds, tokenizer, seq_len)
            eval_ds = _pack_with_domains(eval_ds, tokenizer, seq_len) if eval_ds else None
        sft_config = SFTConfig(
            **common, dataset_text_field=None, packing=False,
            # domain_ids IS in MlambaformerForCausalLM.forward's signature, so HF would
            # normally keep it -- but SFTTrainer does its own column handling and a
            # dropped label here fails SILENTLY (the model raises only if coef > 0, and
            # the CE is simply skipped otherwise). Being explicit is cheaper than
            # discovering it from a flat gate 40 GPU-hours in.
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            callbacks=[_WorkerCallback(), *quant_cbs],
        )
    elif data_format == "text":
        # raw-text CLM pretraining: pack the `text` field, no chat template
        sft_config = SFTConfig(**common, dataset_text_field="text", packing=True)
        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=[_WorkerCallback(), *quant_cbs],
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
    resume_from = cfg.get("_resume_artifact") or _find_latest_checkpoint(out_dir)
    trainer.train(resume_from_checkpoint=resume_from)

    if is_main:
        trainer.save_model(str(out_dir))


if __name__ == "__main__":
    main()
