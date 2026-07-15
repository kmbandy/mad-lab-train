"""Evaluation executor — benchmark model quality across benchmark types.

Benchmarks:
  perplexity  — token-level cross-entropy on calibration set (fully implemented)
  mmlu        — multiple-choice subject accuracy via HF datasets (fully implemented)
  tool_use    — structured tool call accuracy (stub v1)
  conversation — LLM-as-judge coherence/helpfulness (stub v1)
  coding      — pass@1 on executable test code (stub v1)

Model input: SafeTensors directory, GGUF file (via llama-cpp-python), or HF repo.
Checkpoint per completed benchmark — resumes where it left off.
"""
import asyncio
import json
import math
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


def bits_per_byte(total_loss_nats: float, total_tokens: int, total_bytes: int) -> float:
    """Tokenizer-independent compression metric. total_loss_nats is the SUM of
    per-token cross-entropy in nats over the eval set; total_bytes is the UTF-8 byte
    length of the scored text. Comparable across vocab sizes (unlike per-token PPL)."""
    if total_bytes <= 0:
        return 0.0
    return (total_loss_nats / math.log(2)) / total_bytes


class EvalExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        benchmarks = cfg.get("benchmarks") or []
        if not benchmarks:
            raise ValueError("eval stage requires at least one benchmark config")

        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        out_dir = run_datasets_dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_model_path(cfg, run_datasets_dir)

        checkpoint = _load_checkpoint(out_dir)
        completed: set[str] = set(checkpoint.get("completed_benchmarks", []))
        all_results: dict = checkpoint.get("results", {})

        loop = asyncio.get_event_loop()
        executor_ref = self

        def _emit_sync(event_type: str, data: dict) -> None:
            asyncio.run_coroutine_threadsafe(
                executor_ref.emit_event(event_type, data, stage_type="eval"),
                loop,
            )

        for bm_cfg in benchmarks:
            if self._force_pause or self._pause_requested:
                break

            bm_type = bm_cfg["type"]
            bm_key = f"{bm_type}:{_bm_key(bm_cfg)}"

            if bm_key in completed:
                continue

            await self.emit_event("benchmark_started", {"name": bm_type}, stage_type="eval")

            try:
                score = await loop.run_in_executor(
                    None,
                    lambda bm=bm_cfg: _run_benchmark(
                        bm, model_path, run_datasets_dir, _emit_sync
                    ),
                )
            except NotImplementedError as e:
                await self.emit_event("benchmark_skipped", {
                    "name": bm_type, "reason": str(e),
                }, stage_type="eval")
                score = None

            if score is not None:
                metric = _benchmark_metric(bm_type)
                await self.emit_event("benchmark_complete", {
                    "name": bm_type,
                    "score": score,
                    "metric": metric,
                }, stage_type="eval")
                all_results[bm_type] = {"score": score, "metric": metric}

                # Eval gate check
                threshold = bm_cfg.get("threshold")
                if threshold is not None:
                    passed = score >= float(threshold)
                    await self.emit_event("gate_result", {
                        "passed": passed,
                        "score": score,
                        "threshold": threshold,
                    }, stage_type="eval")

                    if not passed:
                        on_fail = bm_cfg.get("on_fail", "pause")
                        if on_fail == "abort":
                            raise RuntimeError(
                                f"Eval gate failed for {bm_type}: "
                                f"score {score:.4f} < threshold {threshold}"
                            )
                        self._pause_requested = True

            completed.add(bm_key)
            checkpoint["completed_benchmarks"] = list(completed)
            checkpoint["results"] = all_results
            _save_checkpoint(out_dir, checkpoint)

        # Write results JSON
        results_path = out_dir / "results.json"
        results_path.write_text(json.dumps(all_results, indent=2))

        if self._force_pause or self._pause_requested:
            return None

        return str(results_path)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True


# ── Benchmark dispatch ────────────────────────────────────────────────────────

def _run_benchmark(bm_cfg: dict, model_path: Path, run_datasets_dir: Path, emit_sync) -> float:
    bm_type = bm_cfg["type"]
    if bm_type == "perplexity":
        return _bench_perplexity(bm_cfg, model_path, run_datasets_dir, emit_sync)
    elif bm_type == "mmlu":
        return _bench_mmlu(bm_cfg, model_path, emit_sync)
    elif bm_type in ("tool_use", "conversation", "coding"):
        raise NotImplementedError(
            f"{bm_type} benchmark is stubbed in v1 — execution not implemented yet"
        )
    else:
        raise ValueError(f"Unknown benchmark type: {bm_type}")


def _bench_perplexity(
    bm_cfg: dict, model_path: Path, run_datasets_dir: Path, emit_sync
) -> float:
    """Compute mean perplexity on calibration set. Score = 1 / (1 + perplexity)."""
    import torch

    dataset_path = (
        Path(os.path.expanduser(bm_cfg["dataset"]))
        if bm_cfg.get("dataset")
        else run_datasets_dir / "calibration.jsonl"
    )
    if not dataset_path.exists():
        dataset_path = run_datasets_dir / "eval.jsonl"

    max_samples = int(bm_cfg.get("max_samples", 200))

    model, tokenizer = _load_safetensors_model(model_path)
    model.eval()

    total_loss = 0.0
    count = 0
    total_loss_nats = 0.0   # MAD-327: summed CE (nats) for bits-per-byte
    total_bytes = 0         # UTF-8 bytes of the scored (post-truncation) text
    total_scored = 0        # scored token positions (causal shift)

    with open(dataset_path) as f:
        for line in f:
            if count >= max_samples:
                break
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                text = " ".join(m.get("content", "") for m in msgs if m.get("content"))
                if not text.strip():
                    continue
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                device = next(model.parameters()).device
                ids = {k: v.to(device) for k, v in ids.items()}
                with torch.no_grad():
                    out = model(**ids, labels=ids["input_ids"])
                total_loss += float(out.loss)
                count += 1

                n_tok = int(ids["input_ids"].shape[1])
                n_scored = max(n_tok - 1, 1)   # causal LM shifts labels by one
                total_loss_nats += float(out.loss) * n_scored
                total_scored += n_scored
                scored_text = tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)
                total_bytes += len(scored_text.encode("utf-8"))

                if count % 20 == 0:
                    emit_sync("sample_evaluated", {
                        "count": count,
                        "total": max_samples,
                    })
            except Exception:
                pass

    if count == 0:
        return 0.0

    emit_sync("bits_per_byte", {
        "value": bits_per_byte(total_loss_nats, total_scored, total_bytes),
        "total_bytes": total_bytes,
        "total_scored_tokens": total_scored,
    })

    avg_loss = total_loss / count
    try:
        ppl = math.exp(avg_loss)
    except (OverflowError, ValueError):
        ppl = float("inf")

    return round(1.0 / (1.0 + ppl), 6)


def _bench_mmlu(bm_cfg: dict, model_path: Path, emit_sync) -> float:
    """Multiple-choice MMLU accuracy. Score = correct / total."""
    from datasets import load_dataset

    subjects = bm_cfg.get("subjects") or ["all"]
    max_samples = bm_cfg.get("max_samples")

    model, tokenizer = _load_safetensors_model(model_path)
    model.eval()

    correct = 0
    total = 0
    choices = ["A", "B", "C", "D"]

    for subject in subjects:
        dataset = load_dataset(
            "cais/mmlu",
            subject,
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        for row in dataset:
            if max_samples and total >= max_samples:
                break

            question = row["question"]
            opts = row["choices"]
            answer_idx = int(row["answer"])

            prompt = (
                f"Question: {question}\n"
                + "\n".join(f"{choices[i]}. {o}" for i, o in enumerate(opts))
                + "\nAnswer:"
            )

            predicted = _get_choice_answer(model, tokenizer, prompt, choices)
            if predicted == choices[answer_idx]:
                correct += 1
            total += 1

            if total % 50 == 0:
                emit_sync("sample_evaluated", {"count": total, "total": max_samples or total})

    return round(correct / total, 6) if total > 0 else 0.0


def _get_choice_answer(model, tokenizer, prompt: str, choices: list[str]) -> str:
    """Select the highest-logit choice token as the model's answer."""
    import torch

    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**ids).logits[0, -1]  # last token logits

    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    scores = [float(logits[cid]) for cid in choice_ids]
    return choices[scores.index(max(scores))]


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_safetensors_model(model_path: Path):
    """Load HF SafeTensors model. GGUF support via llama-cpp-python is a future enhancement."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_path.suffix == ".gguf":
        raise NotImplementedError(
            "Direct GGUF eval is not yet supported. "
            "Convert to SafeTensors first, or run via a llama-server Worker."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model_path(cfg: dict, run_datasets_dir: Path) -> Path:
    if cfg.get("model_path"):
        return Path(os.path.expanduser(cfg["model_path"]))
    for subdir in ("merge", "finetune", "pretrain", "prune"):
        candidate = run_datasets_dir / subdir
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "eval.model_path not set and no upstream executor output found"
    )


def _benchmark_metric(bm_type: str) -> str:
    return {
        "perplexity": "perplexity_score",
        "mmlu": "accuracy",
        "tool_use": "tool_accuracy",
        "conversation": "judge_score",
        "coding": "pass_at_1",
    }.get(bm_type, "score")


def _bm_key(bm_cfg: dict) -> str:
    """Stable key for deduplicating completed benchmarks in checkpoint."""
    return str(hash(json.dumps(bm_cfg, sort_keys=True)))


def _load_checkpoint(out_dir: Path) -> dict:
    cp = out_dir / ".checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(out_dir: Path, data: dict) -> None:
    (out_dir / ".checkpoint.json").write_text(json.dumps(data))
