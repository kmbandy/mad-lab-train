"""Data generation executor — hub-and-spoke coordinator.

Coordinator always runs on mad-lab-main (this process). Workers are
llama.cpp servers on local GPUs or EC2 instances. Context docs are
randomly sampled per generation slot to avoid topic drift.
"""
import asyncio
import json
import os
import random
import uuid
from pathlib import Path

from jinja2 import Template as JinjaTemplate
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor
from pipeline.executors.workers import Worker, WorkerPool, prepare_local_worker


class DataGenExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        out_dir = Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id) / "datagen"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "generated.jsonl"

        cfg = self.config
        model = cfg["model"]
        samples_target = int(cfg.get("samples", 1000))
        temperature = float(cfg.get("temperature", 0.85))
        max_tokens = int(cfg.get("max_tokens", 512))
        ctx_size = int(cfg.get("ctx_size", 2048))
        quality_threshold = float(cfg.get("quality_threshold", 0.0))
        judge_model = cfg.get("judge_model")
        system_prompt = cfg.get("system_prompt", "")
        user_template_str = cfg.get("user_template", "{{ context }}")
        topics = cfg.get("topics") or []

        # Load checkpoint
        checkpoint = _load_checkpoint(out_dir)
        samples_done = checkpoint.get("samples_done", 0)

        # Load context pool from upstream dataset_prep output
        context_pool = _load_context_pool(out_dir.parent)

        # Prepare workers
        workers = await self._prepare_workers(cfg.get("workers", []), model, ctx_size)
        pool = WorkerPool(workers)

        if not pool.workers:
            raise RuntimeError("No healthy workers available for data generation")

        await self.emit_event("stage_started", {
            "stage_type": "data_gen",
            "sequence": 0,
            "workers": len(pool.workers),
            "total_capacity": pool.total_capacity,
        }, stage_type="data_gen")

        user_tmpl = JinjaTemplate(user_template_str)
        remaining = samples_target - samples_done

        try:
            with open(output_path, "a") as out_f:
                tasks: list[asyncio.Task] = []
                semaphore = asyncio.Semaphore(pool.total_capacity)

                async def generate_one(slot_idx: int) -> dict | None:
                    context = _sample_context(context_pool, topics, slot_idx)
                    user_content = user_tmpl.render(context=context, topic=context)
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_content})

                    worker = pool.pick()
                    if not worker:
                        return None
                    response = await worker.generate(messages, temperature, max_tokens)
                    if not response:
                        return None
                    return {"messages": messages + [{"role": "assistant", "content": response}]}

                async def run_slot(slot_idx: int) -> None:
                    nonlocal samples_done
                    async with semaphore:
                        if self._force_pause or self._pause_requested:
                            return
                        record = await generate_one(slot_idx)
                        if record is None:
                            return

                        # Optional quality gate
                        if judge_model and quality_threshold > 0:
                            score = await self._judge_quality(record, judge_model, pool, temperature)
                            if score < quality_threshold:
                                await self.emit_event("sample_filtered", {
                                    "reason": "quality_threshold",
                                    "score": score,
                                }, stage_type="data_gen")
                                return

                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        samples_done += 1

                        await self.emit_event("sample_generated", {
                            "count": samples_done,
                            "total": samples_target,
                        }, stage_type="data_gen")

                        if samples_done % 100 == 0:
                            _save_checkpoint(out_dir, samples_done)

                tasks = [asyncio.create_task(run_slot(i)) for i in range(remaining)]
                await asyncio.gather(*tasks)

        finally:
            for worker in workers:
                await worker.close()

        if self._force_pause or self._pause_requested:
            return None

        return str(output_path)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True

    async def _prepare_workers(self, worker_cfgs: list[dict], model: str, ctx_size: int) -> list[Worker]:
        workers = []
        for cfg in worker_cfgs:
            if cfg["type"] == "local":
                worker = await prepare_local_worker(cfg, model, ctx_size)
                workers.append(worker)
            elif cfg["type"] == "ec2":
                # EC2 bootstrap handled by MAD-80; skip with warning for now
                await self.emit_event("ec2_instance_requested", {
                    "instance_type": cfg.get("instance_type", "unknown"),
                    "note": "EC2 bootstrap not yet implemented (MAD-80)",
                }, stage_type="data_gen")
        return workers

    async def _judge_quality(self, record: dict, judge_model: str, pool: WorkerPool, temperature: float) -> float:
        """Score a generated record with a judge model. Returns 0.0–1.0."""
        messages = record.get("messages", [])
        if len(messages) < 2:
            return 0.0

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        last_asst = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")

        judge_messages = [
            {"role": "system", "content": "You are a quality evaluator. Rate the response 1-10. Reply with only a number."},
            {"role": "user", "content": f"Question: {last_user}\n\nResponse: {last_asst}\n\nRating (1-10):"},
        ]
        worker = pool.pick()
        if not worker:
            return 1.0
        response = await worker.generate(judge_messages, temperature=0.1, max_tokens=5)
        try:
            score = float(response.strip().split()[0]) / 10.0
            return max(0.0, min(1.0, score))
        except Exception:
            return 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_context_pool(datasets_dir: Path) -> list[str]:
    """Load context docs from upstream dataset_prep context.jsonl."""
    context_file = datasets_dir / "context.jsonl"
    if not context_file.exists():
        return []
    pool = []
    with open(context_file) as f:
        for line in f:
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                for m in messages:
                    if m.get("role") == "user":
                        pool.append(m["content"])
                        break
            except Exception:
                pass
    return pool


def _sample_context(pool: list[str], topics: list[str], slot_idx: int) -> str:
    """Randomly sample one context doc, falling back to topic round-robin."""
    if pool:
        return random.choice(pool)
    if topics:
        return topics[slot_idx % len(topics)]
    return ""


def _load_checkpoint(out_dir: Path) -> dict:
    cp = out_dir / ".checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(out_dir: Path, samples_done: int) -> None:
    cp = out_dir / ".checkpoint.json"
    cp.write_text(json.dumps({"samples_done": samples_done}))
