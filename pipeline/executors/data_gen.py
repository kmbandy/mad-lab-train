"""Data generation executor — hub-and-spoke coordinator.

Two modes, selected via `mode` config field:

- `generate` (default) — single-completion synthetic data generation against
  llama.cpp workers, with optional judge-model quality gating. The classic
  finetune-corpus generator.

- `trace_farm` (MAD-163) — programmatic-task trace farming. Samples seed
  entries from Qdrant, asks a question-gen worker to formulate a natural-
  language task from each seed, dispatches the task to a real agent that
  uses tools (Qdrant / personal-kg / web), and captures the full multi-turn
  agent trace in MAD-162 shape with trace_source=programmatic_task. Output
  is the memory-conditioning training signal for MAD-161's routing innovation.

Coordinator always runs on mad-lab-main (this process). Workers are
llama.cpp servers on local GPUs or EC2 instances. Context docs are
randomly sampled per generation slot to avoid topic drift.
"""
import asyncio
import datetime
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

        cfg = self.config
        mode = cfg.get("mode", "generate")
        if mode == "trace_farm":
            return await self._run_trace_farm(cfg, out_dir)
        elif mode != "generate":
            raise ValueError(f"data_gen: unknown mode '{mode}' (expected 'generate' or 'trace_farm')")

        # ── mode: generate (existing single-completion path) ──────────────
        output_path = out_dir / "generated.jsonl"
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

    async def _run_trace_farm(self, cfg: dict, out_dir: Path) -> str | None:
        """Trace farming mode (MAD-163, sub-stages 163.1 + 163.2).

        Stage 1 only: seed sampler + question-gen pre-step. No agent loop yet
        (that's 163.3). Emits "question records" with trace_source=
        programmatic_task_question_seed so we can sanity-check the seed-to-
        question pipeline before wiring tool dispatch.
        """
        seed_source = cfg.get("seed_source") or {}
        question_gen = cfg.get("question_gen") or {}
        pattern_mix = question_gen.get("pattern_mix") or {"direct": 1.0}
        samples_target = int(cfg.get("samples", 1000))
        temperature = float(question_gen.get("temperature", 0.8))
        max_tokens = int(question_gen.get("max_tokens", 256))
        ctx_size = int(cfg.get("ctx_size", 4096))
        question_gen_model = question_gen.get("model", cfg.get("model", ""))

        output_path = out_dir / "trace_farm_questions.jsonl"

        # Resume support — checkpoint key uses a distinct name so it doesn't
        # collide with the `generate` mode checkpoint format.
        checkpoint = _load_checkpoint(out_dir)
        samples_done = int(checkpoint.get("trace_farm_samples_done", 0))

        # Question-gen worker pool — usually a tiny pool (1-2 slots on R9700)
        qg_worker_cfgs = [question_gen.get("worker")] if question_gen.get("worker") else []
        qg_worker_cfgs = [w for w in qg_worker_cfgs if w]
        qg_workers = await self._prepare_workers(qg_worker_cfgs, question_gen_model, ctx_size)
        qg_pool = WorkerPool(qg_workers)
        if not qg_pool.workers:
            raise RuntimeError("trace_farm: no healthy question-gen workers")

        await self.emit_event("stage_started", {
            "stage_type": "data_gen",
            "mode": "trace_farm",
            "sub_stage": "questions_only_163.1_163.2",
            "qg_workers": len(qg_pool.workers),
            "qg_capacity": qg_pool.total_capacity,
            "seed_source_type": seed_source.get("type", "qdrant"),
            "pattern_mix": pattern_mix,
        }, stage_type="data_gen")

        # Seed iterator — yields seed-source points lazily so we don't pull
        # 100K records into memory at once.
        seed_iter = _iter_seed_source(seed_source)

        try:
            with open(output_path, "a") as out_f:
                semaphore = asyncio.Semaphore(qg_pool.total_capacity)

                async def farm_one(slot_idx: int) -> dict | None:
                    seed = next(seed_iter, None)
                    if seed is None:
                        return None
                    pattern = _pick_pattern(pattern_mix)
                    prompt = _render_question_prompt(seed, pattern)
                    messages = [
                        {"role": "system", "content": _QUESTION_GEN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    worker = qg_pool.pick()
                    if not worker:
                        return None
                    response = await worker.generate(messages, temperature, max_tokens)
                    if not response or not response.strip():
                        return None
                    return _build_question_record(
                        question=response.strip(),
                        seed=seed,
                        seed_source_cfg=seed_source,
                        pattern=pattern,
                        task_generator="qdrant_seed_questiongen",
                        model_used=question_gen_model,
                    )

                async def run_slot(slot_idx: int) -> None:
                    nonlocal samples_done
                    async with semaphore:
                        if self._force_pause or self._pause_requested:
                            return
                        record = await farm_one(slot_idx)
                        if record is None:
                            return
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        samples_done += 1
                        await self.emit_event("sample_generated", {
                            "count": samples_done,
                            "total": samples_target,
                            "mode": "trace_farm",
                            "pattern": record["trace"].get("pattern"),
                        }, stage_type="data_gen")
                        if samples_done % 100 == 0:
                            _save_checkpoint(out_dir, samples_done, key="trace_farm_samples_done")

                remaining = samples_target - samples_done
                tasks = [asyncio.create_task(run_slot(i)) for i in range(remaining)]
                await asyncio.gather(*tasks)
        finally:
            for worker in qg_workers:
                await worker.close()

        if self._force_pause or self._pause_requested:
            return None

        return str(output_path)

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


# ── MAD-163 trace farming helpers ─────────────────────────────────────────────

_QUESTION_GEN_SYSTEM_PROMPT = (
    "You generate natural-language questions or tasks for an AI agent to perform. "
    "Given a piece of seed content and a target question pattern, produce exactly "
    "one question or task that fits the pattern. Reply with ONLY the question or "
    "task itself — no preamble, no explanation, no quotation marks, no numbering."
)

_QUESTION_PROMPT_TEMPLATES = {
    "direct": (
        "Seed content:\n{seed}\n\n"
        "Write a single direct question someone might ask whose answer is found "
        "in the seed content. Use natural phrasing — the question should sound "
        "like something a real user would type, not a quiz."
    ),
    "cross_session": (
        "Seed content:\n{seed}\n\n"
        "Write a follow-up question someone would ask in a later session after "
        "having seen this content. They remember it vaguely and want to revisit "
        "or extend it. Phrase it as if continuing a conversation, e.g. "
        "\"about that thing we discussed...\" or \"earlier you mentioned...\"."
    ),
    "contradiction": (
        "Seed content:\n{seed}\n\n"
        "Write a question whose typical answer would CONFLICT with what the "
        "seed states. The agent will need to detect the conflict, surface it, "
        "and reconcile. Don't telegraph the conflict — phrase the question "
        "neutrally as if you don't know there's a tension."
    ),
    "stale": (
        "Seed content:\n{seed}\n\n"
        "Write a question that assumes the seed content is current. The agent "
        "may discover the content is outdated and should be expected to flag "
        "the staleness or attempt to update it."
    ),
    "multi_step": (
        "Seed content:\n{seed}\n\n"
        "Write a multi-part question that requires the agent to look up two or "
        "three related pieces of information beyond just the seed itself, then "
        "integrate them. The seed is one piece of a larger answer."
    ),
    "cross_domain": (
        "Seed content:\n{seed}\n\n"
        "Write a question that requires combining the seed with knowledge from "
        "a different domain (if the seed is code, bring in math; if math, bring "
        "in tooling or empirical data; if dialogue, bring in technical detail; "
        "etc.). Force the agent to cross domain boundaries."
    ),
    "adversarial": (
        "Seed content:\n{seed}\n\n"
        "Write a deliberately tricky, ambiguous, or under-specified question "
        "that an agent might handle poorly. Don't make it nonsensical — make "
        "it the kind of question where a competent agent has to ask for "
        "clarification, hedge, or carefully scope their answer."
    ),
    "open_ended": (
        "Seed content:\n{seed}\n\n"
        "Write an open-ended exploratory question that has no single answer. "
        "It should invite synthesis across multiple sources of information, "
        "judgment calls, and a structured response."
    ),
}


class _SeedPoint:
    """Lightweight uniform shape for a seed entry, regardless of source.

    Fields:
        id: stable identifier (string) for provenance back to source.
        content: the main text/payload string the question is generated from.
        domain: optional domain label if known (code|math|tools|dialogue|memory_ops|other).
        source_type: 'qdrant' | 'kg' | future.
        source_name: collection/agent name for traceability.
        payload: full original payload dict (for debugging / future use).
    """
    __slots__ = ("id", "content", "domain", "source_type", "source_name", "payload")

    def __init__(self, id: str, content: str, domain: str | None,
                 source_type: str, source_name: str, payload: dict | None):
        self.id = id
        self.content = content
        self.domain = domain
        self.source_type = source_type
        self.source_name = source_name
        self.payload = payload or {}


def _iter_seed_source(seed_cfg: dict):
    """Yield SeedPoint objects from the configured seed source.

    Implements the iterator protocol so the caller (`farm_one`) can use
    `next(seed_iter, None)` to pull one seed per trace. Sources supported:
        - qdrant: random-shuffled scroll over a collection (optional domain_filter)
        - kg: agent personal-KG entries (deferred — 163.2.1 follow-up)
    """
    src_type = seed_cfg.get("type", "qdrant")
    if src_type == "qdrant":
        yield from _iter_qdrant_seeds(seed_cfg)
    elif src_type == "kg":
        # Personal-KG sampling deferred to 163.2.1 — KG endpoint differs from
        # Qdrant's scroll API; needs its own iterator.
        raise NotImplementedError("trace_farm: seed_source.type='kg' not yet implemented (163.2.1)")
    else:
        raise ValueError(f"trace_farm: unknown seed_source.type '{src_type}'")


def _iter_qdrant_seeds(cfg: dict):
    """Scroll a Qdrant collection and yield SeedPoint objects in randomized order.

    For 163.2 we pull a batch and shuffle in-memory. At 100K+ entries with
    typical request volumes (~190K traces/day max), we periodically re-scroll
    from a random offset to keep the pool fresh without holding everything
    in RAM. domain_filter is applied via Qdrant payload condition if set.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    url = cfg["url"]
    collection = cfg["collection"]
    batch_size = int(cfg.get("batch_size", 1000))
    domain_filter = cfg.get("domain_filter")

    client = QdrantClient(url=url)
    scroll_filter = None
    if domain_filter:
        scroll_filter = Filter(must=[FieldCondition(key="domain", match=MatchValue(value=domain_filter))])

    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        random.shuffle(points)
        for p in points:
            payload = p.payload or {}
            content = (
                payload.get("content")
                or payload.get("text")
                or payload.get("body")
                or ""
            )
            if not content.strip():
                continue
            yield _SeedPoint(
                id=str(p.id),
                content=content,
                domain=payload.get("domain"),
                source_type="qdrant",
                source_name=collection,
                payload=payload,
            )
        if next_offset is None:
            # Reached the end — wrap back to the beginning so the iterator
            # keeps yielding for long-running farms.
            offset = None
        else:
            offset = next_offset


def _pick_pattern(pattern_mix: dict) -> str:
    """Weighted random pick from the configured pattern distribution."""
    if not pattern_mix:
        return "direct"
    patterns = list(pattern_mix.keys())
    weights = [float(pattern_mix[p]) for p in patterns]
    if sum(weights) <= 0:
        return patterns[0]
    return random.choices(patterns, weights=weights, k=1)[0]


def _render_question_prompt(seed: _SeedPoint, pattern: str) -> str:
    """Build the user-side prompt for the question-gen model."""
    template = _QUESTION_PROMPT_TEMPLATES.get(pattern) or _QUESTION_PROMPT_TEMPLATES["direct"]
    # Truncate very long seeds to keep ctx tight — question-gen needs the gist,
    # not the entire payload. 2000 chars ≈ 500 tokens is plenty.
    seed_text = seed.content[:2000] + ("…" if len(seed.content) > 2000 else "")
    return template.format(seed=seed_text)


def _build_question_record(
    question: str,
    seed: _SeedPoint,
    seed_source_cfg: dict,
    pattern: str,
    task_generator: str,
    model_used: str,
) -> dict:
    """Build a MAD-162-shaped record carrying a programmatic-task QUESTION.

    Sub-stage 163.1+.2 emits these as interim records — trace_source is
    `programmatic_task_question_seed`. When 163.3 lands the agent loop,
    the agent's tool-using response gets appended into the same record's
    messages list and trace_source upgrades to `programmatic_task`.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "messages": [
            {"role": "user", "content": question},
        ],
        "trace": {
            "session_id": "",        # populated when 163.3 dispatches to an agent
            "agent": "question-gen-only",
            "model": model_used,
            "timestamp": timestamp,
            "cwd": "",
            "git_branch": "",
            "trace_source": "programmatic_task_question_seed",
            "domain_tag": seed.domain,
            "task_generator": task_generator,
            "pattern": pattern,
            "seed_id": seed.id,
            "seed_source_type": seed.source_type,
            "seed_source_name": seed.source_name,
            "seed_content_preview": seed.content[:500],
            "memory_calls": [],
            "retrieved_spans": [],
            "reasoning_spans": [],
            "generation_spans": [],
            "task_outcome": None,
        },
    }


def _save_checkpoint(out_dir: Path, samples_done: int, key: str = "samples_done") -> None:
    """Persist a counter into the shared checkpoint file under the given key.

    Default key matches the legacy `generate` mode behavior. `trace_farm` uses
    `trace_farm_samples_done` so both can resume independently in the same
    run directory."""
    cp = out_dir / ".checkpoint.json"
    existing = {}
    if cp.exists():
        try:
            existing = json.loads(cp.read_text())
        except Exception:
            existing = {}
    existing[key] = samples_done
    cp.write_text(json.dumps(existing))
