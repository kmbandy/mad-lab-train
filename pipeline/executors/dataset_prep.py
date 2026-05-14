import asyncio
import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor

_DATASETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           VARCHAR DEFAULT gen_random_uuid(),
    run_id       VARCHAR NOT NULL,
    run_name     VARCHAR,
    source_type  VARCHAR,
    source_name  VARCHAR,
    purpose      VARCHAR DEFAULT 'training',
    content_hash VARCHAR(16),
    messages     JSON,
    token_count  INTEGER,
    created_at   TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_records_run_id ON records (run_id);
CREATE INDEX IF NOT EXISTS idx_records_purpose ON records (purpose);
CREATE INDEX IF NOT EXISTS idx_records_hash ON records (content_hash);
"""


def _open_datasets_db(base_dir: Path):
    import duckdb
    db_path = base_dir / "datasets.db"
    con = duckdb.connect(str(db_path))
    con.execute(_DATASETS_SCHEMA)
    return con


class DatasetPrepExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.models import Run
        from pipeline.settings import settings
        from sqlalchemy import select

        base_dir = Path(os.path.expanduser(settings.log_dir)).parent
        out_dir = base_dir / "datasets" / str(self.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        sources = self.config.get("sources", [])
        train_split = float(self.config.get("train_split", 0.9))
        deduplicate = bool(self.config.get("deduplicate", True))

        # Fetch run name for DuckDB annotation
        run_name: str | None = None
        try:
            result = await self.db.execute(select(Run).where(Run.id == self.run_id))
            run_obj = result.scalar_one_or_none()
            if run_obj:
                run_name = run_obj.name
        except Exception:
            pass

        # Load checkpoint to resume from
        checkpoint = self._load_checkpoint(out_dir)
        completed_sources = set(checkpoint.get("completed_sources", []))
        seen_hashes: set[str] = set(checkpoint.get("seen_hashes", []))

        training_path = out_dir / "training.jsonl"
        context_path = out_dir / "context.jsonl"
        calibration_path = out_dir / "calibration.jsonl"

        training_f = open(training_path, "a")
        context_f = open(context_path, "a")
        calibration_f = open(calibration_path, "a")

        ddb = _open_datasets_db(base_dir)
        now = datetime.now(timezone.utc)

        try:
            for i, source_cfg in enumerate(sources):
                source_name = f"{source_cfg['type']}:{i}"
                if source_name in completed_sources:
                    continue

                if self._force_pause:
                    break

                await self.emit_event("source_started", {
                    "source_name": source_name,
                    "source_type": source_cfg["type"],
                }, stage_type="dataset_prep")

                purpose = source_cfg.get("purpose", "training")
                out_file = {"training": training_f, "context": context_f, "calibration": calibration_f}.get(purpose, training_f)

                count = 0
                async for record in self._iter_source(source_cfg):
                    if self._force_pause or self._pause_requested:
                        break

                    h = _content_hash(record)
                    if deduplicate:
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                    out_file.write(json.dumps(record) + "\n")

                    token_count = _estimate_tokens(record)
                    ddb.execute(
                        "INSERT INTO records (run_id, run_name, source_type, source_name, purpose, "
                        "content_hash, messages, token_count, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        [str(self.run_id), run_name, source_cfg["type"], source_name, purpose,
                         h, json.dumps(record["messages"]), token_count, now],
                    )

                    count += 1

                    if count % 100 == 0:
                        await self.emit_event("records_processed", {
                            "count": count,
                            "source_name": source_name,
                        }, stage_type="dataset_prep")
                        self._save_checkpoint(out_dir, completed_sources, seen_hashes)
                        await asyncio.sleep(0)  # yield to event loop

                await self.emit_event("source_complete", {
                    "source_name": source_name,
                    "total_records": count,
                }, stage_type="dataset_prep")

                if not self._force_pause and not self._pause_requested:
                    completed_sources.add(source_name)
                    self._save_checkpoint(out_dir, completed_sources, seen_hashes)

        finally:
            training_f.close()
            context_f.close()
            calibration_f.close()
            ddb.close()

        if self._force_pause or self._pause_requested:
            return None

        # Split training.jsonl into train/eval
        _split_train_eval(training_path, out_dir / "train.jsonl", out_dir / "eval.jsonl", train_split)

        # Fire-and-forget MotherDuck sync if token is available
        if os.getenv("MOTHERDUCK_TOKEN"):
            asyncio.create_task(_sync_motherduck_bg(base_dir))

        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True

    async def _iter_source(self, cfg: dict):
        source_type = cfg["type"]
        max_records = cfg.get("max_records")

        if source_type == "huggingface":
            async for r in _iter_huggingface(cfg, max_records):
                yield r
        elif source_type == "zim":
            async for r in _iter_zim(cfg, max_records):
                yield r
        elif source_type == "qdrant":
            async for r in _iter_qdrant(cfg, max_records):
                yield r
        elif source_type == "duckdb":
            async for r in _iter_duckdb(cfg, max_records):
                yield r
        elif source_type == "raw":
            async for r in _iter_raw(cfg, max_records):
                yield r
        elif source_type == "claude_jsonl":
            async for r in _iter_claude_jsonl(cfg, max_records):
                yield r

    def _load_checkpoint(self, out_dir: Path) -> dict:
        cp = out_dir / ".checkpoint.json"
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except Exception:
                pass
        return {}

    def _save_checkpoint(self, out_dir: Path, completed: set, seen: set) -> None:
        cp = out_dir / ".checkpoint.json"
        cp.write_text(json.dumps({
            "completed_sources": list(completed),
            "seen_hashes": list(seen),
        }))


# ── Source iterators ─────────────────────────────────────────────

async def _iter_huggingface(cfg: dict, max_records: int | None):
    from datasets import load_dataset
    schema = cfg.get("schema", {})
    fmt = schema.get("format", "messages")
    repo = cfg["repo"]
    split = cfg.get("split", "train")

    ds = load_dataset(repo, split=split, streaming=True, trust_remote_code=False)
    count = 0
    for row in ds:
        if max_records and count >= max_records:
            break
        record = _normalize_row(row, fmt, schema)
        if record:
            yield record
            count += 1
        await asyncio.sleep(0)


async def _iter_zim(cfg: dict, max_records: int | None):
    from bs4 import BeautifulSoup
    from libzim.reader import Archive

    path = os.path.expanduser(cfg["path"])
    query = cfg.get("query")
    archive = Archive(path)
    count = 0

    for i in range(archive.entry_count):
        if max_records and count >= max_records:
            break
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            content = bytes(item.content).decode("utf-8", errors="replace")
            text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
            if not text.strip():
                continue
            if query and query.lower() not in text.lower():
                continue
            yield _wrap_text(text)
            count += 1
        except Exception:
            pass
        if i % 50 == 0:
            await asyncio.sleep(0)


async def _iter_qdrant(cfg: dict, max_records: int | None):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=cfg["url"])
    collection = cfg["collection"]
    limit = min(cfg.get("top_k", 500), max_records or 9999)

    points, _ = client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    for point in points:
        payload = point.payload or {}
        text = payload.get("content") or payload.get("text") or str(payload)
        yield _wrap_text(text)
        await asyncio.sleep(0)


async def _iter_duckdb(cfg: dict, max_records: int | None):
    import duckdb

    path = os.path.expanduser(cfg["path"])
    query = cfg["query"]
    content_col = cfg.get("content_col", "content")

    con = duckdb.connect(path, read_only=True)
    limit = f" LIMIT {max_records}" if max_records else ""
    rows = con.execute(query + limit).fetchall()
    cols = [desc[0] for desc in con.description]
    con.close()

    for row in rows:
        data = dict(zip(cols, row))
        text = str(data.get(content_col, "") or data.get("text", "") or next(iter(data.values()), ""))
        if text.strip():
            yield _wrap_text(text)
        await asyncio.sleep(0)


async def _iter_raw(cfg: dict, max_records: int | None):
    path = Path(os.path.expanduser(cfg["path"]))
    schema = cfg.get("schema", {})
    fmt = schema.get("format", "messages")
    count = 0

    if path.suffix == ".jsonl" or path.suffix == ".json":
        with open(path) as f:
            for line in f:
                if max_records and count >= max_records:
                    break
                try:
                    row = json.loads(line)
                    record = _normalize_row(row, fmt, schema)
                    if record:
                        yield record
                        count += 1
                except Exception:
                    pass
                await asyncio.sleep(0)

    elif path.suffix == ".csv":
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_records and count >= max_records:
                    break
                record = _normalize_row(row, fmt, schema)
                if record:
                    yield record
                    count += 1
                await asyncio.sleep(0)

    elif path.suffix == ".txt":
        with open(path) as f:
            for line in f:
                if max_records and count >= max_records:
                    break
                if line.strip():
                    yield _wrap_text(line.strip())
                    count += 1
                await asyncio.sleep(0)


async def _iter_claude_jsonl(cfg: dict, max_records: int | None):
    """Iterate Claude Code JSONL session transcripts (MAD-164).

    Each transcript file is one session. Emits MAD-162-shaped records — one
    per turn (default) or one per session — with `messages` for pipeline
    compatibility and a `trace` sidecar carrying memory_calls, session metadata,
    and provenance for downstream span annotation and memory-conditioned
    routing training (MAD-161).

    If `reconstruct_injections: true`, each turn is additionally enriched by
    replaying the personal-KG `/context` endpoint against the user prompt
    (mirroring the runtime hook logic) and attaching the result as a synthetic
    `personal-kg-context-replay` memory_call. This recovers the memory-
    conditioning signal that runtime hooks would have injected at session
    time but that was never persisted to disk.
    """
    path = Path(os.path.expanduser(cfg["path"]))
    recursive = bool(cfg.get("recursive", True))
    unit = cfg.get("unit", "turn")
    min_turn_chars = int(cfg.get("min_turn_chars", 0))
    agent = cfg.get("agent", "claude-code")
    base_trace_source = cfg.get("trace_source", "organic_work")
    reconstruct = bool(cfg.get("reconstruct_injections", False))
    kg_url = cfg.get("kg_url", "http://100.102.191.30:18830/context")
    kg_timeout = float(cfg.get("kg_timeout_s", 3.0))
    trace_source = (
        "organic_work_with_replay_injection"
        if reconstruct and base_trace_source == "organic_work"
        else base_trace_source
    )

    if path.is_file():
        files = [path]
    elif path.is_dir():
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        files = sorted(path.glob(pattern))
    else:
        return

    count = 0
    for file_path in files:
        if max_records and count >= max_records:
            break
        try:
            async for record in _parse_claude_session(
                file_path, unit, min_turn_chars, agent, trace_source
            ):
                if max_records and count >= max_records:
                    break
                if reconstruct:
                    await _attach_replay_injection(record, kg_url, kg_timeout)
                yield record
                count += 1
                if count % 50 == 0:
                    await asyncio.sleep(0)
        except Exception:
            # Skip malformed files; never crash the whole run on one bad transcript
            continue


async def _attach_replay_injection(record: dict, kg_url: str, timeout_s: float) -> None:
    """Query the personal-KG /context endpoint with the turn's user prompt and
    prepend the result as a synthetic memory_call. Mirrors the runtime hook
    behavior so historical traces gain the memory-conditioning signal that
    was injected at runtime but never persisted (MAD-164)."""
    user_text = record["messages"][0]["content"] if record["messages"] else ""
    if not user_text:
        return
    query = user_text[:500]
    context = await asyncio.to_thread(_kg_context_request, kg_url, query, timeout_s)
    if context is None:
        return  # request failed — leave the trace unenriched
    success = bool(context.strip())
    synthetic_call = {
        "tool_name": "personal-kg-context-replay",
        "operation_type": "search",
        "query": query,
        "results": context,
        "result_count": context.count("\n[") + (1 if context.strip() else 0),
        "success": success,
        "latency_ms": None,
        "timestamp": record["trace"].get("timestamp", ""),
    }
    record["trace"]["memory_calls"] = [synthetic_call] + record["trace"]["memory_calls"]


def _kg_context_request(url: str, query: str, timeout_s: float) -> str | None:
    """Sync HTTP call to the KG /context endpoint. Returns the context string,
    or None on any failure (which is logged-but-not-fatal at the call site)."""
    import urllib.request
    try:
        payload = json.dumps({"query": query}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode()).get("context", "")
    except Exception:
        return None


async def _parse_claude_session(
    file_path: Path,
    unit: str,
    min_turn_chars: int,
    agent: str,
    trace_source: str,
):
    """Parse one Claude Code JSONL session into MAD-162 turn records.

    A turn starts at a user message with string content (real user prompt)
    and continues until the next such message or EOF. Tool-result wrappers
    (user messages with list content) belong to the turn that issued the
    tool_use.
    """
    messages: list[dict] = []
    session_meta: dict = {}

    with open(file_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") not in ("user", "assistant"):
                continue
            if not session_meta:
                session_meta = {
                    "session_id": obj.get("sessionId", ""),
                    "cwd": obj.get("cwd", ""),
                    "git_branch": obj.get("gitBranch", ""),
                }
            messages.append(obj)

    if not messages:
        return

    turns: list[dict] = []
    current: dict | None = None

    for obj in messages:
        t = obj["type"]
        msg = obj.get("message", {})
        content = msg.get("content", "")
        timestamp = obj.get("timestamp", "")

        if t == "user":
            if isinstance(content, str):
                if current is not None:
                    turns.append(current)
                current = {
                    "user_text": content,
                    "assistant_text_parts": [],
                    "tool_uses": [],
                    "tool_results": {},
                    "started_at": timestamp,
                    "model": None,
                }
            elif isinstance(content, list):
                if current is None:
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        tuid = block.get("tool_use_id", "")
                        current["tool_results"][tuid] = {
                            "content": block.get("content", ""),
                            "is_error": block.get("is_error", False),
                        }

        elif t == "assistant":
            if current is None:
                continue
            if not current["model"]:
                current["model"] = msg.get("model", "")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        current["assistant_text_parts"].append(block.get("text", ""))
                    elif btype == "tool_use":
                        current["tool_uses"].append({
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                            "timestamp": timestamp,
                        })
            elif isinstance(content, str):
                current["assistant_text_parts"].append(content)

    if current is not None:
        turns.append(current)

    if unit == "session":
        if not turns:
            return
        all_user = "\n\n".join(t["user_text"] for t in turns)
        all_asst = "\n\n".join("".join(t["assistant_text_parts"]).strip() for t in turns)
        if len(all_user) + len(all_asst) < min_turn_chars:
            return
        all_calls: list[dict] = []
        for t in turns:
            all_calls.extend(_build_memory_calls(t))
        yield _build_trace_record(
            all_user, all_asst, all_calls, session_meta, agent, trace_source,
            turns[0]["started_at"], turns[0]["model"] or "",
        )
        return

    for turn in turns:
        asst_text = "".join(turn["assistant_text_parts"]).strip()
        if not asst_text and not turn["tool_uses"]:
            continue
        if len(turn["user_text"]) + len(asst_text) < min_turn_chars:
            continue
        yield _build_trace_record(
            turn["user_text"], asst_text, _build_memory_calls(turn),
            session_meta, agent, trace_source,
            turn["started_at"], turn["model"] or "",
        )


def _build_memory_calls(turn: dict) -> list[dict]:
    """Pair tool_use entries with their matching tool_result entries."""
    calls = []
    for use in turn["tool_uses"]:
        result = turn["tool_results"].get(use["id"])
        if result is not None:
            success = not result.get("is_error", False)
            results_content = result.get("content")
            result_count = _count_tool_results(results_content)
        else:
            success = None
            results_content = None
            result_count = 0
        calls.append({
            "tool_name": use["name"],
            "operation_type": _classify_tool_op(use["name"]),
            "query": use["input"],
            "results": results_content,
            "result_count": result_count,
            "success": success,
            "latency_ms": None,
            "timestamp": use["timestamp"],
        })
    return calls


_OP_HINTS = (
    ("search", "search"), ("find", "search"), ("query", "search"),
    ("read", "search"), ("get", "search"), ("list", "search"),
    ("fetch", "search"), ("grep", "search"),
    ("write", "write"), ("create", "write"), ("add", "write"), ("insert", "write"),
    ("update", "update"), ("edit", "update"), ("modify", "update"),
    ("delete", "delete"), ("remove", "delete"),
)


def _classify_tool_op(tool_name: str) -> str:
    """Map a tool name to an MAD-162 operation_type (best-effort by keyword)."""
    if not tool_name:
        return "other"
    n = tool_name.lower()
    for hint, op in _OP_HINTS:
        if hint in n:
            return op
    return "other"


def _count_tool_results(content) -> int:
    if content is None:
        return 0
    if isinstance(content, list):
        return len(content)
    if isinstance(content, str):
        return 1 if content.strip() else 0
    return 1


def _build_trace_record(
    user_text: str,
    asst_text: str,
    memory_calls: list[dict],
    session_meta: dict,
    agent: str,
    trace_source: str,
    started_at: str,
    model: str,
) -> dict:
    """Build a MAD-162-shaped record. `messages` keeps pipeline compat;
    `trace` carries retrieval / provenance / span-annotation slots."""
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ],
        "trace": {
            "session_id": session_meta.get("session_id", ""),
            "agent": agent,
            "model": model,
            "timestamp": started_at,
            "cwd": session_meta.get("cwd", ""),
            "git_branch": session_meta.get("git_branch", ""),
            "trace_source": trace_source,
            "domain_tag": None,
            "memory_calls": memory_calls,
            "retrieved_spans": [],
            "reasoning_spans": [],
            "generation_spans": [],
            "task_outcome": None,
        },
    }


# ── Normalization helpers ─────────────────────────────────────────

def _normalize_row(row: dict, fmt: str, schema: dict) -> dict | None:
    system = schema.get("system_prompt") or (row.get(schema.get("system_col", "")) if schema.get("system_col") else None)
    messages = []

    if system:
        messages.append({"role": "system", "content": str(system)})

    if fmt == "instruction_response":
        instruction = row.get(schema.get("instruction_col", "instruction"), "")
        response = row.get(schema.get("response_col", "response"), "")
        if not instruction:
            return None
        messages.append({"role": "user", "content": str(instruction)})
        if response:
            messages.append({"role": "assistant", "content": str(response)})

    elif fmt == "qa":
        question = row.get(schema.get("question_col", "question"), "")
        answer = row.get(schema.get("answer_col", "answer"), "")
        if not question:
            return None
        messages.append({"role": "user", "content": str(question)})
        if answer:
            messages.append({"role": "assistant", "content": str(answer)})

    elif fmt == "messages":
        raw = row.get("messages") or row.get("conversations") or []
        if not raw:
            return None
        for m in raw:
            role = m.get("role") or ("user" if m.get("from") == "human" else "assistant")
            content = m.get("content") or m.get("value") or ""
            messages.append({"role": role, "content": str(content)})

    elif fmt == "text":
        text = row.get("text") or row.get("content") or next(iter(row.values()), "")
        if not text:
            return None
        messages.append({"role": "user", "content": str(text)})

    if not messages:
        return None
    return {"messages": messages}


def _wrap_text(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def _content_hash(record: dict) -> str:
    content = " ".join(m.get("content", "") for m in record.get("messages", []))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _estimate_tokens(record: dict) -> int:
    content = " ".join(m.get("content", "") for m in record.get("messages", []))
    return max(1, len(content) // 4)


async def _sync_motherduck_bg(base_dir: Path) -> None:
    from pipeline.routers.datasets import _do_sync
    try:
        await asyncio.to_thread(_do_sync)
    except Exception:
        pass  # sync failure never surfaces to the pipeline run


def _split_train_eval(source: Path, train_out: Path, eval_out: Path, train_ratio: float) -> None:
    lines = source.read_text().splitlines()
    if not lines:
        return
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_out.write_text("\n".join(lines[:split_idx]) + "\n")
    eval_out.write_text("\n".join(lines[split_idx:]) + "\n")
