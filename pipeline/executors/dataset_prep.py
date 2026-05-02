import asyncio
import hashlib
import json
import os
import random
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class DatasetPrepExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        out_dir = Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        sources = self.config.get("sources", [])
        train_split = float(self.config.get("train_split", 0.9))
        deduplicate = bool(self.config.get("deduplicate", True))

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

                    if deduplicate:
                        h = _content_hash(record)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                    out_file.write(json.dumps(record) + "\n")
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

        if self._force_pause or self._pause_requested:
            return None

        # Split training.jsonl into train/eval
        _split_train_eval(training_path, out_dir / "train.jsonl", out_dir / "eval.jsonl", train_split)

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


# ── Source iterators ──────────────────────────────────────────────────────────

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


# ── Normalization helpers ─────────────────────────────────────────────────────

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


def _split_train_eval(source: Path, train_out: Path, eval_out: Path, train_ratio: float) -> None:
    lines = source.read_text().splitlines()
    if not lines:
        return
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_out.write_text("\n".join(lines[:split_idx]) + "\n")
    eval_out.write_text("\n".join(lines[split_idx:]) + "\n")
