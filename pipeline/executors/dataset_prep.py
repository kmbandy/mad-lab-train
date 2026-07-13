import asyncio
import hashlib
import json
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor
from pipeline.conversation_messages import events_to_messages

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

        # MAD-356: a real, declared holdout. Sources with purpose: "eval" land here and
        # are NEVER carved out of the training set.
        holdout_path = out_dir / "holdout.jsonl"

        training_f = open(training_path, "a")
        eval_f = open(holdout_path, "a")
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

                # MAD-356. There was no "eval" arm here, and the lookup ended in
                # `.get(purpose, training_f)` -- so a source declaring purpose: "eval"
                # was SILENTLY ROUTED INTO training.jsonl. The pipeline trained on its
                # own holdout and said nothing.
                #
                # Two changes: "eval" now has a destination, and an unrecognised purpose
                # now RAISES instead of defaulting into the training set. A silent
                # default is exactly how the holdout got trained on -- a typo'd purpose
                # must fail loudly, not quietly contaminate the experiment.
                purpose = source_cfg.get("purpose", "training")
                destinations = {
                    "training": training_f,
                    "eval": eval_f,
                    "context": context_f,
                    "calibration": calibration_f,
                }
                if purpose not in destinations:
                    raise ValueError(
                        f"source {source_name!r} declares unknown purpose {purpose!r}; "
                        f"expected one of {sorted(destinations)}. Refusing to guess -- "
                        f"the previous behaviour silently routed it into the training set."
                    )
                out_file = destinations[purpose]

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
            eval_f.close()
            context_f.close()
            calibration_f.close()
            ddb.close()

        if self._force_pause or self._pause_requested:
            return None

        # MAD-356. Two fixes here.
        #
        # (1) If any source declared purpose: "eval", THAT is the holdout -- use it whole
        #     and keep every training record in train.jsonl. Previously such a source was
        #     silently merged into training and then a random 10% of the merged file was
        #     called "eval", so the eval set was a slice OF TRAIN: in-distribution, and
        #     contaminated by construction.
        #
        # (2) The fallback random split is now SEEDED. It used a bare random.shuffle(),
        #     so the split was different on every run -- meaning the 8 MAD-160 cells were
        #     not even being scored on the same eval set. The success criterion of the
        #     whole experiment is a cross-cell eval comparison; it was not well-posed.
        split_seed = int(self.config.get("shuffle_seed", self.config.get("seed", 160)))
        if holdout_path.exists() and holdout_path.stat().st_size > 0:
            shutil.copyfile(training_path, out_dir / "train.jsonl")
            shutil.copyfile(holdout_path, out_dir / "eval.jsonl")
            await self.emit_event("holdout", {
                "source": "declared", "seed": None,
            }, stage_type="dataset_prep")
        else:
            _split_train_eval(
                training_path, out_dir / "train.jsonl", out_dir / "eval.jsonl",
                train_split, seed=split_seed,
            )
            await self.emit_event("holdout", {
                "source": "random_split_of_training", "seed": split_seed,
                "warning": "eval is an in-distribution slice of train; declare a "
                           "purpose:'eval' source for a true holdout",
            }, stage_type="dataset_prep")

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
    # include_tools: emit FULL block-ordered messages (assistant tool_calls +
    # role:'tool' turns) instead of user+assistant-prose. The `trace` sidecar is
    # unchanged, so memory-trace consumers (default flag off) are unaffected.
    include_tools = bool(cfg.get("include_tools", False))
    agent = cfg.get("agent", "claude-code")
    base_trace_source = cfg.get("trace_source", "organic_work")
    reconstruct = bool(cfg.get("reconstruct_injections", False))
    kg_url = cfg.get("kg_url", "http://100.102.191.30:18830/context")
    kg_timeout = float(cfg.get("kg_timeout_s", 3.0))
    annotate_spans = bool(cfg.get("annotate_spans", False))
    span_min_match = int(cfg.get("span_min_match", 30))
    span_max_blob_chars = int(cfg.get("span_max_blob_chars", 200_000))
    embedding_url = cfg.get("embedding_url")
    embedding_model = cfg.get("embedding_model")
    paraphrase_threshold = float(cfg.get("paraphrase_threshold", 0.75))
    paraphrase_min_chars = int(cfg.get("paraphrase_min_chars", 40))
    embedding_timeout = float(cfg.get("embedding_timeout_s", 10.0))
    labeler_cfg = cfg.get("labeler") if annotate_spans else None
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

    # Tier-3 labeler setup is async/lifecycle-bound — set up once before the
    # file walk, tear down in `finally` so workers are always released.
    labeler_pool = None
    labeler_sample_rate = 0.0
    labeler_temperature = 0.0
    labeler_max_tokens = 64
    if labeler_cfg and labeler_cfg.get("workers"):
        labeler_pool = await _setup_labeler_pool(labeler_cfg)
        labeler_sample_rate = float(labeler_cfg.get("sample_rate", 0.1))
        labeler_temperature = float(labeler_cfg.get("temperature", 0.0))
        labeler_max_tokens = int(labeler_cfg.get("max_tokens", 64))

    count = 0
    try:
        for file_path in files:
            if max_records and count >= max_records:
                break
            try:
                async for record in _parse_claude_session(
                    file_path, unit, min_turn_chars, agent, trace_source, include_tools
                ):
                    if max_records and count >= max_records:
                        break
                    if reconstruct:
                        await _attach_replay_injection(record, kg_url, kg_timeout)
                    if annotate_spans:
                        _annotate_spans_heuristic(record, span_min_match, span_max_blob_chars)
                        if embedding_url and embedding_model:
                            await _annotate_spans_paraphrase(
                                record,
                                embedding_url,
                                embedding_model,
                                paraphrase_threshold,
                                paraphrase_min_chars,
                                embedding_timeout,
                            )
                        if labeler_pool is not None and _should_label(record, labeler_sample_rate):
                            await _annotate_spans_labeler(
                                record, labeler_pool, labeler_temperature, labeler_max_tokens
                            )
                    yield record
                    count += 1
                    if count % 50 == 0:
                        await asyncio.sleep(0)
            except Exception:
                # Skip malformed files; never crash the whole run on one bad transcript
                continue
    finally:
        if labeler_pool is not None:
            await _teardown_labeler_pool(labeler_pool)


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


def _annotate_spans_heuristic(record: dict, min_match: int, max_blob_chars: int) -> None:
    """Tier-1 span annotation: verbatim substring matching (MAD-164 step 3).

    Walks the assistant output and marks any substring of length >= min_match
    that appears verbatim in any memory_call's `results` content as a
    retrieved_span. Everything else becomes a generation_span. reasoning_spans
    are left for tier-3 LLM labeling (deferred).

    High precision, lower recall by design. Catches the obvious copy-paste
    (code blocks, file paths, quoted text, error messages) that gives the
    memory-conditioned router its strongest, cheapest signal.
    """
    if len(record["messages"]) < 2:
        return
    asst_text = record["messages"][1].get("content", "")
    if not asst_text:
        return

    blob = _collect_retrieved_blob(record, max_blob_chars)
    if not blob:
        record["trace"]["generation_spans"] = [(0, len(asst_text))]
        return

    retrieved = _verbatim_matches(asst_text, blob, min_match)
    record["trace"]["retrieved_spans"] = retrieved
    record["trace"]["generation_spans"] = _complement_spans(retrieved, len(asst_text))
    # reasoning_spans intentionally untouched — populated by tier-3 LLM pass


def _collect_retrieved_blob(record: dict, max_chars: int) -> str:
    """Flatten every memory_call's results content into one searchable blob.

    Caps total length at `max_chars` to keep difflib's matching tractable on
    sessions with huge tool dumps (large file reads, repo greps, etc.)."""
    parts: list[str] = []
    total = 0
    for call in record["trace"].get("memory_calls", []):
        for text in _iter_result_text(call.get("results")):
            if not text:
                continue
            remaining = max_chars - total
            if remaining <= 0:
                return "\n".join(parts)
            if len(text) > remaining:
                parts.append(text[:remaining])
                return "\n".join(parts)
            parts.append(text)
            total += len(text) + 1  # +1 for the join separator
    return "\n".join(parts)


def _iter_result_text(results):
    """Yield string fragments from a memory_call.results value, regardless of
    shape (None, str, list of strings, list of {type, text} blocks, etc.)."""
    if results is None:
        return
    if isinstance(results, str):
        yield results
        return
    if isinstance(results, list):
        for item in results:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                # tool_result blocks commonly look like {"type": "text", "text": "..."}
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    yield text
                elif isinstance(text, list):
                    yield from _iter_result_text(text)
        return
    if isinstance(results, dict):
        text = results.get("text") or results.get("content") or ""
        if isinstance(text, str):
            yield text


def _verbatim_matches(asst_text: str, blob: str, min_match: int) -> list[tuple[int, int]]:
    """Find non-overlapping substrings of `asst_text` of length >= min_match
    that appear in `blob`. Returns a sorted, merged list of (start, end) tuples
    indexing into asst_text."""
    import difflib
    matcher = difflib.SequenceMatcher(None, asst_text, blob, autojunk=False)
    spans: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        if block.size >= min_match:
            spans.append((block.a, block.a + block.size))
    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and union overlapping/adjacent span ranges."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _complement_spans(retrieved: list[tuple[int, int]], total_len: int) -> list[tuple[int, int]]:
    """Compute the spans of asst_text NOT covered by `retrieved` — these are
    generation spans by default. Tier-3 LLM labeling may later reclassify
    some of these as reasoning_spans."""
    gens: list[tuple[int, int]] = []
    cursor = 0
    for start, end in retrieved:
        if start > cursor:
            gens.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_len:
        gens.append((cursor, total_len))
    return gens


async def _annotate_spans_paraphrase(
    record: dict,
    embedding_url: str,
    embedding_model: str,
    threshold: float,
    min_chars: int,
    timeout_s: float,
) -> None:
    """Tier-2 span annotation: paraphrase detection via remote embeddings (MAD-164).

    Splits the un-retrieved portion of the assistant output into sentences,
    chunks the retrieved blob the same way, embeds both sides through an
    OpenAI-compatible /embeddings HTTP service, and marks any assistant
    sentence whose max cosine similarity to a retrieved chunk exceeds
    `threshold` as a retrieved_span. Runs after the tier-1 verbatim pass —
    only candidates that survive tier 1 (i.e. are still in generation_spans)
    are submitted, which keeps embedding load proportional to novelty rather
    than total session length."""
    if len(record["messages"]) < 2:
        return
    asst_text = record["messages"][1].get("content", "")
    if not asst_text:
        return

    blob_chars_cap = 200_000  # mirror heuristic cap
    blob = _collect_retrieved_blob(record, blob_chars_cap)
    if not blob:
        return

    # Candidate sentences: only the slices of asst_text that tier 1 left as
    # generation_spans, further split on sentence-ish boundaries.
    gen_spans = record["trace"].get("generation_spans", [])
    candidates: list[tuple[int, int]] = []
    for span_start, span_end in gen_spans:
        for s, e in _split_into_sentences(asst_text, span_start, span_end):
            if e - s >= min_chars:
                candidates.append((s, e))
    if not candidates:
        return

    # Reference chunks from the retrieved blob, same sentence split.
    ref_chunks_full = list(_split_into_sentences(blob, 0, len(blob)))
    ref_chunks = [(s, e) for s, e in ref_chunks_full if e - s >= min_chars]
    if not ref_chunks:
        return

    candidate_texts = [asst_text[s:e] for s, e in candidates]
    reference_texts = [blob[s:e] for s, e in ref_chunks]

    cand_emb = await asyncio.to_thread(
        _embed_via_http, embedding_url, embedding_model, candidate_texts, timeout_s
    )
    if cand_emb is None:
        return
    ref_emb = await asyncio.to_thread(
        _embed_via_http, embedding_url, embedding_model, reference_texts, timeout_s
    )
    if ref_emb is None:
        return

    new_spans: list[tuple[int, int]] = []
    for (cstart, cend), c_vec in zip(candidates, cand_emb):
        if _max_cosine(c_vec, ref_emb) >= threshold:
            new_spans.append((cstart, cend))

    if not new_spans:
        return

    merged_retrieved = _merge_spans(record["trace"]["retrieved_spans"] + new_spans)
    record["trace"]["retrieved_spans"] = merged_retrieved
    record["trace"]["generation_spans"] = _complement_spans(merged_retrieved, len(asst_text))


def _split_into_sentences(text: str, start: int, end: int):
    """Yield (start, end) char offsets into `text` for sentence-like fragments
    within the [start, end) range. Cheap delimiter split — `. `, `? `, `! `,
    newlines. Returns half-open intervals."""
    if end <= start:
        return
    cursor = start
    for i in range(start, end):
        ch = text[i]
        if ch in ".?!" and i + 1 < end and text[i + 1] in (" ", "\n"):
            yield (cursor, i + 1)
            cursor = i + 2
        elif ch == "\n" and i > cursor:
            yield (cursor, i)
            cursor = i + 1
    if cursor < end:
        yield (cursor, end)


def _embed_via_http(url: str, model: str, inputs: list[str], timeout_s: float) -> list[list[float]] | None:
    """POST an OpenAI-compatible /embeddings request and return the vectors.

    Body: {"model": "...", "input": [...]} → {"data": [{"embedding": [...]}, ...]}
    Returns None on any failure — caller treats as a soft skip."""
    import urllib.request
    if not inputs:
        return []
    try:
        payload = json.dumps({"model": model, "input": inputs}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        data = body.get("data") or []
        if len(data) != len(inputs):
            return None
        return [item.get("embedding") or [] for item in data]
    except Exception:
        return None


async def _setup_labeler_pool(labeler_cfg: dict):
    """Bring up a WorkerPool of llama.cpp-compatible servers for tier-3 span
    labeling. Reuses the data_gen worker abstraction so workers are configured
    identically across stages."""
    from pipeline.executors.workers import Worker, WorkerConfig, WorkerPool

    workers = []
    for w_cfg in labeler_cfg.get("workers", []):
        if w_cfg.get("type") != "local":
            continue
        worker = Worker(WorkerConfig(
            type="local",
            host=w_cfg.get("host", "localhost"),
            port=int(w_cfg.get("port", 8080)),
            parallel=int(w_cfg.get("parallel", 16)),
            model=labeler_cfg.get("model", ""),
        ))
        await worker.connect()
        if await worker.health_check():
            workers.append(worker)
        else:
            await worker.close()
    return WorkerPool(workers) if workers else None


async def _teardown_labeler_pool(pool) -> None:
    for worker in pool.workers:
        try:
            await worker.close()
        except Exception:
            pass


def _should_label(record: dict, sample_rate: float) -> bool:
    """Deterministic-by-record sampling: keeps repeat runs over the same
    corpus stable, so quality audits target the same N records each time."""
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    key = (record["trace"].get("session_id", "") + record["trace"].get("timestamp", "")).encode()
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < sample_rate


_LABELER_SYSTEM_PROMPT = (
    "You classify a fragment of an AI assistant's response. The assistant has "
    "access to retrieved context (memory, search results, tool output). Decide "
    "whether the fragment is REASONING (synthesis, integration, or commentary "
    "on retrieved content) or GENERATION (independent content not grounded in "
    "anything retrieved). Reply with a single word: REASONING or GENERATION."
)


async def _annotate_spans_labeler(record: dict, pool, temperature: float, max_tokens: int) -> None:
    """Tier-3 span annotation: LLM-based reasoning/generation labeling (MAD-164).

    For each span tier 1+2 left in generation_spans, asks the labeler whether
    the fragment is REASONING (synthesizes retrieved content) or GENERATION
    (independent). Reasoning-labeled spans move to reasoning_spans; the rest
    stay in generation_spans. Subject to sample_rate at the iterator level."""
    if len(record["messages"]) < 2:
        return
    asst_text = record["messages"][1].get("content", "")
    if not asst_text:
        return
    gen_spans = list(record["trace"].get("generation_spans", []))
    if not gen_spans:
        return

    reasoning: list[tuple[int, int]] = []
    new_gens: list[tuple[int, int]] = []
    for start, end in gen_spans:
        fragment = asst_text[start:end].strip()
        if len(fragment) < 40:
            new_gens.append((start, end))
            continue
        worker = pool.pick()
        if worker is None:
            new_gens.append((start, end))
            continue
        messages = [
            {"role": "system", "content": _LABELER_SYSTEM_PROMPT},
            {"role": "user", "content": fragment[:2000]},
        ]
        try:
            verdict = await worker.generate(messages, temperature, max_tokens)
        except Exception:
            verdict = None
        if verdict and "REASONING" in verdict.upper():
            reasoning.append((start, end))
        else:
            new_gens.append((start, end))

    if reasoning:
        record["trace"]["reasoning_spans"] = _merge_spans(
            record["trace"].get("reasoning_spans", []) + reasoning
        )
        record["trace"]["generation_spans"] = _merge_spans(new_gens)


def _max_cosine(vec: list[float], refs: list[list[float]]) -> float:
    """Max cosine similarity of `vec` against any reference vector. Pure-Python
    so the iterator stays free of numpy/torch deps; OK at sentence scale."""
    if not vec or not refs:
        return 0.0
    v_norm = sum(x * x for x in vec) ** 0.5
    if v_norm == 0:
        return 0.0
    best = 0.0
    for ref in refs:
        if not ref or len(ref) != len(vec):
            continue
        dot = 0.0
        r_norm = 0.0
        for a, b in zip(vec, ref):
            dot += a * b
            r_norm += b * b
        if r_norm == 0:
            continue
        sim = dot / (v_norm * (r_norm ** 0.5))
        if sim > best:
            best = sim
    return best


async def _parse_claude_session(
    file_path: Path,
    unit: str,
    min_turn_chars: int,
    agent: str,
    trace_source: str,
    include_tools: bool = False,
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
                    # ordered event stream for include_tools rendering (additive)
                    "events": [{"kind": "user", "text": content}],
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
                        current["events"].append({
                            "kind": "tool_result",
                            "tool_use_id": tuid,
                            "content": block.get("content", ""),
                            "is_error": block.get("is_error", False),
                        })

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
                        current["events"].append({"kind": "assistant", "text": block.get("text", "")})
                    elif btype == "tool_use":
                        current["tool_uses"].append({
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                            "timestamp": timestamp,
                        })
                        current["events"].append({
                            "kind": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        })
            elif isinstance(content, str):
                current["assistant_text_parts"].append(content)
                current["events"].append({"kind": "assistant", "text": content})

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
        msgs = None
        if include_tools:
            all_events = [e for t in turns for e in t["events"]]
            msgs = events_to_messages(all_events)
        yield _build_trace_record(
            all_user, all_asst, all_calls, session_meta, agent, trace_source,
            turns[0]["started_at"], turns[0]["model"] or "", messages=msgs,
        )
        return

    for turn in turns:
        asst_text = "".join(turn["assistant_text_parts"]).strip()
        if not asst_text and not turn["tool_uses"]:
            continue
        if len(turn["user_text"]) + len(asst_text) < min_turn_chars:
            continue
        msgs = events_to_messages(turn["events"]) if include_tools else None
        yield _build_trace_record(
            turn["user_text"], asst_text, _build_memory_calls(turn),
            session_meta, agent, trace_source,
            turn["started_at"], turn["model"] or "", messages=msgs,
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
    messages: list | None = None,
) -> dict:
    """Build a MAD-162-shaped record. `messages` keeps pipeline compat;
    `trace` carries retrieval / provenance / span-annotation slots.

    When `messages` is supplied (include_tools path) it replaces the default
    user+assistant-prose pair with full tool-interleaved turns; the `trace`
    sidecar is identical either way."""
    return {
        "messages": messages if messages is not None else [
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


def _split_train_eval(
    source: Path, train_out: Path, eval_out: Path, train_ratio: float, seed: int = 160
) -> None:
    """MAD-356: SEEDED. This used a bare `random.shuffle(lines)`, so the train/eval split
    was different on every run -- the 8 MAD-160 cells were not being scored on the same
    eval set, which makes the cross-cell comparison (the entire point of the experiment)
    ill-posed. Use a local Random so we do not perturb global RNG state either.

    Note this eval set is still an in-distribution slice of train. It is a fallback; a
    real holdout means declaring a source with purpose: "eval".
    """
    lines = source.read_text().splitlines()
    if not lines:
        return
    random.Random(seed).shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_out.write_text("\n".join(lines[:split_idx]) + "\n")
    eval_out.write_text("\n".join(lines[split_idx:]) + "\n")
