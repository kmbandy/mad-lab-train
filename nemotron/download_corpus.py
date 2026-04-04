#!/usr/bin/env python3
"""Download and assemble the pretraining corpus for the 1B base model.

Sources:
  1. arXiv bulk (q-fin.*, math.*, stat.*, cs.AI/LG/CL) — public S3
  2. SEC EDGAR filings (10-K, 10-Q, 8-K) — free API
  3. Wikipedia (full English, all articles) — HuggingFace
  4. FineWeb-Edu (score>=4, sample-100BT) — HuggingFace
  5. GitHub code (Python + R) — codeparrot/github-code
  6. Fiction / creative writing — PleIAs/US-PD-Books + TinyStoriesV2
  7. Existing assets (tool_calls.jsonl, quant SO Q&A)

Token targets per source:
  FineWeb-Edu   ~50B  (50M docs × ~1k tokens avg)
  GitHub code   ~10B  (10M files × ~1k tokens avg)
  arXiv         ~8B   (~2M papers × ~4k tokens avg)
  Wikipedia     ~4B   (all English Wikipedia)
  EDGAR         ~3.5B (500k filings × ~7k tokens avg)
  Fiction       ~5B   (5M docs)
  Assets        ~50M  (existing datasets)
  ─────────────────────────────────────────────
  TOTAL         ~80B  (process_corpus.py multi-epochs small sources to hit 100B)

Output: ~/corpus/raw/<source>/*.jsonl  (one doc per line, {"text": "..."})

Run this on mad-lab (downloads to local disk).
Then run process_corpus.py to tokenize + pack + upload to S3.
"""

from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path

import httpx
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR = Path("/mnt/mainpc/corpus/raw")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# arXiv categories to pull
ARXIV_CATEGORIES = [
    "q-fin",   # quantitative finance
    "math",    # mathematics
    "stat",    # statistics
    "cs.AI",   # artificial intelligence
    "cs.LG",   # machine learning
    "cs.CL",   # computation and language (NLP)
]

# SEC filing types
EDGAR_FORM_TYPES = ["10-K", "10-Q", "8-K"]

# FineWeb-Edu minimum quality score
FINEWEB_MIN_SCORE = 4.0

# GitHub code languages
GITHUB_LANGUAGES = {"Python", "R"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> int:
    """Append records to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# 1. arXiv papers (via RedPajama-Data-1T HuggingFace dataset)
# ---------------------------------------------------------------------------

def download_arxiv() -> None:
    """Pull arXiv papers from two complementary sources:

      1. gfissore/arxiv-abstracts-2021 — 2M abstracts with category metadata,
         filtered to our target domains (q-fin, math, stat, cs.AI/LG/CL).
         Title + abstract per paper. ~500M tokens.

      2. ccdv/arxiv-summarization — ~215k full papers (article + abstract),
         no filtering needed (already high quality). ~1B tokens.

    Combined: ~2M records, ~1.5B tokens of scientific text.
    """
    out_path = CORPUS_DIR / "arxiv" / "arxiv.jsonl"

    # -- Part 1: 2M filtered abstracts ------------------------------------
    target_cats = set(ARXIV_CATEGORIES)
    abstracts_target = 1_500_000
    existing = _count_existing(out_path)

    if existing < abstracts_target:
        print(f"  [arxiv] streaming abstracts from gfissore/arxiv-abstracts-2021 "
              f"(have {existing:,}/{abstracts_target:,})...")

        ds = load_dataset(
            "gfissore/arxiv-abstracts-2021",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        records = []
        written = existing

        for doc in ds:
            raw_cats = doc.get("categories") or ""
            cats = raw_cats if isinstance(raw_cats, list) else raw_cats.split()
            if not any(any(c.startswith(t) for t in target_cats) for c in cats):
                continue

            title    = (doc.get("title")    or "").replace("\n", " ").strip()
            abstract = (doc.get("abstract") or "").replace("\n", " ").strip()
            if not abstract:
                continue

            text = f"{title}\n\n{abstract}"
            records.append({"text": text, "source": "arxiv_abstract",
                             "categories": " ".join(cats) if cats else ""})
            written += 1

            if len(records) >= 2000:
                _write_jsonl(out_path, records)
                records = []
                print(f"  [arxiv] {written:,} abstracts written...", end="\r")

            if written >= abstracts_target:
                break

        if records:
            _write_jsonl(out_path, records)
        print(f"\n  [arxiv] abstracts done — {written:,} records")

    # -- Part 2: full papers (article body text) --------------------------
    full_target = abstracts_target + 200_000
    existing = _count_existing(out_path)

    if existing < full_target:
        print(f"  [arxiv] streaming full papers from ccdv/arxiv-summarization "
              f"(have {existing:,}/{full_target:,})...")

        ds2 = load_dataset(
            "ccdv/arxiv-summarization",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        records = []
        written = existing

        for doc in ds2:
            article  = (doc.get("article")  or "").strip()
            abstract = (doc.get("abstract") or "").strip()
            if not article or len(article) < 500:
                continue
            text = f"{abstract}\n\n{article}" if abstract else article
            records.append({"text": text[:15_000], "source": "arxiv_full"})
            written += 1

            if len(records) >= 2000:
                _write_jsonl(out_path, records)
                records = []
                print(f"  [arxiv] {written:,} full papers written...", end="\r")

            if written >= full_target:
                break

        if records:
            _write_jsonl(out_path, records)
        print(f"\n  [arxiv] full papers done — {written:,} total records")


# ---------------------------------------------------------------------------
# 2. SEC EDGAR filings
# ---------------------------------------------------------------------------

def download_edgar(max_filings: int = 500_000) -> None:
    """Pull SEC filings from EDGAR full-text search API.

    Free, no auth required. 500k filings × ~7k tokens avg ≈ 3.5B tokens.
    SEC allows up to 10 req/s. Runs all 3 form types in parallel via asyncio
    with a shared semaphore — cuts wall time from ~42hrs to ~28 minutes.
    """
    out_path = CORPUS_DIR / "edgar" / "edgar.jsonl"
    existing = _count_existing(out_path)
    if existing >= max_filings:
        print(f"  [edgar] already have {existing:,} filings, skipping")
        return

    print(f"  [edgar] fetching {max_filings:,} filings in parallel "
          f"(have {existing:,})...")

    asyncio.run(_edgar_async(out_path, max_filings, existing))
    total = _count_existing(out_path)
    print(f"\n  [edgar] done — {total:,} filings")


async def _edgar_async(out_path: Path, max_filings: int, existing: int) -> None:
    base_url    = "https://efts.sec.gov/LATEST/search-index"
    headers     = {"User-Agent": "mad-lab-research corpus@mad-lab.local"}
    target      = max_filings // len(EDGAR_FORM_TYPES)

    # SEC rate limit: 10 req/s. Semaphore shared across all form-type tasks.
    sem         = asyncio.Semaphore(10)
    write_lock  = asyncio.Lock()
    write_buf: list[dict] = []
    written_total = existing

    async def _flush(force: bool = False) -> None:
        nonlocal write_buf, written_total
        if not write_buf:
            return
        if force or len(write_buf) >= 2000:
            async with write_lock:
                _write_jsonl(out_path, write_buf)
                written_total += len(write_buf)
                print(f"  [edgar] {written_total:,} filings written...", end="\r")
                write_buf = []

    async def _fetch_form(client: httpx.AsyncClient, form_type: str) -> None:
        nonlocal write_buf
        page = 0
        count = 0
        while count < target:
            async with sem:
                try:
                    resp = await client.get(
                        base_url,
                        params={
                            "q":        "",
                            "dateRange": "custom",
                            "startdt":  "2001-01-01",
                            "enddt":    "2026-01-01",
                            "forms":    form_type,
                            "from":     page * 10,
                            "size":     10,
                        },
                        headers=headers,
                        timeout=30,
                    )
                except Exception as e:
                    print(f"\n  [edgar:{form_type}] request error: {e}, retrying...")
                    await asyncio.sleep(2)
                    continue

            if resp.status_code == 429:
                await asyncio.sleep(5)
                continue
            if resp.status_code in (404,) or resp.status_code >= 500:
                page += 1
                continue
            if resp.status_code != 200:
                await asyncio.sleep(1)
                continue

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break   # exhausted this form type

            batch: list[dict] = []
            for hit in hits:
                src   = hit.get("_source", {})
                parts = []
                if src.get("display_names"):
                    parts.append(src["display_names"][0])
                if src.get("file_date"):
                    parts.append(f"Filed: {src['file_date']}")
                if src.get("period_of_report"):
                    parts.append(f"Period: {src['period_of_report']}")
                if src.get("description"):
                    parts.append(src["description"])
                if src.get("biz_location"):
                    parts.append(f"Location: {src['biz_location']}")
                text = "\n".join(parts)
                if len(text.strip()) < 100:
                    continue
                batch.append({"text": text.strip(), "source": "edgar",
                               "form_type": form_type})
                count += 1

            async with write_lock:
                write_buf.extend(batch)

            await _flush()
            page += 1

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_fetch_form(client, ft) for ft in EDGAR_FORM_TYPES])

    await _flush(force=True)


# ---------------------------------------------------------------------------
# ZIM file paths (local HDD)
# ---------------------------------------------------------------------------

ZIM_DIR = Path("/mnt/hdd/project-nomad/storage/zim")

ZIM_WIKIPEDIA  = ZIM_DIR / "wikipedia_en_all_maxi_2026-02.zim"
ZIM_STACKEXCHANGE = {
    "stackoverflow":      ZIM_DIR / "stackoverflow.com_en_all_2023-11.zim",
    "math_se":            ZIM_DIR / "math.stackexchange.com_en_all_2026-02.zim",
    "stats_se":           ZIM_DIR / "stats.stackexchange.com_en_all_2026-02.zim",
    "quant_se":           ZIM_DIR / "quant.stackexchange.com_en_all_2026-02.zim",
    "rpg_se":             ZIM_DIR / "rpg.stackexchange.com_en_all_2026-02.zim",
    "softeng_se":         ZIM_DIR / "softwareengineering.stackexchange.com_en_all_2026-02.zim",
}


def _extract_zim(zim_path: Path, out_path: Path, source_name: str,
                 max_articles: int = 99_999_999) -> None:
    """Extract article text from a ZIM file into JSONL format.

    Uses libzim to iterate all articles, strips HTML tags, writes
    {"text": ..., "source": ..., "title": ...} per article.
    """
    import re
    from libzim.reader import Archive  # type: ignore

    existing = _count_existing(out_path)
    if existing >= max_articles:
        print(f"  [{source_name}] already have {existing:,} articles, skipping")
        return

    print(f"  [{source_name}] extracting from {zim_path.name} "
          f"(have {existing:,})...")

    tag_re  = re.compile(r"<[^>]+>")
    ref_re  = re.compile(r"\[\d+\]")

    archive  = Archive(str(zim_path))
    records  = []
    written  = existing
    skipped  = 0
    total    = archive.entry_count

    for idx in range(total):
        try:
            entry = archive._get_entry_by_id(idx)
            if entry.is_redirect:
                skipped += 1
                continue
            item    = entry.get_item()
            mimetype = item.mimetype
            if "html" not in mimetype:
                skipped += 1
                continue
            raw  = bytes(item.content).decode("utf-8", errors="replace")

            # Strip HTML
            text = tag_re.sub(" ", raw)
            text = ref_re.sub("", text)
            text = " ".join(text.split())

            if len(text) < 200:
                skipped += 1
                continue

            records.append({"text": text[:15_000], "source": source_name,
                             "title": entry.title})
            written += 1
        except Exception:
            skipped += 1
            continue

        if len(records) >= 2000:
            _write_jsonl(out_path, records)
            records = []
            gc.collect()
            print(f"  [{source_name}] {written:,} articles written...", end="\r")

        if written >= max_articles:
            break

    if records:
        _write_jsonl(out_path, records)

    print(f"\n  [{source_name}] done — {written:,} articles "
          f"(skipped {skipped:,} non-article entries)")


# ---------------------------------------------------------------------------
# 3. Wikipedia (from local ZIM file)
# ---------------------------------------------------------------------------

def download_wikipedia() -> None:
    """Extract full English Wikipedia from local ZIM file.

    Uses /mnt/hdd/project-nomad/storage/zim/wikipedia_en_all_maxi_2026-02.zim
    (115GB local copy). Zero bandwidth, no HuggingFace streaming, no OOM risk.
    ~6.7M articles × ~2KB avg = ~4B tokens. Cap at 7M to avoid scanning the
    maxi ZIM's extra talk/disambig pages (27M total entries).
    """
    out_path = CORPUS_DIR / "wikipedia" / "wikipedia.jsonl"
    _extract_zim(ZIM_WIKIPEDIA, out_path, "wikipedia", max_articles=7_000_000)


# ---------------------------------------------------------------------------
# 3b. StackExchange (from local ZIM files)
# ---------------------------------------------------------------------------

def download_stackexchange() -> None:
    """Extract all StackExchange ZIM files from local HDD.

    Sources:
      - Stack Overflow (2023-11) — ~50M Q&A pairs, code-heavy
      - Math SE (2026-02)        — proofs, equations, problem solving
      - Stats SE (2026-02)       — statistics, ML theory, R/Python
      - Quant SE (2026-02)       — quantitative finance, trading
      - RPG SE (2026-02)         — D&D, roleplay, game mechanics
      - Software Eng SE (2026-02) — architecture, design patterns

    Combined: massive high-quality Q&A corpus covering all our target domains.
    """
    for name, zim_path in ZIM_STACKEXCHANGE.items():
        if not zim_path.exists():
            print(f"  [{name}] ZIM not found at {zim_path}, skipping")
            continue
        out_path = CORPUS_DIR / "stackexchange" / f"{name}.jsonl"
        _extract_zim(zim_path, out_path, name)


# ---------------------------------------------------------------------------
# 4. FineWeb-Edu (high quality educational/technical web text)
# ---------------------------------------------------------------------------

def download_fineweb(max_docs: int = 50_000_000) -> None:
    """Pull high-scoring FineWeb-Edu documents (score >= 4).

    Uses sample-100BT (~100B token subset of FineWeb-Edu). With score>=4
    filter roughly 40-50% of docs pass, giving ~40-50B tokens — the largest
    single component of our corpus. High-quality math, science, code, general.
    """
    out_path = CORPUS_DIR / "fineweb" / "fineweb.jsonl"
    existing = _count_existing(out_path)
    if existing >= max_docs:
        print(f"  [fineweb] already have {existing:,} docs, skipping")
        return

    print(f"  [fineweb] streaming from HuggingFace sample-100BT "
          f"(have {existing:,}/{max_docs:,})...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    records = []
    written = existing
    skipped = 0

    for doc in ds:
        score = doc.get("score", 0)
        if score < FINEWEB_MIN_SCORE:
            skipped += 1
            continue

        text = doc.get("text", "")
        if len(text) < 200:
            skipped += 1
            continue

        records.append({"text": text[:10_000], "source": "fineweb",
                         "score": score})
        written += 1

        if len(records) >= 5000:
            _write_jsonl(out_path, records)
            records = []
            gc.collect()
            print(f"  [fineweb] {written:,} docs written "
                  f"(skipped {skipped:,})...", end="\r")

        if written >= max_docs:
            break

    if records:
        _write_jsonl(out_path, records)

    print(f"\n  [fineweb] done — {written:,} docs (skipped {skipped:,})")


# ---------------------------------------------------------------------------
# 5. GitHub code (Python + R)
# ---------------------------------------------------------------------------

def download_github_code(max_files: int = 10_000_000) -> None:
    """Pull Python and R source files from codeparrot/github-code.

    ~115B tokens total; we filter to Python + R (our core languages) and
    take 10M files ≈ 10B tokens. Provides code pre-exposure so the base
    model has genuine coding intuition before any fine-tuning.
    """
    out_path = CORPUS_DIR / "github_code" / "github_code.jsonl"
    existing = _count_existing(out_path)
    if existing >= max_files:
        print(f"  [github_code] already have {existing:,} files, skipping")
        return

    print(f"  [github_code] streaming Python + R from codeparrot/github-code "
          f"(have {existing:,}/{max_files:,})...")

    ds = load_dataset(
        "codeparrot/github-code",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    records = []
    written = existing
    skipped = 0

    for doc in ds:
        lang = doc.get("language", "")
        if lang not in GITHUB_LANGUAGES:
            skipped += 1
            continue

        code = doc.get("code", "")
        if not code or len(code) < 100:
            skipped += 1
            continue

        # Skip files that are mostly non-code (data files, auto-generated)
        lines = code.split("\n")
        if len(lines) < 5:
            skipped += 1
            continue

        records.append({"text": code[:8_000], "source": "github_code",
                         "language": lang,
                         "repo": doc.get("repo_name", ""),
                         "path": doc.get("path", "")})
        written += 1

        if len(records) >= 5000:
            _write_jsonl(out_path, records)
            records = []
            print(f"  [github_code] {written:,} files written "
                  f"(skipped {skipped:,})...", end="\r")

        if written >= max_files:
            break

    if records:
        _write_jsonl(out_path, records)

    print(f"\n  [github_code] done — {written:,} files (skipped {skipped:,})")


# ---------------------------------------------------------------------------
# 6. Fiction / creative writing
# ---------------------------------------------------------------------------

def download_fiction(max_docs: int = 5_000_000) -> None:
    """Pull fiction and narrative text for roleplay domain coverage.

    Two sources:
      - PleIAs/US-PD-Books: public domain books (novels, stories, essays).
        ~200k books chunked into passages; rich long-form narrative.
      - roneneldan/TinyStoriesV2: short narrative completions; good for
        teaching coherent short-form story structure.

    Target: ~5M docs ≈ 5B tokens.
    """
    out_path = CORPUS_DIR / "fiction" / "fiction.jsonl"
    existing = _count_existing(out_path)
    if existing >= max_docs:
        print(f"  [fiction] already have {existing:,} docs, skipping")
        return

    written = existing
    records: list[dict] = []

    def _flush() -> None:
        nonlocal records
        if records:
            _write_jsonl(out_path, records)
            records = []

    # -- PleIAs/US-PD-Books (public domain books) --------------------------
    books_target = int(max_docs * 0.80)   # 80% from books
    if written < books_target:
        print(f"  [fiction] loading PleIAs/US-PD-Books "
              f"(have {written:,}/{books_target:,})...")
        try:
            ds_books = load_dataset(
                "PleIAs/US-PD-Books",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            for doc in ds_books:
                text = doc.get("text", "") or doc.get("complete_text", "")
                if not text or len(text) < 500:
                    continue
                # Books are long — chunk into ~3000 char passages for variety
                for i in range(0, min(len(text), 300_000), 3000):
                    chunk = text[i: i + 3000].strip()
                    if len(chunk) < 200:
                        continue
                    records.append({"text": chunk, "source": "fiction_books"})
                    written += 1
                    if written % 50_000 == 0:
                        _flush()
                        print(f"  [fiction] {written:,} docs written...", end="\r")
                    if written >= books_target:
                        break
                if written >= books_target:
                    break
        except Exception as e:
            print(f"\n  [fiction] PleIAs/US-PD-Books error: {e}, falling back to TinyStories only")

    _flush()

    # -- TinyStoriesV2 (short-form narrative variety) ----------------------
    stories_target = max_docs
    if written < stories_target:
        print(f"  [fiction] loading TinyStoriesV2 "
              f"(have {written:,}/{stories_target:,})...")
        try:
            ds_stories = load_dataset(
                "roneneldan/TinyStoriesV2",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            for doc in ds_stories:
                text = doc.get("text", "")
                if not text or len(text) < 100:
                    continue
                records.append({"text": text, "source": "fiction_tinystories"})
                written += 1
                if len(records) >= 2000:
                    _flush()
                    print(f"  [fiction] {written:,} docs written...", end="\r")
                if written >= stories_target:
                    break
        except Exception as e:
            print(f"\n  [fiction] TinyStoriesV2 error: {e}")

    _flush()
    print(f"\n  [fiction] done — {written:,} docs")


# ---------------------------------------------------------------------------
# 7. Music lyrics
# ---------------------------------------------------------------------------

def download_lyrics(max_docs: int = 400_000) -> None:
    """Pull song lyrics for poetic structure, rhyme, and creative writing coverage.

    Lyrics teach: rhyme/rhythm patterns, short-form emotional expression,
    verse/chorus structure, and metaphor density — all valuable for D&D NPC
    dialogue, roleplay flavor text, and creative writing in general.

    Uses sebastianpineda/lyrics (~250k songs) as primary source, with
    fallback to mbien/genius_lyrics if available.

    ~400k songs × ~300 tokens avg ≈ ~120M tokens. Small sprinkle but high signal.
    """
    out_path = CORPUS_DIR / "lyrics" / "lyrics.jsonl"
    existing = _count_existing(out_path)
    if existing >= max_docs:
        print(f"  [lyrics] already have {existing:,} docs, skipping")
        return

    print(f"  [lyrics] loading from HuggingFace (have {existing:,}/{max_docs:,})...")

    records: list[dict] = []
    written = existing

    # Primary: sebastianpineda/lyrics
    try:
        ds = load_dataset(
            "sebastianpineda/lyrics",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for doc in ds:
            # Field names vary — try common keys
            lyrics = (doc.get("lyrics") or doc.get("text") or
                      doc.get("lyric") or doc.get("content") or "")
            artist = doc.get("artist") or doc.get("artist_name") or ""
            title  = doc.get("title")  or doc.get("song")         or ""

            if not lyrics or len(lyrics) < 100:
                continue

            # Format with artist/title header so the model learns attribution
            header = ""
            if artist and title:
                header = f"{title} — {artist}\n\n"
            elif title:
                header = f"{title}\n\n"

            text = header + lyrics.strip()
            records.append({"text": text, "source": "lyrics",
                             "artist": artist, "title": title})
            written += 1

            if len(records) >= 2000:
                _write_jsonl(out_path, records)
                records = []
                print(f"  [lyrics] {written:,} songs written...", end="\r")

            if written >= max_docs:
                break

    except Exception as e:
        print(f"\n  [lyrics] sebastianpineda/lyrics error: {e}, trying fallback...")

        # Fallback: mbien/genius_lyrics
        try:
            ds2 = load_dataset(
                "mbien/genius_lyrics",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            for doc in ds2:
                lyrics = doc.get("lyrics") or doc.get("text") or ""
                if not lyrics or len(lyrics) < 100:
                    continue
                artist = doc.get("artist") or ""
                title  = doc.get("title")  or ""
                header = f"{title} — {artist}\n\n" if artist and title else ""
                records.append({"text": header + lyrics.strip(),
                                 "source": "lyrics"})
                written += 1
                if len(records) >= 2000:
                    _write_jsonl(out_path, records)
                    records = []
                if written >= max_docs:
                    break
        except Exception as e2:
            print(f"\n  [lyrics] fallback also failed: {e2}")

    if records:
        _write_jsonl(out_path, records)

    print(f"\n  [lyrics] done — {written:,} songs")


# ---------------------------------------------------------------------------
# 8. Existing assets (tool_calls + quant SO)
# ---------------------------------------------------------------------------

def convert_existing_assets() -> None:
    """Convert existing fine-tuning datasets to raw pretraining text format."""

    # Tool calls — extract as raw tool-use examples
    tool_calls_path = Path("/home/kmbandy/mad-lab-mcp/datasets/tool_calls.jsonl")
    out_path = CORPUS_DIR / "existing" / "tool_calls_raw.jsonl"

    if tool_calls_path.exists() and not out_path.exists():
        print("  [existing] converting tool_calls.jsonl...")
        records = []
        with tool_calls_path.open() as f:
            for line in f:
                try:
                    d    = json.loads(line)
                    turns = d.get("conversations", [])
                    text  = "\n\n".join(
                        f"[{t['from'].upper()}]: {t['value']}"
                        for t in turns if t.get("value")
                    )
                    if len(text) > 200:
                        records.append({"text": text, "source": "tool_calls"})
                except Exception:
                    pass
        _write_jsonl(out_path, records)
        print(f"  [existing] {len(records):,} tool_call conversations converted")
    elif not tool_calls_path.exists():
        print(f"  [existing] tool_calls.jsonl not found at {tool_calls_path}, skipping")

    # StackExchange + SO extracted datasets — already processed from ZIM files
    se_datasets = {
        "stackoverflow":      Path("/home/kmbandy/mad-lab-mcp/datasets/stackoverflow_accepted.jsonl"),
        "math_se":            Path("/home/kmbandy/mad-lab-mcp/datasets/math_accepted.jsonl"),
        "stats_se":           Path("/home/kmbandy/mad-lab-mcp/datasets/stats_accepted.jsonl"),
        "softwareeng_se":     Path("/home/kmbandy/mad-lab-mcp/datasets/softwareengineering_accepted.jsonl"),
        "rpg_se":             Path("/home/kmbandy/mad-lab-mcp/datasets/rpg_accepted.jsonl"),
        "quant_so":           Path("/home/kmbandy/mad-lab-mcp/datasets/quant_train.jsonl"),
        "quant_so_eval":      Path("/home/kmbandy/mad-lab-mcp/datasets/quant_so_accepted.jsonl"),
    }

    for se_name, se_path in se_datasets.items():
        out_se = CORPUS_DIR / "existing" / f"{se_name}_raw.jsonl"
        if out_se.exists():
            print(f"  [existing] {se_name} already converted, skipping")
            continue
        if not se_path.exists():
            print(f"  [existing] {se_name} not found at {se_path}, skipping")
            continue
        records = []
        with se_path.open() as f:
            for line in f:
                try:
                    doc   = json.loads(line)
                    convs = doc.get("conversations", [])
                    text  = "\n\n".join(
                        f"[{t['from'].upper()}]: {t['value']}"
                        for t in convs if t.get("value")
                    )
                    if len(text) > 200:
                        records.append({"text": text, "source": se_name})
                except Exception:
                    pass
        if records:
            _write_jsonl(out_se, records)
            print(f"  [existing] {se_name}: {len(records):,} records converted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Mad-lab 1B base pretraining corpus downloader")
    print("Target: ~80B raw tokens (process_corpus.py cycles to 100B)")
    print("=" * 60)

    # ── Local sources first (fast, no OOM risk) ──────────────────────────
    print("\n[1/9] Converting existing assets (local disk)...")
    convert_existing_assets()

    print("\n[2/9] Wikipedia (local ZIM)...")
    download_wikipedia()

    print("\n[3/9] StackExchange ZIMs (SO, Math, Stats, Quant, RPG, SoftEng)...")
    download_stackexchange()

    print("\n[4/9] SEC EDGAR filings (10-K, 10-Q, 8-K)...")
    download_edgar()

    # ── Network sources (may OOM — local data already safe above) ─────────
    print("\n[5/9] arXiv (peS2o via HuggingFace)...")
    download_arxiv()

    print("\n[6/9] FineWeb-Edu (score >= 4, sample-100BT)...")
    download_fineweb()

    print("\n[7/9] GitHub code (Python + R)...")
    download_github_code()

    print("\n[8/9] Fiction / creative writing...")
    download_fiction()

    print("\n[9/9] Music lyrics...")
    download_lyrics()

    # Summary
    print("\n" + "=" * 60)
    print("Corpus summary:")
    sources = ["arxiv", "edgar", "wikipedia", "stackexchange", "fineweb",
               "github_code", "fiction", "lyrics", "existing"]
    total = 0
    for src in sources:
        src_dir = CORPUS_DIR / src
        count   = (
            sum(_count_existing(p) for p in src_dir.glob("*.jsonl"))
            if src_dir.exists() else 0
        )
        total  += count
        print(f"  {src:14}: {count:>12,} documents")
    print(f"  {'TOTAL':14}: {total:>12,} documents")
    print("\nNext: run process_corpus.py to tokenize + pack + upload to S3")


if __name__ == "__main__":
    main()
