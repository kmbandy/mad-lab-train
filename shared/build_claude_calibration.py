"""Build a calibration / SFT jsonl from Claude Code transcripts (tool-interleaved turns).

Drives ``dataset_prep._iter_claude_jsonl`` with ``include_tools`` WITHOUT the
DB-bound executor harness — a one-shot, dependency-light corpus dump (no postgres,
no DuckDB). Uses the exact same battle-tested parsing + ``conversation_messages``
renderer the pipeline uses, so output matches what the full pipeline would emit.

Output: jsonl of ``{"messages": [...]}`` — ready for ml8 calibration (the
calib_corpus messages source) or SFT. Reusable: point --path at any Claude
session directory.

    python shared/build_claude_calibration.py \
        --path ~/.claude/projects --out /home/kmbandy/models/calib_sources/claude_traces.jsonl \
        --unit turn --min-turn-chars 200
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from pipeline.executors.dataset_prep import _iter_claude_jsonl


async def _build(args) -> None:
    cfg = {
        "type": "claude_jsonl",
        "path": args.path,
        "recursive": True,
        "unit": args.unit,
        "include_tools": True,
        "min_turn_chars": args.min_turn_chars,
        "agent": "claude-code",
        "trace_source": "organic_work",
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n = 0
    approx_tokens = 0
    with open(out_path, "w") as f:
        async for rec in _iter_claude_jsonl(cfg, args.max_records):
            msgs = rec.get("messages") or []
            if not msgs:
                continue
            blob = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(blob.encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
            approx_tokens += sum(len(str(m.get("content", ""))) for m in msgs) // 4
            if n % 500 == 0:
                print(f"[build-claude-calib] {n} records / ~{approx_tokens} tok", flush=True)
    print(f"[build-claude-calib] DONE {n} records / ~{approx_tokens} tok -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="~/.claude/projects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", choices=["turn", "session"], default="turn")
    ap.add_argument("--min-turn-chars", type=int, default=200)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(_build(args))


if __name__ == "__main__":
    main()
