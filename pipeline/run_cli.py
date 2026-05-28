"""Headless one-shot runner: execute a Run config's stages without the HTTP API.

Usage: python -m pipeline.run_cli --config /path/run.json
Run JSON shape (subset of schemas.RunCreate):
    {"name": "...", "stages": [{"stage_type": "pretrain", "config": {...}}, ...]}
"""
import argparse
import asyncio
import json
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from pipeline.orchestrator import _make_executor


async def _run(run_cfg: dict) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    run_id = uuid.uuid4()
    async with SessionLocal() as db:
        for stage in run_cfg["stages"]:
            stype = stage["stage_type"]
            cfg = stage["config"]
            executor = _make_executor(stype, run_id, uuid.uuid4(), cfg, db)
            print(f"[run_cli] starting stage={stype}", flush=True)
            result = await executor.run()
            print(f"[run_cli] stage={stype} result={result}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    run_cfg = json.loads(args.config.read_text())
    asyncio.run(_run(run_cfg))


if __name__ == "__main__":
    main()
