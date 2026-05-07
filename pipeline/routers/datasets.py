import asyncio
import os
from pathlib import Path

from fastapi import APIRouter

from pipeline.settings import settings

router = APIRouter(prefix="/datasets", tags=["datasets"])

_sync_state: dict = {"status": "idle", "last_synced": None, "error": None, "synced_count": 0}


def _do_sync() -> dict:
    import duckdb

    token = os.getenv("MOTHERDUCK_TOKEN", "")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set in pipeline service environment")

    db_path = Path(os.path.expanduser(settings.log_dir)).parent / "datasets.db"
    if not db_path.exists():
        raise RuntimeError("datasets.db not found — run a dataset_prep stage first")

    os.environ["motherduck_token"] = token
    con = duckdb.connect("md:mad-lab")

    con.execute(f"ATTACH '{db_path}' AS local (READ_ONLY)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS records AS
        SELECT * FROM local.records LIMIT 0
    """)
    result = con.execute("""
        INSERT OR IGNORE INTO records
        SELECT * FROM local.records
    """)
    synced = result.rowcount if result else 0
    con.close()
    return {"synced": synced}


@router.post("/sync")
async def sync_datasets():
    """Sync local datasets.db to MotherDuck mad-lab database."""
    if _sync_state["status"] == "running":
        return {"status": "running", "message": "Sync already in progress"}

    _sync_state["status"] = "running"
    _sync_state["error"] = None

    async def _run():
        from datetime import datetime, timezone
        try:
            result = await asyncio.to_thread(_do_sync)
            _sync_state["status"] = "idle"
            _sync_state["last_synced"] = datetime.now(timezone.utc).isoformat()
            _sync_state["synced_count"] = result["synced"]
        except Exception as e:
            _sync_state["status"] = "error"
            _sync_state["error"] = str(e)

    asyncio.create_task(_run())
    return {"status": "started"}


@router.get("/sync/status")
async def sync_status():
    return _sync_state
