from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.db import get_db

router = APIRouter(tags=["system"])

VERSION = "2.0.0"


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"
    return {"status": "ok", "version": VERSION, "db": db_status}


@router.get("/hardware")
async def hardware() -> dict:
    import psutil

    cpu_pct = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    gpus = _gpu_stats()

    return {
        "gpus": gpus,
        "cpu_pct": cpu_pct,
        "ram_used_gb": round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
    }


def _gpu_stats() -> list[dict]:
    # Try ROCm (AMD) first via rocm-smi, then NVIDIA via pynvml
    gpus = _rocm_stats()
    if gpus:
        return gpus
    return _nvml_stats()


def _rocm_stats() -> list[dict]:
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
        import json
        data = json.loads(out.stdout)
        result = []
        for i, (card, info) in enumerate(data.items()):
            used = int(info.get("VRAM Total Used Memory (B)", 0))
            total = int(info.get("VRAM Total Memory (B)", 1))
            util = int(info.get("GPU use (%)", 0))
            result.append({
                "index": i,
                "name": card,
                "vram_used_gb": round(used / 1e9, 2),
                "vram_total_gb": round(total / 1e9, 2),
                "utilization_pct": util,
            })
        return result
    except Exception:
        return []


def _nvml_stats() -> list[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "vram_used_gb": round(mem.used / 1e9, 2),
                "vram_total_gb": round(mem.total / 1e9, 2),
                "utilization_pct": util.gpu,
            })
        return result
    except Exception:
        return []
