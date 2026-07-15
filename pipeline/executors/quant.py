"""Quantization executor — SafeTensors → GGUF via llama.cpp CLI.

Pipeline:
  1. Convert HF SafeTensors to F16 GGUF (llama-convert-hf-to-gguf)
  2. Optional: generate importance matrix (llama-imatrix)
  3. For each quant_type: quantize F16 GGUF (llama-quantize)

Each completed quant_type is a checkpoint boundary — resume skips finished types.
"""
import asyncio
import json
import os
import re
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class QuantExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        out_dir = run_datasets_dir / "quant"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_model_path(cfg, run_datasets_dir)
        quant_types = cfg.get("quant_types") or ["Q4_K_M"]
        use_imatrix = bool(cfg.get("imatrix", False))
        imatrix_dataset = _resolve_imatrix_dataset(cfg, run_datasets_dir) if use_imatrix else None
        output_prefix = cfg.get("output_prefix") or str(self.run_id)[:8]

        # Load checkpoint — track completed quant types
        checkpoint = _load_checkpoint(out_dir, cfg.get("_resume_artifact"))
        completed_types: set[str] = set(checkpoint.get("completed_types", []))
        f16_path_saved: str | None = checkpoint.get("f16_path")
        imatrix_path_saved: str | None = checkpoint.get("imatrix_path")

        await self.emit_event("stage_started", {
            "stage_type": "quant",
            "quant_types": quant_types,
            "imatrix": use_imatrix,
        }, stage_type="quant")

        # ── Step 1: Convert to F16 GGUF ───────────────────────────────────────
        if f16_path_saved and Path(f16_path_saved).exists():
            f16_path = Path(f16_path_saved)
        else:
            await self.emit_event("convert_started", {
                "model_path": str(model_path),
            }, stage_type="quant")
            f16_path = out_dir / f"{output_prefix}-f16.gguf"
            await self._convert_to_f16(model_path, f16_path)
            checkpoint["f16_path"] = str(f16_path)
            _save_checkpoint(out_dir, checkpoint)

        if self._force_pause or self._pause_requested:
            return None

        # ── Step 2: Optional imatrix ───────────────────────────────────────────
        if use_imatrix:
            if imatrix_path_saved and Path(imatrix_path_saved).exists():
                imatrix_path = Path(imatrix_path_saved)
            else:
                if not imatrix_dataset:
                    raise RuntimeError(
                        "imatrix=true requires a calibration dataset "
                        "(dataset_prep stage or explicit imatrix_dataset path)"
                    )
                imatrix_path = out_dir / f"{output_prefix}-imatrix.dat"
                await self._generate_imatrix(f16_path, imatrix_dataset, imatrix_path)
                checkpoint["imatrix_path"] = str(imatrix_path)
                _save_checkpoint(out_dir, checkpoint)

            if self._force_pause or self._pause_requested:
                return None
        else:
            imatrix_path = None

        # ── Step 3: Quantize each type ────────────────────────────────────────
        output_paths: list[str] = []
        for qtype in quant_types:
            if self._force_pause or self._pause_requested:
                break
            if qtype in completed_types:
                out_path = out_dir / f"{output_prefix}-{qtype.lower()}.gguf"
                output_paths.append(str(out_path))
                continue

            await self.emit_event("quant_started", {
                "quant_type": qtype,
            }, stage_type="quant")

            out_path = out_dir / f"{output_prefix}-{qtype.lower()}.gguf"
            await self._quantize(f16_path, out_path, qtype, imatrix_path)

            size_gb = out_path.stat().st_size / 1e9 if out_path.exists() else 0.0
            await self.emit_event("quant_complete", {
                "quant_type": qtype,
                "output_path": str(out_path),
                "size_gb": round(size_gb, 3),
            }, stage_type="quant")

            completed_types.add(qtype)
            checkpoint["completed_types"] = list(completed_types)
            checkpoint_path = _save_checkpoint(out_dir, checkpoint, len(completed_types))
            await self.record_checkpoint(
                len(completed_types),
                str(checkpoint_path),
                {
                    "quant_types_completed": sorted(completed_types),
                    "current_quant_type": qtype,
                },
            )
            output_paths.append(str(out_path))

        if self._force_pause or self._pause_requested:
            return None

        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True

    # ── Subprocess helpers ────────────────────────────────────────────────────

    async def _convert_to_f16(self, model_path: Path, out_path: Path) -> None:
        """Convert HF SafeTensors directory to F16 GGUF."""
        cmd = _find_convert_cmd()
        args = [*cmd, "--outtype", "f16", "--outfile", str(out_path), str(model_path)]
        await self._run_subprocess(args, "convert")

    async def _generate_imatrix(
        self, f16_path: Path, calibration_path: Path, out_path: Path
    ) -> None:
        """Generate importance matrix from calibration dataset."""
        # llama-imatrix expects plain text; extract content from JSONL
        text_path = out_path.parent / f"{out_path.stem}-cal.txt"
        _extract_text_from_jsonl(calibration_path, text_path)

        args = [
            "llama-imatrix",
            "-m", str(f16_path),
            "-f", str(text_path),
            "-o", str(out_path),
            "--chunks", "128",
        ]
        await self._run_subprocess(args, "imatrix")

    async def _quantize(
        self,
        f16_path: Path,
        out_path: Path,
        qtype: str,
        imatrix_path: Path | None,
    ) -> None:
        """Run llama-quantize, streaming progress events."""
        args = ["llama-quantize"]
        if imatrix_path:
            args += ["--imatrix", str(imatrix_path)]
        args += [str(f16_path), str(out_path), qtype]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        tensor_re = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")
        assert proc.stdout is not None
        async for line in proc.stdout:
            if self._force_pause:
                proc.kill()
                break
            text = line.decode(errors="replace")
            m = tensor_re.search(text)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                pct = round(current / total * 100, 1) if total else 0.0
                await self.emit_event("quant_progress", {
                    "quant_type": qtype,
                    "percent": pct,
                }, stage_type="quant")

        await proc.wait()
        if proc.returncode not in (0, None) and not self._force_pause:
            raise RuntimeError(f"llama-quantize failed (exit {proc.returncode}) for {qtype}")

    async def _run_subprocess(self, args: list[str], step: str) -> None:
        """Run a subprocess to completion, raising on non-zero exit."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for _ in proc.stdout:
            if self._force_pause:
                proc.kill()
                return
        await proc.wait()
        if proc.returncode not in (0, None) and not self._force_pause:
            raise RuntimeError(f"{step} step failed (exit {proc.returncode})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model_path(cfg: dict, run_datasets_dir: Path) -> Path:
    """Explicit config path overrides auto-wiring from finetune/merge/pretrain output."""
    if cfg.get("model_path"):
        return Path(os.path.expanduser(cfg["model_path"]))
    # Auto-wire: finetune adapter-merged output, then pretrain, then merge
    for subdir in ("merge", "finetune", "pretrain"):
        candidate = run_datasets_dir / subdir
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "quant.model_path not set and no upstream finetune/merge/pretrain output found"
    )


def _resolve_imatrix_dataset(cfg: dict, run_datasets_dir: Path) -> Path | None:
    if cfg.get("imatrix_dataset"):
        return Path(os.path.expanduser(cfg["imatrix_dataset"]))
    # Auto-wire from dataset_prep calibration output
    candidate = run_datasets_dir / "calibration.jsonl"
    return candidate if candidate.exists() else None


def _find_convert_cmd() -> list[str]:
    """Find the llama.cpp HF-to-GGUF converter — binary first, then Python script."""
    import shutil
    if shutil.which("llama-convert-hf-to-gguf"):
        return ["llama-convert-hf-to-gguf"]
    # Fall back to convert_hf_to_gguf.py alongside the llama-quantize binary
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin:
        script = Path(quantize_bin).parent / "convert_hf_to_gguf.py"
        if script.exists():
            return ["python3", str(script)]
    raise RuntimeError(
        "Cannot find llama-convert-hf-to-gguf or convert_hf_to_gguf.py. "
        "Is llama.cpp installed?"
    )


def _extract_text_from_jsonl(jsonl_path: Path, text_path: Path) -> None:
    """Write plain text lines from a ChatML JSONL for llama-imatrix calibration."""
    with open(jsonl_path) as fin, open(text_path, "w") as fout:
        for line in fin:
            try:
                record = json.loads(line)
                for msg in record.get("messages", []):
                    content = msg.get("content", "").strip()
                    if content:
                        fout.write(content + "\n")
            except Exception:
                pass


def _load_checkpoint(out_dir: Path, resume_artifact: str | None = None) -> dict:
    cp = Path(resume_artifact) if resume_artifact else out_dir / ".checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(out_dir: Path, data: dict, sequence: int | None = None) -> Path:
    (out_dir / ".checkpoint.json").write_text(json.dumps(data))
    if sequence is None:
        return out_dir / ".checkpoint.json"
    snapshot = out_dir / "checkpoints" / f"checkpoint-{sequence}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(data))
    return snapshot
