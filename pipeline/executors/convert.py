"""Convert executor — model format conversion without quantization.

Formats: gguf_f16, gguf_bf16, safetensors_fp32.
Auto-wires input from upstream finetune/merge/pretrain output.
"""
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor
from pipeline.executors.quant import _find_convert_cmd


class ConvertExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)

    async def run(self) -> str | None:
        import asyncio
        from pipeline.settings import settings

        cfg = self.config
        fmt = cfg.get("format", "gguf_f16")
        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        out_dir = run_datasets_dir / "convert"
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = _resolve_input(cfg, run_datasets_dir)
        out_name = input_path.stem if input_path.is_file() else input_path.name
        out_file = out_dir / f"{out_name}-{fmt.replace('_', '-')}.gguf"

        await self.emit_event("convert_started", {
            "input_path": str(input_path),
            "format": fmt,
        }, stage_type="convert")

        if fmt in ("gguf_f16", "gguf_bf16"):
            outtype = "f16" if fmt == "gguf_f16" else "bf16"
            cmd = _find_convert_cmd()
            args = [*cmd, "--outtype", outtype, "--outfile", str(out_file), str(input_path)]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Convert failed (exit {proc.returncode}): {stderr.decode()[:400]}"
                )

        elif fmt == "safetensors_fp32":
            out_file = out_dir / out_name  # directory output
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: _upcast_to_fp32(input_path, out_file)
            )

        else:
            raise ValueError(f"Unknown format: {fmt}")

        size_gb = (
            out_file.stat().st_size / 1e9
            if out_file.is_file()
            else sum(f.stat().st_size for f in out_file.rglob("*") if f.is_file()) / 1e9
        )
        await self.emit_event("convert_complete", {
            "output_path": str(out_file),
            "format": fmt,
            "size_gb": round(size_gb, 3),
        }, stage_type="convert")

        return str(out_file)

    async def pause(self) -> None:
        pass  # convert is atomic — no mid-operation pause

    async def force_pause(self) -> None:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_input(cfg: dict, run_datasets_dir: Path) -> Path:
    if cfg.get("input_path"):
        return Path(os.path.expanduser(cfg["input_path"]))
    for subdir in ("merge", "finetune", "pretrain"):
        candidate = run_datasets_dir / subdir
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "convert.input_path not set and no upstream finetune/merge/pretrain output found"
    )


def _upcast_to_fp32(input_path: Path, out_path: Path) -> None:
    """Load model in native dtype and re-save in fp32 SafeTensors."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(input_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(input_path),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
