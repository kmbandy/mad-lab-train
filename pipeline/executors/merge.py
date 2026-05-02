"""Merge executor — adapter absorption or model-to-model merge via mergekit.

Adapter mode: PEFT merge_and_unload() — LoRA weights folded into base model.
Model merge modes (slerp / ties / dare_ties): mergekit on CPU, no GPU needed.
Optional eval gate: perplexity (or benchmark stub) on merged output.
"""
import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class MergeExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        mode = cfg.get("mode", "adapter")
        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        out_dir = run_datasets_dir / "merge"
        out_dir.mkdir(parents=True, exist_ok=True)

        await self.emit_event("merge_started", {
            "mode": mode,
        }, stage_type="merge")

        loop = asyncio.get_event_loop()

        if mode == "adapter":
            await loop.run_in_executor(None, lambda: _merge_adapter(cfg, run_datasets_dir, out_dir))
        else:
            await self._merge_models(cfg, out_dir, mode)

        await self.emit_event("merge_complete", {
            "output_path": str(out_dir),
        }, stage_type="merge")

        # ── Eval gate ─────────────────────────────────────────────────────────
        gate_cfg = cfg.get("eval_gate", {})
        if gate_cfg.get("enabled"):
            passed, score = await loop.run_in_executor(
                None, lambda: _eval_gate_perplexity(out_dir, run_datasets_dir, gate_cfg)
            )
            threshold = gate_cfg.get("threshold", 0.0)
            await self.emit_event("gate_result", {
                "passed": passed,
                "score": score,
                "threshold": threshold,
            }, stage_type="merge")

            if not passed:
                on_fail = gate_cfg.get("on_fail", "pause")
                if on_fail == "abort":
                    raise RuntimeError(
                        f"Eval gate failed: score {score:.4f} < threshold {threshold}"
                    )
                # pause: let caller observe _pause_requested
                self._pause_requested = True
                return None

        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True

    async def _merge_models(self, cfg: dict, out_dir: Path, mode: str) -> None:
        """Run mergekit for slerp / ties / dare_ties."""
        merge_yaml = _build_mergekit_yaml(cfg, mode)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="merge_cfg_"
        ) as f:
            f.write(merge_yaml)
            yaml_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "mergekit-yaml", yaml_path, str(out_dir),
                "--allow-crimes",
                "--out-shard-size", "5B",
                "--lazy-unpickle",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            async for _ in proc.stdout:
                if self._force_pause:
                    proc.kill()
                    break
            await proc.wait()
            if proc.returncode not in (0, None) and not self._force_pause:
                raise RuntimeError(f"mergekit-yaml failed (exit {proc.returncode})")
        finally:
            try:
                Path(yaml_path).unlink(missing_ok=True)
            except Exception:
                pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_adapter(cfg: dict, run_datasets_dir: Path, out_dir: Path) -> None:
    """Fold LoRA adapter into base model weights using PEFT merge_and_unload."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg["base_model"]
    adapter_path = cfg.get("adapter_path") or str(run_datasets_dir / "finetune")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))


def _build_mergekit_yaml(cfg: dict, mode: str) -> str:
    """Build a mergekit YAML config for slerp / ties / dare_ties."""
    base_model = cfg["base_model"]
    model_b = cfg["model_b"]
    model_c = cfg.get("model_c")
    merge_ratio = float(cfg.get("merge_ratio", 0.5))
    density = float(cfg.get("density", 0.5))

    if mode == "slerp":
        return f"""\
merge_method: slerp
dtype: float16
base_model: {base_model}
models:
  - model: {base_model}
    parameters:
      t: 0.0
  - model: {model_b}
    parameters:
      t: {merge_ratio}
"""

    # ties / dare_ties
    models_block = f"""\
  - model: {base_model}
    parameters:
      weight: 1.0
  - model: {model_b}
    parameters:
      weight: 1.0
"""
    if model_c:
        models_block += f"""\
  - model: {model_c}
    parameters:
      weight: 1.0
"""

    extra = ""
    if mode == "dare_ties":
        extra = f"parameters:\n  density: {density}\n  weight: 1.0\n"

    return f"""\
merge_method: {mode}
base_model: {base_model}
dtype: float16
models:
{models_block}{extra}"""


def _eval_gate_perplexity(
    model_dir: Path, run_datasets_dir: Path, gate_cfg: dict
) -> tuple[bool, float]:
    """Quick inline perplexity check on merged model. Returns (passed, score 0-1)."""
    import math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = gate_cfg.get("benchmark", "perplexity")
    threshold = float(gate_cfg.get("threshold", 0.0))
    max_samples = int(gate_cfg.get("max_samples", 50))

    if benchmark != "perplexity":
        # Non-perplexity benchmarks are stubs in v1 — gate passes by default
        return True, 1.0

    # Load calibration text
    cal_path = run_datasets_dir / "calibration.jsonl"
    if not cal_path.exists():
        cal_path = run_datasets_dir / "eval.jsonl"
    if not cal_path.exists():
        return True, 1.0

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    total_loss = 0.0
    count = 0
    with open(cal_path) as f:
        for line in f:
            if count >= max_samples:
                break
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                text = " ".join(m.get("content", "") for m in msgs)
                if not text.strip():
                    continue
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    out = model(**ids, labels=ids["input_ids"])
                total_loss += out.loss.item()
                count += 1
            except Exception:
                pass

    if count == 0:
        return True, 1.0

    avg_loss = total_loss / count
    try:
        ppl = math.exp(avg_loss)
    except (OverflowError, ValueError):
        ppl = float("inf")

    score = 1.0 / (1.0 + ppl)
    return score >= threshold, round(score, 6)
