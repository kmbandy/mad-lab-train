"""Prune executor — structural and unstructured pruning.

Methods:
  wanda      — unstructured: weight × activation-norm scoring, zero out lowest N%
  shortgpt   — structural: remove layers with lowest Block Influence score
  llm_pruner — structural: gradient-based attention head / MLP channel removal (stub v1)
  slicegpt   — structural: PCA weight matrix compression (stub v1)

Always operates on full-precision SafeTensors. Output is a pruned SafeTensors
directory ready for a healing finetune.
"""
import asyncio
import json
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor, WeightPager


class PruneExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        method = cfg.get("method", "wanda")
        pruning_ratio = float(cfg.get("pruning_ratio", 0.2))
        model_source = cfg.get("model_source", "huggingface")

        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        out_dir = run_datasets_dir / "prune"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve model path ─────────────────────────────────────────────────
        if model_source == "huggingface":
            model_path = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _download_hf_model(cfg["model_id"], out_dir)
            )
        else:
            if cfg.get("model_path"):
                model_path = Path(os.path.expanduser(cfg["model_path"]))
            else:
                model_path = _resolve_upstream_model(run_datasets_dir)

        # ── Resolve calibration dataset ───────────────────────────────────────
        if cfg.get("calibration_dataset"):
            cal_path = Path(os.path.expanduser(cfg["calibration_dataset"]))
        else:
            cal_path = run_datasets_dir / "calibration.jsonl"
            if not cal_path.exists():
                cal_path = run_datasets_dir / "train.jsonl"

        await self.emit_event("stage_started", {
            "stage_type": "prune",
            "method": method,
            "pruning_ratio": pruning_ratio,
        }, stage_type="prune")

        loop = asyncio.get_event_loop()
        executor_ref = self

        def _emit_sync(event_type: str, data: dict) -> None:
            asyncio.run_coroutine_threadsafe(
                executor_ref.emit_event(event_type, data, stage_type="prune"),
                loop,
            )

        def _do_prune() -> None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()

            cal_texts = _load_cal_texts(cal_path, max_samples=128)

            _emit_sync("importance_scored", {"method": method})

            if method == "wanda":
                _prune_wanda(model, tokenizer, cal_texts, pruning_ratio, _emit_sync)
            elif method == "shortgpt":
                _prune_shortgpt(model, tokenizer, cal_texts, pruning_ratio, _emit_sync)
            elif method == "llm_pruner":
                _prune_llm_pruner(cfg, model, tokenizer, cal_texts, pruning_ratio, _emit_sync)
            elif method == "slicegpt":
                raise NotImplementedError(
                    "SliceGPT is not yet implemented in v1. "
                    "Use wanda or shortgpt."
                )
            else:
                raise ValueError(f"Unknown pruning method: {method}")

            # Save pruned model
            model.save_pretrained(str(out_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(out_dir))

        await loop.run_in_executor(None, _do_prune)

        if self._force_pause or self._pause_requested:
            return None

        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True


# ── Pruning methods ───────────────────────────────────────────────────────────

def _prune_wanda(model, tokenizer, cal_texts, pruning_ratio, emit_sync) -> None:
    """Unstructured Wanda: zero out weights with lowest |W| × ||X||₂ score."""
    import torch

    activation_cache: dict[str, list] = {}
    hooks = []

    def _make_hook(name):
        def hook(_module, inp, _out):
            activation_cache.setdefault(name, []).append(inp[0].detach().float().cpu())
        return hook

    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))
            linear_names.append(name)

    # Run calibration forward passes
    with torch.no_grad():
        for text in cal_texts[:64]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            ids = {k: v.to(device) for k, v in ids.items()}
            try:
                model(**ids)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    total = len(linear_names)
    for idx, (name, module) in enumerate(
        (n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    ):
        acts = activation_cache.get(name, [])
        W = module.weight.data.float()

        if acts:
            X = torch.cat(acts, dim=0)
            if X.dim() == 3:
                X = X.reshape(-1, X.shape[-1])
            act_norm = X.norm(dim=0).to(W.device)
            importance = W.abs() * act_norm.unsqueeze(0)
        else:
            importance = W.abs()

        flat = importance.flatten()
        k = max(1, int(flat.numel() * pruning_ratio))
        threshold = torch.topk(flat, k, largest=False).values.max()
        mask = (importance > threshold).to(module.weight.dtype)
        module.weight.data = (W * mask).to(module.weight.dtype)

        params_removed = int((mask == 0).sum().item())
        emit_sync("layer_pruned", {
            "layer_idx": idx + 1,
            "total_layers": total,
            "params_removed": params_removed,
        })


def _prune_shortgpt(model, tokenizer, cal_texts, pruning_ratio, emit_sync) -> None:
    """ShortGPT: remove layers with lowest Block Influence (BI) score.

    BI_l = 1 − mean(cos_sim(hidden_in_l, hidden_out_l))
    Layers with BI ≈ 0 are near-identity transforms — safe to remove.
    """
    import torch
    import torch.nn.functional as F

    layers = _get_model_layers(model)
    n_layers = len(layers)
    n_remove = max(1, int(n_layers * pruning_ratio))

    # Capture (input, output) hidden states per layer
    layer_inputs: dict[int, list] = {}
    layer_outputs: dict[int, list] = {}
    hooks = []

    def _make_hooks(idx):
        def pre_hook(_module, inp):
            layer_inputs.setdefault(idx, []).append(inp[0].detach().float().cpu())
        def post_hook(_module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            layer_outputs.setdefault(idx, []).append(hidden.detach().float().cpu())
        return pre_hook, post_hook

    for i, layer in enumerate(layers):
        pre, post = _make_hooks(i)
        hooks.append(layer.register_forward_pre_hook(pre))
        hooks.append(layer.register_forward_hook(post))

    with torch.no_grad():
        for text in cal_texts[:32]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            ids = {k: v.to(device) for k, v in ids.items()}
            try:
                model(**ids)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    # Compute BI per layer
    bi_scores: list[float] = []
    for i in range(n_layers):
        ins = layer_inputs.get(i, [])
        outs = layer_outputs.get(i, [])
        if not ins or not outs:
            bi_scores.append(1.0)  # unknown → keep
            continue
        h_in = torch.cat(ins, dim=0).reshape(-1, ins[0].shape[-1])
        h_out = torch.cat(outs, dim=0).reshape(-1, outs[0].shape[-1])
        min_len = min(h_in.shape[0], h_out.shape[0])
        cos = F.cosine_similarity(h_in[:min_len], h_out[:min_len], dim=-1)
        bi_scores.append(float(1.0 - cos.mean().item()))

    emit_sync("importance_scored", {
        "method": "shortgpt",
        "bi_scores": [round(s, 4) for s in bi_scores],
    })

    # Remove layers with lowest BI (most transparent)
    remove_indices = set(
        sorted(range(n_layers), key=lambda i: bi_scores[i])[:n_remove]
    )
    kept_layers = [l for i, l in enumerate(layers) if i not in remove_indices]

    import torch.nn as nn
    _set_model_layers(model, nn.ModuleList(kept_layers))

    # Update config to reflect new num_hidden_layers
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(kept_layers)

    for removed_idx in sorted(remove_indices):
        emit_sync("layer_pruned", {
            "layer_idx": removed_idx,
            "total_layers": n_layers,
            "params_removed": 0,  # structural removal — all params in layer
        })


def _prune_llm_pruner(cfg, model, tokenizer, cal_texts, pruning_ratio, emit_sync) -> None:
    """LLM-Pruner: gradient-based structural pruning (block_wise mode).

    WeightPager slot reserved for future NVMe→VRAM weight routing integration.
    v1: raises NotImplementedError — install llm-pruner from GitHub to enable.
    """
    lp_cfg = cfg.get("llm_pruner", {})
    weight_pager_cls: str | None = lp_cfg.get("weight_pager")

    if weight_pager_cls:
        # WeightPager protocol slot — instantiate if provided, otherwise skip
        try:
            import importlib
            module_path, cls_name = weight_pager_cls.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            pager: WeightPager = getattr(mod, cls_name)()
            emit_sync("weight_pager_attached", {"class": weight_pager_cls})
        except Exception as e:
            emit_sync("weight_pager_failed", {"error": str(e)})

    raise NotImplementedError(
        "LLM-Pruner is not bundled in v1. "
        "Install from https://github.com/horseee/LLM-Pruner and integrate via llm_pruner.weight_pager. "
        "Use method: wanda or method: shortgpt for now."
    )


# ── Architecture helpers ──────────────────────────────────────────────────────

def _get_model_layers(model):
    """Return the ModuleList of transformer decoder layers."""
    for attr_path in [
        "model.layers",       # LLaMA, Mistral, Qwen2, Gemma
        "transformer.h",      # GPT-2, Falcon
        "model.h",
        "gpt_neox.layers",    # GPT-NeoX
        "bert.encoder.layer", # BERT-family
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__iter__"):
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot find transformer layers — unsupported architecture for ShortGPT pruning"
    )


def _set_model_layers(model, new_layers) -> None:
    """Replace the ModuleList of transformer decoder layers."""
    for attr_path in [
        "model.layers",
        "transformer.h",
        "model.h",
        "gpt_neox.layers",
        "bert.encoder.layer",
    ]:
        parts = attr_path.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], new_layers)
                return
        except AttributeError:
            continue


# ── Dataset / model loading helpers ──────────────────────────────────────────

def _load_cal_texts(cal_path: Path, max_samples: int = 128) -> list[str]:
    texts = []
    try:
        with open(cal_path) as f:
            for line in f:
                if len(texts) >= max_samples:
                    break
                try:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    text = " ".join(m.get("content", "") for m in msgs if m.get("content"))
                    if text.strip():
                        texts.append(text)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return texts


def _resolve_upstream_model(run_datasets_dir: Path) -> Path:
    for subdir in ("merge", "finetune", "pretrain"):
        candidate = run_datasets_dir / subdir
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "prune.model_path not set and no upstream executor output found"
    )


def _download_hf_model(model_id: str, out_dir: Path) -> Path:
    """Download HF model to a local cache dir. Returns local path."""
    from huggingface_hub import snapshot_download
    from pipeline.executors.upload import _get_hf_token

    token = _get_hf_token()
    cache_dir = out_dir.parent / "_hf_cache"
    local = snapshot_download(
        repo_id=model_id,
        token=token,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    return Path(local)
