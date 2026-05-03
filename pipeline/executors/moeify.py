"""MoEify executor — dense-to-MoE upcycling via sparse weight replication.

Supported output architectures:
  LlamaForCausalLM / MistralForCausalLM  →  MixtralForCausalLM
  Qwen2ForCausalLM                        →  Qwen2MoeForCausalLM

Expert init strategies:
  copy   — every expert starts with the same FFN weights (sparse upcycling paper default)
  split  — intermediate dim partitioned evenly across experts; forces early specialization
  random — Xavier-uniform init; requires post-upcycling fine-tuning to be useful

Router init:
  random  — Xavier uniform (router learns from scratch)
  uniform — all-zeros logits (equal routing at init, smooth loss start)
  svd     — first left singular vector of gate_proj used as a warm-start hint

Output is safetensors + updated config.json, ready for HF inference or a finetune stage.
"""
import asyncio
import json
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class MoEifyExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False

    async def run(self) -> str | None:
        from pipeline.settings import settings

        cfg = self.config
        run_dir = Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        out_dir = cfg.get("output_path") or str(run_dir / "moeify")
        out_dir = Path(os.path.expanduser(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        await self.emit_event("stage_started", {
            "stage_type": "moeify",
            "num_experts": cfg.get("num_experts", 8),
            "num_experts_per_tok": cfg.get("num_experts_per_tok", 2),
            "expert_init": cfg.get("expert_init", "copy"),
        }, stage_type="moeify")

        loop = asyncio.get_event_loop()
        executor_ref = self

        def _emit_sync(event_type: str, data: dict) -> None:
            asyncio.run_coroutine_threadsafe(
                executor_ref.emit_event(event_type, data, stage_type="moeify"),
                loop,
            )

        def _should_stop() -> bool:
            return executor_ref._force_pause or executor_ref._pause_requested

        await loop.run_in_executor(None, lambda: _run_moeify(cfg, out_dir, _emit_sync, _should_stop))

        if self._force_pause or self._pause_requested:
            return None

        await self.emit_event("stage_complete", {"output_path": str(out_dir)}, stage_type="moeify")
        return str(out_dir)

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True


# ── Core upcycling logic ──────────────────────────────────────────────────────

def _run_moeify(cfg: dict, out_dir: Path, emit: callable, should_stop: callable) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer

    base_model = cfg["base_model"]
    num_experts = int(cfg.get("num_experts", 8))
    num_experts_per_tok = int(cfg.get("num_experts_per_tok", 2))
    expert_init = cfg.get("expert_init", "copy")
    router_init = cfg.get("router_init", "random")
    shared_expert = bool(cfg.get("shared_expert", False))
    shared_expert_ratio = float(cfg.get("shared_expert_ratio", 0.25))
    balance_loss_coef = float(cfg.get("balance_loss_coef", 0.01))
    router_z_loss_coef = float(cfg.get("router_z_loss_coef", 0.001))
    gpu_target = cfg.get("gpu_target", "cpu")
    target_layers: list[int] | None = cfg.get("target_layers") or None  # None = all

    hf_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    arch = (getattr(hf_config, "architectures", None) or [""])[0]

    converter = _get_converter(arch)
    if converter is None:
        raise ValueError(
            f"MoEify does not support architecture '{arch}'. "
            "Supported: LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM."
        )

    emit("config_loaded", {"architecture": arch, "model_type": hf_config.model_type})

    # Load weights onto CPU (avoid VRAM pressure for large models)
    device_map = gpu_target if gpu_target not in ("auto", "cpu") else "cpu"
    emit("weights_loading", {"device_map": device_map})

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    hidden_size = hf_config.hidden_size
    intermediate_size = hf_config.intermediate_size
    num_layers = hf_config.num_hidden_layers

    if target_layers is None:
        target_layers = list(range(num_layers))

    moe_intermediate_size = cfg.get("moe_intermediate_size") or (
        intermediate_size if expert_init == "copy" else intermediate_size // num_experts
    )
    moe_intermediate_size = int(moe_intermediate_size)

    emit("upcycling_started", {
        "total_layers": num_layers,
        "target_layer_count": len(target_layers),
        "moe_intermediate_size": moe_intermediate_size,
    })

    new_state_dict: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        layer_idx = _parse_layer_idx(key)
        if layer_idx is None or layer_idx not in target_layers or not converter.is_ffn_key(key):
            new_state_dict[key] = tensor
            continue
        # FFN keys in target layers are replaced per-expert below — skip for now

    # Upcycle each target layer
    for i, layer_idx in enumerate(target_layers):
        if should_stop():
            break

        ffn_weights = converter.extract_ffn(state_dict, layer_idx)
        expert_tensors = _init_experts(ffn_weights, num_experts, moe_intermediate_size, expert_init)
        router_weight = _init_router(ffn_weights, num_experts, hidden_size, router_init)

        converter.write_moe_layer(new_state_dict, layer_idx, expert_tensors, router_weight)

        if shared_expert:
            shared_tensors = _init_shared_expert(ffn_weights, moe_intermediate_size, shared_expert_ratio)
            converter.write_shared_expert(new_state_dict, layer_idx, shared_tensors)

        emit("layer_converted", {
            "layer_idx": layer_idx,
            "done": i + 1,
            "total": len(target_layers),
        })

    if should_stop():
        return

    # Copy non-FFN, non-target-layer weights that weren't already transferred
    for key, tensor in state_dict.items():
        if key not in new_state_dict:
            layer_idx = _parse_layer_idx(key)
            if layer_idx is None or layer_idx not in target_layers or not converter.is_ffn_key(key):
                new_state_dict[key] = tensor

    emit("saving_weights", {"num_tensors": len(new_state_dict)})
    save_file(new_state_dict, str(out_dir / "model.safetensors"))

    # Write updated config
    moe_config = converter.build_config(
        hf_config,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert=shared_expert,
        shared_expert_ratio=shared_expert_ratio,
        balance_loss_coef=balance_loss_coef,
        router_z_loss_coef=router_z_loss_coef,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(moe_config, f, indent=2)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    # Write generation_config stub
    gen_cfg_src = Path(base_model) / "generation_config.json"
    if gen_cfg_src.exists():
        import shutil
        shutil.copy(gen_cfg_src, out_dir / "generation_config.json")


# ── Expert initialization ─────────────────────────────────────────────────────

def _init_experts(
    ffn: dict[str, "torch.Tensor"],
    num_experts: int,
    moe_intermediate_size: int,
    strategy: str,
) -> list[dict[str, "torch.Tensor"]]:
    import torch

    gate = ffn["gate"]    # [intermediate_size, hidden_size]
    up   = ffn["up"]      # [intermediate_size, hidden_size]
    down = ffn["down"]    # [hidden_size, intermediate_size]

    experts = []
    if strategy == "copy":
        for _ in range(num_experts):
            experts.append({
                "gate": gate.clone(),
                "up":   up.clone(),
                "down": down.clone(),
            })
    elif strategy == "split":
        gate_chunks = torch.chunk(gate, num_experts, dim=0)
        up_chunks   = torch.chunk(up,   num_experts, dim=0)
        down_chunks = torch.chunk(down, num_experts, dim=1)
        for g, u, d in zip(gate_chunks, up_chunks, down_chunks):
            experts.append({"gate": g.clone(), "up": u.clone(), "down": d.clone()})
    elif strategy == "random":
        hidden_size = gate.shape[1]
        dtype = gate.dtype
        for _ in range(num_experts):
            g = torch.empty(moe_intermediate_size, hidden_size, dtype=dtype)
            u = torch.empty(moe_intermediate_size, hidden_size, dtype=dtype)
            d = torch.empty(hidden_size, moe_intermediate_size, dtype=dtype)
            torch.nn.init.xavier_uniform_(g.float()).to(dtype)
            torch.nn.init.xavier_uniform_(u.float()).to(dtype)
            torch.nn.init.xavier_uniform_(d.float()).to(dtype)
            experts.append({"gate": g, "up": u, "down": d})
    else:
        raise ValueError(f"Unknown expert_init strategy: {strategy}")

    return experts


def _init_router(
    ffn: dict[str, "torch.Tensor"],
    num_experts: int,
    hidden_size: int,
    strategy: str,
) -> "torch.Tensor":
    import torch

    dtype = ffn["gate"].dtype

    if strategy == "random":
        w = torch.empty(num_experts, hidden_size, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(w)
        return w.to(dtype)

    if strategy == "uniform":
        return torch.zeros(num_experts, hidden_size, dtype=dtype)

    if strategy == "svd":
        gate = ffn["gate"].float()  # [intermediate_size, hidden_size]
        try:
            _, _, Vt = torch.linalg.svd(gate, full_matrices=False)
            top_k = min(num_experts, Vt.shape[0])
            base = Vt[:top_k]
            if top_k < num_experts:
                pad = torch.zeros(num_experts - top_k, hidden_size)
                base = torch.cat([base, pad], dim=0)
            return base.to(dtype)
        except Exception:
            w = torch.empty(num_experts, hidden_size, dtype=torch.float32)
            torch.nn.init.xavier_uniform_(w)
            return w.to(dtype)

    raise ValueError(f"Unknown router_init strategy: {strategy}")


def _init_shared_expert(
    ffn: dict[str, "torch.Tensor"],
    moe_intermediate_size: int,
    ratio: float,
) -> dict[str, "torch.Tensor"]:
    import torch

    shared_size = max(1, int(moe_intermediate_size * ratio))
    gate = ffn["gate"]
    return {
        "gate": gate[:shared_size].clone(),
        "up":   ffn["up"][:shared_size].clone(),
        "down": ffn["down"][:, :shared_size].clone(),
    }


# ── Architecture converters ───────────────────────────────────────────────────

class _LlamaMixtralConverter:
    """LlamaForCausalLM / MistralForCausalLM → MixtralForCausalLM"""

    def is_ffn_key(self, key: str) -> bool:
        return any(s in key for s in (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj."))

    def extract_ffn(self, sd: dict, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}.mlp"
        return {
            "gate": sd[f"{prefix}.gate_proj.weight"],
            "up":   sd[f"{prefix}.up_proj.weight"],
            "down": sd[f"{prefix}.down_proj.weight"],
        }

    def write_moe_layer(self, out: dict, layer_idx: int, experts: list, router: "torch.Tensor") -> None:
        prefix = f"model.layers.{layer_idx}.block_sparse_moe"
        out[f"{prefix}.gate.weight"] = router
        for e, exp in enumerate(experts):
            out[f"{prefix}.experts.{e}.w1.weight"] = exp["gate"]
            out[f"{prefix}.experts.{e}.w3.weight"] = exp["up"]
            out[f"{prefix}.experts.{e}.w2.weight"] = exp["down"]

    def write_shared_expert(self, out: dict, layer_idx: int, shared: dict) -> None:
        # Mixtral doesn't have a native shared-expert field; store as auxiliary tensors
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.shared_expert"
        out[f"{prefix}.w1.weight"] = shared["gate"]
        out[f"{prefix}.w3.weight"] = shared["up"]
        out[f"{prefix}.w2.weight"] = shared["down"]

    def build_config(self, hf_cfg, *, num_experts, num_experts_per_tok, moe_intermediate_size,
                     shared_expert, shared_expert_ratio, balance_loss_coef, router_z_loss_coef) -> dict:
        d = hf_cfg.to_dict()
        d["architectures"] = ["MixtralForCausalLM"]
        d["model_type"] = "mixtral"
        d["num_local_experts"] = num_experts
        d["num_experts_per_tok"] = num_experts_per_tok
        d["intermediate_size"] = moe_intermediate_size
        d["output_router_logits"] = False
        d["router_aux_loss_coef"] = balance_loss_coef
        d["router_z_loss_coef"] = router_z_loss_coef
        if shared_expert:
            d["shared_expert_intermediate_size"] = int(moe_intermediate_size * shared_expert_ratio)
        return d


class _Qwen2MoeConverter:
    """Qwen2ForCausalLM → Qwen2MoeForCausalLM"""

    def is_ffn_key(self, key: str) -> bool:
        return any(s in key for s in (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj."))

    def extract_ffn(self, sd: dict, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}.mlp"
        return {
            "gate": sd[f"{prefix}.gate_proj.weight"],
            "up":   sd[f"{prefix}.up_proj.weight"],
            "down": sd[f"{prefix}.down_proj.weight"],
        }

    def write_moe_layer(self, out: dict, layer_idx: int, experts: list, router: "torch.Tensor") -> None:
        prefix = f"model.layers.{layer_idx}.mlp"
        out[f"{prefix}.gate.weight"] = router
        for e, exp in enumerate(experts):
            out[f"{prefix}.experts.{e}.gate_proj.weight"] = exp["gate"]
            out[f"{prefix}.experts.{e}.up_proj.weight"]   = exp["up"]
            out[f"{prefix}.experts.{e}.down_proj.weight"] = exp["down"]

    def write_shared_expert(self, out: dict, layer_idx: int, shared: dict) -> None:
        prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        out[f"{prefix}.gate_proj.weight"] = shared["gate"]
        out[f"{prefix}.up_proj.weight"]   = shared["up"]
        out[f"{prefix}.down_proj.weight"] = shared["down"]
        # Qwen2Moe shared-expert gate is a scalar; init to 1.0
        import torch
        out[f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight"] = torch.ones(1, 1)

    def build_config(self, hf_cfg, *, num_experts, num_experts_per_tok, moe_intermediate_size,
                     shared_expert, shared_expert_ratio, balance_loss_coef, router_z_loss_coef) -> dict:
        d = hf_cfg.to_dict()
        d["architectures"] = ["Qwen2MoeForCausalLM"]
        d["model_type"] = "qwen2_moe"
        d["num_experts"] = num_experts
        d["num_experts_per_tok"] = num_experts_per_tok
        d["moe_intermediate_size"] = moe_intermediate_size
        d["output_router_logits"] = False
        d["router_aux_loss_coef"] = balance_loss_coef
        d["router_z_loss_coef"] = router_z_loss_coef
        if shared_expert:
            d["shared_expert"] = True
            d["shared_expert_intermediate_size"] = int(moe_intermediate_size * shared_expert_ratio)
        return d


_CONVERTERS = {
    "LlamaForCausalLM":   _LlamaMixtralConverter(),
    "MistralForCausalLM": _LlamaMixtralConverter(),
    "Qwen2ForCausalLM":   _Qwen2MoeConverter(),
}


def _get_converter(arch: str):
    return _CONVERTERS.get(arch)


def _parse_layer_idx(key: str) -> int | None:
    """Extract layer index from a state dict key like 'model.layers.12.mlp...'"""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
