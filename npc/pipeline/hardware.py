"""
Hardware detection and device configuration.

Auto-detects CUDA vs ROCm and returns a HardwareProfile with everything
downstream code needs — device string, dtype, bitsandbytes availability, etc.
No config needed; just call detect().
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HardwareProfile:
    backend: Literal["cuda", "rocm", "cpu"]
    device: str                        # "cuda:0", "cuda", "cpu"
    compute_dtype: str                 # "float16" or "bfloat16"
    fp16: bool
    bf16: bool
    bnb_available: bool                # bitsandbytes 4-bit quantization
    gpu_name: str = ""
    vram_mb: int = 0
    sm_version: int = 0                # CUDA: 61, 70, 80 etc. 0 = unknown/ROCm
    notes: list[str] = field(default_factory=list)

    @property
    def supports_flash_attn(self) -> bool:
        # Flash attention requires sm_80+ (Ampere) on CUDA
        return self.backend == "cuda" and self.sm_version >= 80

    @property
    def supports_4bit(self) -> bool:
        return self.bnb_available

    def summary(self) -> str:
        lines = [
            f"Backend : {self.backend.upper()}",
            f"Device  : {self.device}",
            f"GPU     : {self.gpu_name} ({self.vram_mb} MB VRAM)" if self.gpu_name else "GPU     : unknown",
            f"dtype   : {'fp16' if self.fp16 else 'bf16' if self.bf16 else 'fp32'}",
            f"4-bit   : {'yes (bitsandbytes)' if self.bnb_available else 'no'}",
            f"flash   : {'yes' if self.supports_flash_attn else 'no'}",
        ]
        if self.notes:
            lines += [f"note    : {n}" for n in self.notes]
        return "\n".join(lines)


def detect(verbose: bool = True) -> HardwareProfile:
    """Auto-detect available GPU hardware and return a HardwareProfile."""
    import torch

    notes: list[str] = []

    # ---- CUDA ----
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            sm = props.major * 10 + props.minor
            vram_mb = props.total_memory // (1024 * 1024)
            gpu_name = props.name

            # bf16 native support requires sm_80+ (Ampere)
            use_bf16 = sm >= 80
            compute_dtype = "bfloat16" if use_bf16 else "float16"

            bnb_available = _check_bnb_cuda()
            if not bnb_available:
                notes.append("bitsandbytes not available — 4-bit QLoRA disabled")

            if sm < 70:
                notes.append(
                    f"sm_{sm} is below sm_70 — using torch 2.4.x build required; "
                    "set_submodule patch needed for transformers 5.x"
                )

            profile = HardwareProfile(
                backend="cuda",
                device="cuda:0",
                compute_dtype=compute_dtype,
                fp16=not use_bf16,
                bf16=use_bf16,
                bnb_available=bnb_available,
                gpu_name=gpu_name,
                vram_mb=vram_mb,
                sm_version=sm,
                notes=notes,
            )
            if verbose:
                print("[hardware] Detected CUDA GPU:")
                print(profile.summary())
            return profile

        except Exception as e:
            notes.append(f"CUDA probe error: {e}")

    # ---- ROCm (AMD) ----
    # torch.cuda.is_available() returns True on ROCm too, but we can
    # distinguish by checking for the HIP runtime.
    if _check_rocm():
        gpu_name, vram_mb = _probe_rocm()
        # ROCm / RDNA2+ supports bf16 in software but it's slow without hardware support.
        # RDNA3 (gfx11xx) has hardware bf16; RDNA2 (gfx10xx) does not.
        # Default to fp16 to be safe across all ROCm GPUs.
        bnb_available = _check_bnb_rocm()
        notes.append("ROCm detected — gfx803 (RX 480) is pure transformer only, no SSM/Mamba")
        if not bnb_available:
            notes.append("bitsandbytes ROCm not available — install bitsandbytes-rocm")

        profile = HardwareProfile(
            backend="rocm",
            device="cuda",   # ROCm exposes itself as "cuda" to PyTorch
            compute_dtype="float16",
            fp16=True,
            bf16=False,
            bnb_available=bnb_available,
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            notes=notes,
        )
        if verbose:
            print("[hardware] Detected ROCm GPU:")
            print(profile.summary())
        return profile

    # ---- CPU fallback ----
    notes.append("No GPU detected — training on CPU will be extremely slow")
    profile = HardwareProfile(
        backend="cpu",
        device="cpu",
        compute_dtype="float32",
        fp16=False,
        bf16=False,
        bnb_available=False,
        notes=notes,
    )
    if verbose:
        print("[hardware] No GPU detected, falling back to CPU")
    return profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_bnb_cuda() -> bool:
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except ImportError:
        return False


def _check_bnb_rocm() -> bool:
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except ImportError:
        return False


def _check_rocm() -> bool:
    """Returns True if HIP/ROCm runtime is present."""
    return shutil.which("rocm-smi") is not None or shutil.which("/opt/rocm/bin/rocm-smi") is not None


def _probe_rocm() -> tuple[str, int]:
    """Returns (gpu_name, vram_mb) from rocm-smi."""
    rocm_smi = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    try:
        out = subprocess.check_output(
            [rocm_smi, "--showproductname", "--csv"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        name = out.strip().splitlines()[-1].split(",")[-1].strip()
    except Exception:
        name = "AMD GPU"

    try:
        out = subprocess.check_output(
            [rocm_smi, "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        # Header: device,VRAM Total Memory (B),...
        vram_b = int(out.strip().splitlines()[-1].split(",")[1].strip())
        vram_mb = vram_b // (1024 * 1024)
    except Exception:
        vram_mb = 0

    return name, vram_mb


if __name__ == "__main__":
    detect()
