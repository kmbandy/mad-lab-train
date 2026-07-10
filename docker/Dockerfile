# syntax=docker/dockerfile:1.7
#
# mad-lab-train training image (FastAPI orchestrator, or headless one-shot run).
#
# WHAT CHANGED (2026-07-10)
#
# 1. The `builder` stage is GONE. It compiled PyTorch from source for gfx1030;gfx1201
#    (MAX_JOBS=24, ~3-4h) because "prebuilt wheels omit gfx1030 -> 6900 XT SIGSEGV".
#    No longer true: `torch==2.13.0+rocm7.2` ships gfx1030, gfx1201, gfx942 AND gfx950 --
#    a superset of what the source build produced -- and a bf16 matmul on the 6900 XT was
#    verified against it. Image build drops from hours to minutes.
#
# 2. No ROCm base image. The +rocm7.2 wheel BUNDLES its own ROCm userspace (libamdhip64,
#    libhsa-runtime64, libamd_comgr, librocblas) inside site-packages/torch/lib and loads
#    THOSE, never /opt/rocm's -- verified against /proc/<pid>/maps. So this needs the
#    kernel driver (/dev/kfd, /dev/dri, passed at run time) plus a few system libraries,
#    not a ~20 GB rocm/dev-ubuntu base. The old ROCM_BASE default
#    (rocm/dev-ubuntu-22.04:7.2.3-complete) was stale anyway: it predates the 2026-06-20
#    migration, and the host now runs TheRock ROCm 7.13.
#
#    Corollary: container-torch never loads /opt/rocm, so it gets NONE of TheRock's tooling
#    (rocprofv3, the gfx1201 ATT trace decoder, s_alloc_vgpr). Correct for training; wrong
#    for kernel profiling -- profile on the host.
#
# 3. Every install is pinned or --no-deps, and the build now FAILS if a CUDA torch leaks
#    in. See docker/constraints.txt for why that is not paranoia.
#
# Build (from the mad-lab-train repo root):
#   podman build -f docker/Dockerfile \
#       --build-context mlambaformer=../mlambaformer \
#       -t mlambaformer-train:latest .
FROM docker.io/library/ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/workspace/.hf-cache

# Ubuntu 24.04 ships Python 3.12, matching the cp312 wheel.
#   libnuma1/libdrm2/libdrm-amdgpu1/libelf1/libzstd1 : bundled HSA runtime + rocBLAS
#   libatomic1/libgomp1                              : `import torch` fails without them
#   gcc                                              : NOT optional. Triton's AMD backend
#       JIT-compiles `hip_utils` at RUNTIME, so the first Mamba-3 kernel launch dies with
#       "RuntimeError: Failed to find C compiler" without it. An image without gcc passes
#       every CPU test and then breaks on the first hybrid training step.
#   patch                                            : applies docker/patches/*.patch
#   git                                              : the transformers pin is a git+ URL
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git ca-certificates gcc patch \
        libnuma1 libdrm2 libdrm-amdgpu1 libelf1 libzstd1 libatomic1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mlmf
COPY docker/constraints.txt /opt/mlmf/constraints.txt

# 1) torch FIRST, from the ROCm index ONLY. Nothing after this may introduce a torch.
RUN pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
        torch==2.13.0+rocm7.2 pytorch-triton-rocm

# 2) mlambaformer's exact dependency closure, --no-deps so pip never resolves. This is
#    what carries the Zyphra transformers fork at the vendored ZAYA SHA.
COPY --from=mlambaformer docker/requirements.lock.txt /opt/mlmf/mlambaformer-requirements.lock.txt
RUN pip install --no-deps -c /opt/mlmf/constraints.txt \
        -r /opt/mlmf/mlambaformer-requirements.lock.txt

# 3) mamba_ssm from a pinned SHA, NOT PyPI. The released mamba-ssm==2.3.2.post1 sdist
#    predates upstream PR #977 and reaches for CUDA_HOME even when building no extension,
#    dying with "OSError: CUDA_HOME environment variable is not set" on an AMD box. #977
#    gates the extension behind MAMBA_KEEP_CUDA_BUILD; it is Mamba-1 only, and Mamba-3 is
#    pure Triton, so nothing compiles.
#    --no-build-isolation: setup.py does `import torch`, which pip's isolation hides.
#    --no-deps: install_requires pulls tilelang + quack-kernels -> nvidia-cutlass-dsl,
#               cuda-toolkit, and a CUDA torch.
ARG MAMBA_SSM_SHA=f577286d052741c35d39cd43bdc3fad27120f22c
RUN pip install --no-deps --no-build-isolation -c /opt/mlmf/constraints.txt \
        "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@${MAMBA_SSM_SHA}"

# 3b) MAD-247: Mamba-3's SISO Triton kernels cannot run on RDNA4 (gfx1201). utils.py builds
#     cos/sin/tanh/sech2 out of NVIDIA PTX inline asm (constraints="=f,f"); "f" has no AMD
#     register class, so LLVM aborts the process --
#       error: couldn't allocate output register for constraint 'f'
#     -- on the first SISO forward. `maxnreg` is likewise unsupported by Triton's AMD
#     backend. Patch sourced from mlambaformer; see its docker/patches/README.md.
#     NOTE: do NOT `import mamba_ssm` at build time. Its package __init__ reaches torch.cuda,
#     and the build container has no GPU by design ("No CUDA GPUs are available").
COPY --from=mlambaformer docker/patches/0001-mamba3-rocm-rdna4-runtime.patch /opt/mlmf/
RUN SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && patch -p1 -d "$SITE" < /opt/mlmf/0001-mamba3-rocm-rdna4-runtime.patch

# 4) training stack. Resolved WITH deps but UNDER constraints, so torch and transformers
#    cannot be replaced; everything else is free to pull what it needs.
RUN pip install -c /opt/mlmf/constraints.txt \
        "trl>=0.12,<0.16" "datasets>=2.19" "accelerate>=0.34" "deepspeed>=0.14" \
        sentencepiece pyyaml aiosqlite

# 5) the two repos.
#    mlambaformer : --no-deps; its closure was installed in step 2.
#    mad-lab-train: WITH deps (under constraints) so the FastAPI server + SQLAlchemy
#    orchestrator stack is present for both `serve` and `run` modes. Its deps include
#    bitsandbytes + mergekit (heavy; mergekit declares torch, satisfied by step 1 and held
#    by the constraint). If either fails on ROCm, move it to an optional extra -- do NOT
#    silently drop server deps.
COPY --from=mlambaformer . /src/mlambaformer
RUN pip install --no-deps -c /opt/mlmf/constraints.txt /src/mlambaformer

ARG INSTALL_MODE=release
RUN --mount=type=bind,source=.,target=/src/mad-lab-train,rw \
    if [ "$INSTALL_MODE" = "release" ]; then \
        pip install -c /opt/mlmf/constraints.txt /src/mad-lab-train ; \
    fi

# 6) Fail the build if anything above pulled a CUDA torch, lost an arch, dragged in
#    tilelang, or swapped the Zyphra transformers fork for a PyPI build. This turns the
#    exact failure these constraints exist to prevent into a BUILD ERROR rather than a
#    silent four-hour debugging session.
RUN python3 - <<'PY'
import importlib.util as u
import pathlib
import shutil
import sysconfig
import torch
assert torch.version.hip, f"torch has no HIP runtime: {torch.__version__}"
assert torch.version.cuda is None, f"CUDA torch leaked in: {torch.__version__}"
assert "+rocm7.2" in torch.__version__, torch.__version__
archs = torch.cuda.get_arch_list()   # compile-time list; needs no device
for need in ("gfx1201", "gfx1030", "gfx942"):
    assert need in archs, f"missing {need}: {archs}"
assert u.find_spec("tilelang") is None, "tilelang leaked in (drags CUDA torch)"
assert u.find_spec("mamba_ssm") is not None, "mamba_ssm missing"

# Triton's AMD backend shells out to a C compiler at first kernel launch.
assert shutil.which("cc") or shutil.which("gcc"), "no C compiler: Triton JIT will fail at runtime"

# MAD-247 patch must have taken, or every Mamba-3 SISO forward aborts on gfx1201.
# Checked textually: importing mamba_ssm reaches torch.cuda, and there is no GPU at build time.
_m3 = pathlib.Path(sysconfig.get_paths()["purelib"]) / "mamba_ssm/ops/triton/mamba3"
_utils = (_m3 / "utils.py").read_text()
assert "_IS_HIP" in _utils and "def _maxnreg" in _utils, "mamba3 utils.py was not patched"
assert "inline_asm_elementwise" not in _utils.split("if _IS_HIP:")[0], \
    "PTX inline asm still on the unconditional path"
for _n in ("mamba3_siso_fwd.py", "mamba3_siso_bwd.py"):
    assert "maxnreg=r" not in (_m3 / _n).read_text(), f"{_n} still passes maxnreg"

import transformers  # noqa: F401
from transformers.models import zamba2  # noqa: F401  # proves the Zyphra fork, not PyPI
import mlambaformer  # noqa: F401  # safe: its mamba import is lazy, unlike mamba_ssm's
print("OK", torch.__version__, "hip", torch.version.hip,
      "| transformers", transformers.__version__, "| MAD-247 patched")
PY

WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
