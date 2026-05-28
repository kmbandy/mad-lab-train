# syntax=docker/dockerfile:1.7
ARG ROCM_BASE=rocm/dev-ubuntu-22.04:7.2.3-complete

# ── builder: compile torch for gfx1030;gfx1201 into a wheel ───────────────────
FROM ${ROCM_BASE} AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH="gfx1030;gfx1201"
ENV MAX_JOBS=24
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git cmake ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build/pytorch
# pytorch source tree is bind-mounted at build time (named build-context "pytorch").
RUN --mount=type=bind,source=pytorch,target=/build/pytorch,rw \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install --no-cache-dir -r requirements.txt \
    && python3 setup.py bdist_wheel --dist-dir /wheels

# ── runtime: lean image with the custom torch + training stack ────────────────
FROM ${ROCM_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.hf-cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

# 1) custom torch wheel from the builder stage
COPY --from=builder /wheels/*.whl /tmp/wheels/
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir /tmp/wheels/*.whl

# 2) training stack — constraints prevent any transitive PyPI torch
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        "transformers>=4.57,<4.60" "trl>=0.12,<0.16" "datasets>=2.19" \
        "accelerate>=0.34" "deepspeed>=0.14" sentencepiece pyyaml

# 3) the two repos
ARG INSTALL_MODE=release
RUN --mount=type=bind,source=.,target=/src/mad-lab-train,rw \
    if [ "$INSTALL_MODE" = "release" ]; then \
        pip install --no-cache-dir --no-deps /src/mad-lab-train ; \
    fi
COPY --from=mlambaformer . /src/mlambaformer
RUN pip install --no-cache-dir -c /tmp/constraints.txt /src/mlambaformer

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
