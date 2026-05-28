# mlambaformer training container

Pinned ROCm image that bundles the custom multi-arch PyTorch build, the mad-lab-train
pipeline, and mlambaformer. Host churn cannot affect it; rebuild only on deliberate bumps.

## ROCm base tag — VERIFY BEFORE BUILDING

`docker/Dockerfile` defaults `ARG ROCM_BASE=rocm/dev-ubuntu-22.04:7.2.3-complete`.
This tag was NOT verified at authoring time (no docker in the authoring shell). Before
the first build, confirm it exists and supports gfx1201 (RDNA4):

    docker manifest inspect rocm/dev-ubuntu-22.04:7.2.3-complete

If absent, pick the nearest `7.2.x` tag that documents RDNA4/gfx1201 support and pass it
via `--build-arg ROCM_BASE=<tag>`. Record the verified tag here.

## Build (first build compiles torch for gfx1030;gfx1201 — 3-4 hr, cached after)

    cd ~/GitHub/mad-lab-train
    DOCKER_BUILDKIT=1 docker buildx build \
      -f docker/Dockerfile \
      --build-context mlambaformer=../mlambaformer \
      --build-context pytorch=$HOME/GitHub/pytorch \
      --build-arg ROCM_BASE=rocm/dev-ubuntu-22.04:7.2.3-complete \
      --build-arg INSTALL_MODE=release \
      -t mlambaformer-train:latest .

`mamba_ssm` is intentionally absent (needs the MAD-248 ROCm kernel port). The image
runs the all-attention and MLA cells today.

## Run

GPU passthrough is required on every run:

    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined -e HSA_VISIBLE_DEVICES=0

- Serve (dashboard connects over HTTP):

      docker run --rm <gpu flags> -p 8848:8848 mlambaformer-train:latest serve

- Headless one-shot (executes a Run config's stages, then exits):

      docker run --rm <gpu flags> -v /path/run:/run mlambaformer-train:latest run --config /run/run.json

See `docker-compose.yml` for the full passthrough + volume setup.
