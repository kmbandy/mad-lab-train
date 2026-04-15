#!/bin/bash
set -euo pipefail

# ============================================================
# ec2-gpu-expert-gen.sh
# Launches 3 g7e.2xlarge spot instances to generate GPU expert
# synthetic training data using DS-Coder V2 Lite Q6_K GGUF.
#
# Usage:
#   ./ec2-gpu-expert-gen.sh --package            # bundle pipeline, stage to S3
#   ./ec2-gpu-expert-gen.sh --launch             # spin up 3 spot instances
#   ./ec2-gpu-expert-gen.sh --run <variant>      # run on EC2 (called by user-data)
#   ./ec2-gpu-expert-gen.sh --userdata <variant> # print bootstrap script
# ============================================================

# --- S3 / model config ---
S3_BUCKET="mad-lab-files"
S3_MODEL_KEY="quant-output/deepseek-coder-v2-lite-instruct-Q6_K.gguf"
S3_JOB_PREFIX="gpu-expert-jobs"
S3_OUTPUT_PREFIX="gpu-expert-output"
S3_LLAMA_BUILD_PREFIX="llama-builds"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp"
MOUNTPOINT_S3_VERSION="1.22.2"

# --- Generation config ---
LLAMA_PORT=8080
LLAMA_PARALLEL=256      # parallel slots — DS-Coder V2 Lite GGUF hard limit is n_seq_max=256
LLAMA_CTX=2048          # context per slot; total = 256*2048 = 524,288
SAMPLES_WRITER=15000    # per instance; 3 instances = 45k writer raw
SAMPLES_OPUS=10000      # per instance; 3 instances = 30k opus raw (75k total, filter to 50k)
CONCURRENCY=256         # match LLAMA_PARALLEL

# --- Temperature variants ---
VARIANTS=("t60" "t75" "t90")
WRITER_TEMPS=("0.60" "0.75" "0.90")
OPUS_TEMPS=("0.55"  "0.70" "0.85")

# --- EC2 config ---
EC2_REGION="us-east-2"
EC2_AMI="ami-03bda78a7c7c13b45"   # DLAMI with CUDA (us-east-2)
EC2_INSTANCE_TYPE="g7e.2xlarge"
EC2_KEY_NAME="mad-lab-key"
EC2_IAM_PROFILE="arn:aws:iam::080869524552:instance-profile/mad-lab-ec2-quant-role"
EC2_VPC_ID="vpc-0a8e766998c7d5e23"
EC2_MAX_SPOT_PRICE="0.85"
EC2_STORAGE_GB=100

# --- SCRIPT DIR (absolute path of this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPC_DIR="$SCRIPT_DIR"   # this script lives in npc/

# ============================================================
# HELPERS
# ============================================================

die() { echo "ERROR: $*" >&2; exit 1; }

ensure_python_deps() {
    python3 -c "import yaml" &>/dev/null || pip install -q pyyaml
}

setup_s3_mount() {
    if ! command -v mount-s3 &>/dev/null; then
        echo "  Installing mountpoint-s3 ${MOUNTPOINT_S3_VERSION}..."
        wget -q "https://s3.amazonaws.com/mountpoint-s3-release/${MOUNTPOINT_S3_VERSION}/x86_64/mount-s3.deb"
        sudo apt-get install -y ./mount-s3.deb -qq
        rm mount-s3.deb
    fi
}

wait_for_llama() {
    local port=$1
    local max_wait=120
    local waited=0
    echo "  Waiting for llama-server on port $port..."
    while ! curl -sf "http://localhost:$port/health" | grep -q '"status":"ok"' 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            die "llama-server did not become ready in ${max_wait}s"
        fi
    done
    echo "  llama-server ready (${waited}s)"
}

# ============================================================
# MODE: --package
# Bundles the pipeline + 3 temperature-variant theme configs
# and stages everything to S3.
# ============================================================

mode_package() {
    echo "=== Packaging GPU expert generation pipeline ==="

    # Validate source dirs exist
    [ -d "$NPC_DIR/pipeline" ]              || die "npc/pipeline/ not found at $NPC_DIR"
    [ -d "$NPC_DIR/themes/gpu_architecture" ] || die "npc/themes/gpu_architecture/ not found"
    [ -f "$NPC_DIR/themes/gpu_architecture/theme.yaml" ] || die "theme.yaml not found"

    # Confirm Q6K model is in S3
    echo "[1] Verifying Q6K model in S3..."
    aws s3 ls "s3://${S3_BUCKET}/${S3_MODEL_KEY}" --region "$EC2_REGION" \
        || die "Q6K model not found at s3://${S3_BUCKET}/${S3_MODEL_KEY}"
    echo "  Model confirmed in S3"

    ensure_python_deps

    # Build one tarball per variant
    for i in "${!VARIANTS[@]}"; do
        VARIANT="${VARIANTS[$i]}"
        WRITER_TEMP="${WRITER_TEMPS[$i]}"
        OPUS_TEMP="${OPUS_TEMPS[$i]}"
        JOB_NAME="gpu-expert-${VARIANT}"

        echo ""
        echo "[$(($i+2))] Packaging variant: $VARIANT (writer=${WRITER_TEMP} opus=${OPUS_TEMP})"

        STAGING="$SCRIPT_DIR/../.staging-${JOB_NAME}"
        rm -rf "$STAGING"
        mkdir -p "$STAGING"

        # Copy pipeline code
        cp -r "$NPC_DIR/pipeline" "$STAGING/"

        # Copy theme, then patch temperatures in theme.yaml
        mkdir -p "$STAGING/themes"
        cp -r "$NPC_DIR/themes/gpu_architecture" "$STAGING/themes/"

        python3 - <<PYEOF
import yaml, copy
with open("${NPC_DIR}/themes/gpu_architecture/theme.yaml") as f:
    cfg = yaml.safe_load(f)

# Patch temperatures
cfg["generators"]["writer"]["temperature"] = float("${WRITER_TEMP}")
cfg["generators"]["opus"]["temperature"]   = float("${OPUS_TEMP}")

# Disable kiwix — not available on EC2
if "lore" in cfg and "kiwix" in cfg["lore"]:
    cfg["lore"]["kiwix"]["enabled"] = False

with open("${STAGING}/themes/gpu_architecture/theme.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"  Patched theme.yaml: writer={float('${WRITER_TEMP}')}, opus={float('${OPUS_TEMP}')}, kiwix=disabled")
PYEOF

        # Load Qdrant Cloud creds from secrets.env
        QDRANT_SECRETS="${SCRIPT_DIR}/../../mad-lab-mcp/secrets.env"
        QDRANT_CLOUD_URL_VAL=""
        QDRANT_CLOUD_KEY_VAL=""
        if [ -f "$QDRANT_SECRETS" ]; then
            QDRANT_CLOUD_URL_VAL=$(grep '^QDRANT_CLOUD_URL=' "$QDRANT_SECRETS" | cut -d= -f2- | tr -d '[:space:]')
            QDRANT_CLOUD_KEY_VAL=$(grep '^QDRANT_CLOUD_API_KEY=' "$QDRANT_SECRETS" | cut -d= -f2- | tr -d '[:space:]')
        fi
        [ -n "$QDRANT_CLOUD_URL_VAL" ] || die "QDRANT_CLOUD_URL not found in $QDRANT_SECRETS"
        [ -n "$QDRANT_CLOUD_KEY_VAL" ] || die "QDRANT_CLOUD_API_KEY not found in $QDRANT_SECRETS"
        echo "  Qdrant Cloud URL: $QDRANT_CLOUD_URL_VAL"

        # Write EC2 run config
        cat > "$STAGING/run_ec2.yaml" <<RUNCFG
# EC2 run config — GPU expert generation (variant: ${VARIANT})
# All endpoints → same local llama-server (DS-Coder V2 Lite Q6_K)

writer_api_base: "http://localhost:${LLAMA_PORT}/v1"
writer_model: "default"

opus_api_base: "http://localhost:${LLAMA_PORT}/v1"
opus_model: "default"

qwen_api_base: "http://localhost:${LLAMA_PORT}/v1"
qwen_model: "default"

# Fast filter — same server (DS-Coder handles this too)
fast_filter_api_base: "http://localhost:${LLAMA_PORT}/v1"
fast_filter_model: "default"

# Qdrant Cloud — lore retrieval (amd_gpu_docs, github_prs, arxiv_gpu)
qdrant_url: "${QDRANT_CLOUD_URL_VAL}"
qdrant_api_key: "${QDRANT_CLOUD_KEY_VAL}"
qdrant_collection: "memory"
qdrant_sources:
  - amd_gpu_docs
  - github_prs
  - arxiv_gpu

# No Kiwix or ChromaDB on EC2
chromadb_path: ""
kiwix_base: ""

# Generation targets (15k + 10k = 25k per instance, 75k total across 3)
samples_writer: ${SAMPLES_WRITER}
samples_opus:   ${SAMPLES_OPUS}

concurrency: ${CONCURRENCY}

# Validation thresholds
min_quality_score: 0.65
cross_review_threshold: 0.8

# Output — written to NVMe, uploaded to S3 after completion
output_dir: "/mnt/nvme/output"
RUNCFG

        echo "  Wrote run_ec2.yaml"

        # Tarball
        TARBALL="${JOB_NAME}.tar.gz"
        tar -czf "$TARBALL" -C "$STAGING/.." "$(basename "$STAGING")"
        echo "  Tarball: $(du -sh "$TARBALL" | cut -f1)"

        # Stage to S3
        aws s3 cp "$TARBALL" "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}" \
            --region "$EC2_REGION" --no-progress
        echo "  Staged: s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}"

        rm -rf "$STAGING" "$TARBALL"
    done

    # Stage this script itself
    aws s3 cp "$0" "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/ec2-gpu-expert-gen.sh" \
        --region "$EC2_REGION" --no-progress
    echo ""
    echo "=== Package complete ==="
    echo "  3 variants staged: ${VARIANTS[*]}"
    echo ""
    echo "Launch with:"
    echo "  $0 --launch"
}

# ============================================================
# MODE: --run <variant>
# Runs on EC2. Downloads model, builds llama-server, generates.
# ============================================================

mode_run() {
    VARIANT=${1:-}
    [ -n "$VARIANT" ] || die "--run requires a variant (t60, t75, or t90)"
    JOB_NAME="gpu-expert-${VARIANT}"
    TARBALL="${JOB_NAME}.tar.gz"
    STAGING_NAME=".staging-${JOB_NAME}"

    echo "=== GPU expert gen run: $VARIANT ==="

    echo "[1] Installing system deps..."
    MISSING=()
    command -v git   >/dev/null || MISSING+=(git)
    command -v cmake >/dev/null || MISSING+=(cmake)
    dpkg -s build-essential &>/dev/null || MISSING+=(build-essential)
    dpkg -s python3-venv    &>/dev/null || MISSING+=(python3-venv)
    dpkg -s python3-pip     &>/dev/null || MISSING+=(python3-pip)
    if [ ${#MISSING[@]} -gt 0 ]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq --no-install-recommends "${MISSING[@]}"
    else
        echo "  System deps OK"
    fi

    echo "[2] Unpacking pipeline..."
    aws s3 cp "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}" . \
        --region "$EC2_REGION" --no-progress
    tar -xzf "$TARBALL"
    PIPELINE_DIR="$(pwd)/${STAGING_NAME}"
    echo "  Pipeline at: $PIPELINE_DIR"

    echo "[3] Setting up Python venv..."
    python3 -m venv /tmp/gen-venv
    source /tmp/gen-venv/bin/activate
    pip install -q openai jinja2 pyyaml requests qdrant-client

    echo "[4] Mounting NVMe instance store..."
    # Add CUDA to PATH
    for cuda_path in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -d "$cuda_path" ] && export PATH="$cuda_path:$PATH"
    done

    ROOT_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | while read dev; do
        lsblk -no MOUNTPOINT "$dev" 2>/dev/null | grep -q '^/$' && echo "$dev" && break
    done)
    NVME_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | while read dev; do
        [ "$dev" != "$ROOT_DEV" ] && echo "$dev" && break
    done)

    if [ -n "$NVME_DEV" ]; then
        EXISTING_MOUNT=$(lsblk -no MOUNTPOINT "$NVME_DEV" 2>/dev/null | grep -v '^$' | head -1)
        if [ -n "$EXISTING_MOUNT" ]; then
            WORK_DIR="$EXISTING_MOUNT"
            echo "  NVMe pre-mounted at $WORK_DIR"
        else
            blkid "$NVME_DEV" | grep -q ext4 || sudo mkfs.ext4 -F "$NVME_DEV"
            WORK_DIR="/mnt/nvme"
            sudo mkdir -p "$WORK_DIR"
            sudo mount "$NVME_DEV" "$WORK_DIR"
            echo "  Mounted $NVME_DEV → $WORK_DIR"
        fi
        sudo chmod 777 "$WORK_DIR"
    else
        echo "  WARNING: No NVMe found — using /tmp (limited space)"
        WORK_DIR="/tmp/gpu-expert-work"
        mkdir -p "$WORK_DIR"
    fi

    # Fix output dir in run config to use actual WORK_DIR
    sed -i "s|/mnt/nvme/output|${WORK_DIR}/output|g" "$PIPELINE_DIR/run_ec2.yaml"
    mkdir -p "${WORK_DIR}/output"

    echo "[5] Downloading Q6K model from S3..."
    MODEL_PATH="${WORK_DIR}/model.gguf"
    if [ -f "$MODEL_PATH" ]; then
        echo "  Model already present ($(du -sh "$MODEL_PATH" | cut -f1))"
    else
        aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_KEY}" "$MODEL_PATH" \
            --region "$EC2_REGION" --no-progress
        echo "  Model ready ($(du -sh "$MODEL_PATH" | cut -f1))"
    fi

    echo "[6] Building llama-server (or restoring from cache)..."
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '.' || echo "cpu")
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" \
        | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' | tr '.' '_' || echo "none")
    CACHE_KEY="sm${GPU_ARCH}_cuda${CUDA_VER}"
    BIN_S3="s3://${S3_BUCKET}/${S3_LLAMA_BUILD_PREFIX}/${CACHE_KEY}"
    LLAMA_DIR="${WORK_DIR}/llama.cpp"

    echo "  GPU: sm${GPU_ARCH}  CUDA: ${CUDA_VER}  cache: ${CACHE_KEY}"

    if [ ! -d "$LLAMA_DIR" ]; then
        git clone --depth=1 "$LLAMA_CPP_REPO" "$LLAMA_DIR"
    fi
    mkdir -p "${LLAMA_DIR}/build/bin"

    if aws s3 ls "${BIN_S3}/llama-server" &>/dev/null; then
        echo "  Cache hit — restoring llama-server binary..."
        aws s3 cp "${BIN_S3}/llama-server" "${LLAMA_DIR}/build/bin/llama-server" \
            --region "$EC2_REGION" --no-progress
        chmod +x "${LLAMA_DIR}/build/bin/llama-server"
        echo "  Skipping build"
    else
        echo "  Cache miss — building llama.cpp with CUDA (this takes ~15 min)..."
        cd "$LLAMA_DIR"
        START=$(date +%s)
        cmake -B build \
            -DGGML_CUDA=ON \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build build --target llama-server -j "$(nproc)"
        END=$(date +%s)
        echo "  Build done in $((END - START))s"
        aws s3 cp build/bin/llama-server "${BIN_S3}/llama-server" \
            --region "$EC2_REGION" --no-progress
        echo "  Cached → $CACHE_KEY"
        cd -
    fi

    LLAMA_BIN="${LLAMA_DIR}/build/bin/llama-server"
    LLAMA_PORT_WRITER=8080
    LLAMA_PORT_OPUS=8081
    LLAMA_CTX_TOTAL=$(( LLAMA_CTX * LLAMA_PARALLEL ))

    # Two servers — same model loaded twice (~13GB each), separate KV caches.
    # Total VRAM: ~26GB model + ~34GB KV = ~60GB, well within 97GB on B200.
    # Writer and opus generate in parallel instead of sequentially.

    echo "[7] Starting llama-server x2 (parallel=${LLAMA_PARALLEL} each)..."
    "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$LLAMA_PORT_WRITER" \
        --parallel "$LLAMA_PARALLEL" \
        --ctx-size "$LLAMA_CTX_TOTAL" \
        -ngl 999 \
        --log-disable \
        > "${WORK_DIR}/llama-server-writer.log" 2>&1 &
    LLAMA_PID_WRITER=$!

    "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$LLAMA_PORT_OPUS" \
        --parallel "$LLAMA_PARALLEL" \
        --ctx-size "$LLAMA_CTX_TOTAL" \
        -ngl 999 \
        --log-disable \
        > "${WORK_DIR}/llama-server-opus.log" 2>&1 &
    LLAMA_PID_OPUS=$!

    echo "  Writer server PID: $LLAMA_PID_WRITER (port $LLAMA_PORT_WRITER)"
    echo "  Opus server PID:   $LLAMA_PID_OPUS (port $LLAMA_PORT_OPUS)"
    wait_for_llama "$LLAMA_PORT_WRITER"
    wait_for_llama "$LLAMA_PORT_OPUS"

    # Patch run configs to point each model at its own server port
    cp "$PIPELINE_DIR/run_ec2.yaml" "$PIPELINE_DIR/run_ec2_writer.yaml"
    cp "$PIPELINE_DIR/run_ec2.yaml" "$PIPELINE_DIR/run_ec2_opus.yaml"
    sed -i "s|localhost:${LLAMA_PORT}|localhost:${LLAMA_PORT_WRITER}|g" "$PIPELINE_DIR/run_ec2_writer.yaml"
    sed -i "s|localhost:${LLAMA_PORT}|localhost:${LLAMA_PORT_OPUS}|g"   "$PIPELINE_DIR/run_ec2_opus.yaml"

    echo "[8] Generating writer + opus in parallel..."
    cd "$PIPELINE_DIR"
    START=$(date +%s)

    python3 pipeline/generate.py \
        --config run_ec2_writer.yaml \
        --theme themes/gpu_architecture \
        --model writer >> /home/ubuntu/gen-writer.log 2>&1 &
    PID_WRITER=$!

    python3 pipeline/generate.py \
        --config run_ec2_opus.yaml \
        --theme themes/gpu_architecture \
        --model opus >> /home/ubuntu/gen-opus.log 2>&1 &
    PID_OPUS=$!

    echo "  writer PID: $PID_WRITER  opus PID: $PID_OPUS"
    wait $PID_WRITER && echo "  Writer done" || echo "  [warn] writer exited non-zero"
    wait $PID_OPUS   && echo "  Opus done"   || echo "  [warn] opus exited non-zero"

    END=$(date +%s)
    echo "  Both done in $((END - START))s"

    echo "[9] Fast-filter validation (parallel)..."
    python3 pipeline/validate.py \
        --config run_ec2_writer.yaml \
        --theme themes/gpu_architecture \
        --pass fast-filter \
        --source writer >> /home/ubuntu/gen-writer.log 2>&1 &
    PID_FF_WRITER=$!

    python3 pipeline/validate.py \
        --config run_ec2_opus.yaml \
        --theme themes/gpu_architecture \
        --pass fast-filter \
        --source opus >> /home/ubuntu/gen-opus.log 2>&1 &
    PID_FF_OPUS=$!

    wait $PID_FF_WRITER || echo "  [warn] fast-filter writer returned non-zero"
    wait $PID_FF_OPUS   || echo "  [warn] fast-filter opus returned non-zero"

    echo "[11] Uploading output to S3..."
    OUTPUT_DIR="${WORK_DIR}/output"
    S3_OUT="s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/${VARIANT}"

    for f in raw_writer raw_opus filtered_writer filtered_opus; do
        fpath="${OUTPUT_DIR}/${f}.jsonl"
        if [ -f "$fpath" ]; then
            COUNT=$(wc -l < "$fpath")
            aws s3 cp "$fpath" "${S3_OUT}/${f}.jsonl" \
                --region "$EC2_REGION" --no-progress
            echo "  Uploaded ${f}.jsonl (${COUNT} samples)"
        else
            echo "  [skip] ${f}.jsonl not found"
        fi
    done

    kill "$LLAMA_PID_WRITER" "$LLAMA_PID_OPUS" 2>/dev/null || true

    echo ""
    echo "=== Done: variant $VARIANT ==="
    echo "Output at: ${S3_OUT}/"
    echo "Terminating instance..."
    INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id \
        -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT http://169.254.169.254/latest/api/token \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')")
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$EC2_REGION"
}

# ============================================================
# MODE: --userdata <variant>
# Prints the EC2 user-data bootstrap script.
# ============================================================

mode_userdata() {
    VARIANT=${1:-"<variant>"}
    cat <<USERDATA
#!/bin/bash
# EC2 user-data bootstrap for GPU expert gen: $VARIANT
set -euo pipefail

LOG=/home/ubuntu/gen.log
touch \$LOG && chown ubuntu:ubuntu \$LOG
echo "=== gpu-expert-gen bootstrap \$(date) ===" >> \$LOG

cd /home/ubuntu
aws s3 cp s3://${S3_BUCKET}/${S3_JOB_PREFIX}/ec2-gpu-expert-gen.sh ./ec2-gpu-expert-gen.sh >> \$LOG 2>&1
chmod +x ec2-gpu-expert-gen.sh
chown ubuntu:ubuntu ec2-gpu-expert-gen.sh

sudo -u ubuntu -H bash /home/ubuntu/ec2-gpu-expert-gen.sh --run ${VARIANT} >> \$LOG 2>&1
USERDATA
}

# ============================================================
# MODE: --launch
# Spins up 3 g7e.2xlarge spot instances, one per variant.
# ============================================================

mode_launch() {
    echo "=== Launching 3 x ${EC2_INSTANCE_TYPE} spot instances ==="

    echo "[1] Resolving caller IP..."
    MY_IP=$(curl -sf https://checkip.amazonaws.com || curl -sf https://ifconfig.me)
    [ -n "$MY_IP" ] || die "Could not determine public IP"
    echo "  Caller IP: $MY_IP"

    echo "[2] Setting up security group..."
    SG_NAME="gpu-expert-gen-sg"
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$EC2_VPC_ID" \
        --query 'SecurityGroups[0].GroupId' --output text \
        --region "$EC2_REGION" 2>/dev/null || echo "")

    if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SG_NAME" \
            --description "GPU expert gen SSH access" \
            --vpc-id "$EC2_VPC_ID" \
            --region "$EC2_REGION" \
            --query 'GroupId' --output text)
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr "${MY_IP}/32" \
            --region "$EC2_REGION"
        echo "  Created SG: $SG_ID"
    else
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr "${MY_IP}/32" \
            --region "$EC2_REGION" 2>/dev/null || true
        echo "  Reusing SG: $SG_ID"
    fi

    echo ""
    for VARIANT in "${VARIANTS[@]}"; do
        echo "[launch] Variant: $VARIANT ..."
        USERDATA_B64=$(mode_userdata "$VARIANT" | base64 -w 0)

        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id "$EC2_AMI" \
            --instance-type "$EC2_INSTANCE_TYPE" \
            --key-name "$EC2_KEY_NAME" \
            --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"DeleteOnTermination\":true,\"VolumeSize\":${EC2_STORAGE_GB},\"VolumeType\":\"gp3\",\"Iops\":3000,\"Throughput\":125}}]" \
            --network-interfaces "[{\"AssociatePublicIpAddress\":true,\"DeviceIndex\":0,\"Groups\":[\"$SG_ID\"]}]" \
            --iam-instance-profile "Arn=$EC2_IAM_PROFILE" \
            --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$EC2_MAX_SPOT_PRICE\"}}" \
            --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
            --tag-specifications "[{\"ResourceType\":\"instance\",\"Tags\":[{\"Key\":\"Name\",\"Value\":\"gpu-expert-gen-${VARIANT}\"}]}]" \
            --user-data "$USERDATA_B64" \
            --region "$EC2_REGION" \
            --query 'Instances[0].InstanceId' --output text)

        echo "  Instance: $INSTANCE_ID"

        PUBLIC_IP=""
        while [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "None" ]; do
            sleep 3
            PUBLIC_IP=$(aws ec2 describe-instances \
                --instance-ids "$INSTANCE_ID" \
                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                --output text --region "$EC2_REGION" 2>/dev/null || echo "")
        done

        echo "  Public IP: $PUBLIC_IP"
        echo "  Monitor:  ssh -i ~/.ssh/${EC2_KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f ~/gen.log'"
        echo ""
    done

    echo "=== All 3 instances launched ==="
    echo ""
    echo "Output will appear at:"
    for VARIANT in "${VARIANTS[@]}"; do
        echo "  s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/${VARIANT}/"
    done
}

# ============================================================
# MAIN
# ============================================================

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 --package              # validate, bundle 3 variants, stage to S3"
    echo "  $0 --launch               # spin up 3 x g7e.2xlarge spot instances"
    echo "  $0 --run <variant>        # run on EC2 (t60 | t75 | t90)"
    echo "  $0 --userdata <variant>   # print EC2 user-data bootstrap"
    exit 1
fi

case "$1" in
    --package)  mode_package ;;
    --launch)   mode_launch  ;;
    --run)
        [ -n "${2:-}" ] || die "--run requires a variant argument (t60, t75, t90)"
        mode_run "$2"
        ;;
    --userdata)
        [ -n "${2:-}" ] || die "--userdata requires a variant argument"
        mode_userdata "$2"
        ;;
    *) die "Unknown option: $1" ;;
esac
