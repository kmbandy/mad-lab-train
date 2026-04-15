#!/bin/bash
set -euo pipefail

# ============================================================
# ec2-tool-calls-gen.sh
# Launches a g7e.2xlarge spot instance to generate 50k synthetic
# tool-calling training data using Qwen3.5-122B-A10B (UD-IQ4_NL).
#
# Usage:
#   ./ec2-tool-calls-gen.sh --package            # bundle pipeline, stage to S3
#   ./ec2-tool-calls-gen.sh --launch             # spin up spot instance
#   ./ec2-tool-calls-gen.sh --run                # run on EC2 (called by user-data)
#   ./ec2-tool-calls-gen.sh --userdata           # print bootstrap script
# ============================================================

# --- S3 / model config ---
S3_BUCKET="mad-lab-files"
S3_JOB_PREFIX="tool-calls-jobs"
S3_OUTPUT_PREFIX="tool-calls-output"
S3_LLAMA_BUILD_PREFIX="llama-builds"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp"

# Qwen3.5-122B MoE — pull direct from HF on EC2 (public, ~60GB)
# UD-IQ4_NL is a 3-part split under the UD-IQ4_NL/ subfolder
HF_REPO="unsloth/Qwen3.5-122B-A10B-GGUF"
HF_SUBFOLDER="UD-IQ4_NL"
HF_FILENAME_PREFIX="Qwen3.5-122B-A10B-UD-IQ4_NL"
HF_SHARDS=3

# --- Generation config ---
LLAMA_PORT=8080
LLAMA_PARALLEL=96       # 96 slots × 2048 ctx = ~90GB total, fits 96GB with 59GB model (7GB headroom)
LLAMA_CTX=2048          # 2048 per slot — generator prompts are ~1100 tokens + 1500 output, needs 2048 min
PER_SCENARIO=5000       # 10 scenarios × 5000 = 50k total samples
CONCURRENCY=256

# --- EC2 config ---
EC2_REGION="us-east-2"
EC2_AMI="ami-03bda78a7c7c13b45"   # DLAMI with CUDA (us-east-2)
EC2_INSTANCE_TYPE="g7e.2xlarge"
EC2_KEY_NAME="mad-lab-key"
EC2_IAM_PROFILE="arn:aws:iam::080869524552:instance-profile/mad-lab-ec2-quant-role"
EC2_VPC_ID="vpc-0a8e766998c7d5e23"
EC2_MAX_SPOT_PRICE="0.85"
EC2_STORAGE_GB=100

JOB_NAME="tool-calls-qwen35-122b"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPC_DIR="$SCRIPT_DIR"

# ============================================================
# HELPERS
# ============================================================

die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_llama() {
    local port=$1
    local max_wait=600
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
# ============================================================

mode_package() {
    echo "=== Packaging tool-call generation pipeline ==="

    MCP_DIR="$SCRIPT_DIR/../../mad-lab-mcp"
    [ -f "$MCP_DIR/bin/generate_tool_calls.py" ] \
        || die "generate_tool_calls.py not found at $MCP_DIR/bin/"

    STAGING="$SCRIPT_DIR/../.staging-${JOB_NAME}"
    rm -rf "$STAGING"
    mkdir -p "$STAGING"

    # Copy generation scripts
    cp "$MCP_DIR/bin/generate_tool_calls.py" "$STAGING/"
    cp "$MCP_DIR/bin/generate_tool_calls_b.py" "$STAGING/" 2>/dev/null || true

    echo "  Copied generation scripts"

    # Tarball
    TARBALL="${JOB_NAME}.tar.gz"
    tar -czf "$TARBALL" -C "$STAGING/.." "$(basename "$STAGING")"
    echo "  Tarball: $(du -sh "$TARBALL" | cut -f1)"

    # Stage to S3
    aws s3 cp "$TARBALL" "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}" \
        --region "$EC2_REGION" --no-progress
    echo "  Staged: s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}"

    # Stage this script
    aws s3 cp "$0" "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/ec2-tool-calls-gen.sh" \
        --region "$EC2_REGION" --no-progress
    echo "  Staged: ec2-tool-calls-gen.sh"

    rm -rf "$STAGING" "$TARBALL"

    echo ""
    echo "=== Package complete ==="
    echo ""
    echo "Launch with:"
    echo "  $0 --launch"
}

# ============================================================
# MODE: --run
# Runs on EC2.
# ============================================================

mode_run() {
    TARBALL="${JOB_NAME}.tar.gz"
    STAGING_NAME=".staging-${JOB_NAME}"

    echo "=== Tool call gen run: Qwen3.5-122B-A10B ==="

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

    for cuda_path in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -d "$cuda_path" ] && export PATH="$cuda_path:$PATH"
    done

    echo "[2] Unpacking pipeline..."
    aws s3 cp "s3://${S3_BUCKET}/${S3_JOB_PREFIX}/${TARBALL}" . \
        --region "$EC2_REGION" --no-progress
    tar -xzf "$TARBALL"
    PIPELINE_DIR="$(pwd)/${STAGING_NAME}"
    echo "  Pipeline at: $PIPELINE_DIR"

    echo "[3] Setting up Python venv..."
    python3 -m venv /tmp/gen-venv
    source /tmp/gen-venv/bin/activate
    pip install -q httpx huggingface_hub

    echo "[4] Mounting NVMe instance store..."
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
        echo "  WARNING: No NVMe found — using /tmp"
        WORK_DIR="/tmp/tool-calls-work"
        mkdir -p "$WORK_DIR"
    fi

    mkdir -p "${WORK_DIR}/output"

    echo "[5] Downloading Qwen3.5-122B-A10B UD-IQ4_NL (3 shards) from HuggingFace..."
    MODEL_DIR="${WORK_DIR}/UD-IQ4_NL"
    MODEL_PATH="${MODEL_DIR}/${HF_FILENAME_PREFIX}-00001-of-00003.gguf"
    mkdir -p "$MODEL_DIR"
    if [ -f "$MODEL_PATH" ]; then
        echo "  Model shards already present"
    else
        python3 - <<PYEOF
from huggingface_hub import hf_hub_download
total = ${HF_SHARDS}
for i in range(1, total + 1):
    fname = f"${HF_SUBFOLDER}/${HF_FILENAME_PREFIX}-{i:05d}-of-{total:05d}.gguf"
    print(f"  Downloading {fname}...")
    hf_hub_download(
        repo_id="${HF_REPO}",
        filename=fname,
        local_dir="${WORK_DIR}",
    )
    print(f"  Shard {i}/{total} done")
PYEOF
        echo "  All shards downloaded"
    fi
    echo "  Model ready ($(du -sh "$MODEL_DIR" | cut -f1))"

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
        echo "  Cache miss — building llama.cpp with CUDA (~15 min)..."
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
    LLAMA_CTX_TOTAL=$(( LLAMA_CTX * LLAMA_PARALLEL ))

    echo "[7] Starting llama-server (parallel=${LLAMA_PARALLEL} ctx_per_slot=${LLAMA_CTX})..."
    "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$LLAMA_PORT" \
        --parallel "$LLAMA_PARALLEL" \
        --ctx-size "$LLAMA_CTX_TOTAL" \
        -ngl 999 \
        --log-disable \
        > "${WORK_DIR}/llama-server.log" 2>&1 &
    LLAMA_PID=$!
    echo "  Server PID: $LLAMA_PID"
    wait_for_llama "$LLAMA_PORT"

    echo "[8] Generating 50k tool-calling samples (${PER_SCENARIO} per scenario)..."
    OUT_FILE="${WORK_DIR}/output/tool_calls.jsonl"
    cd "$PIPELINE_DIR"
    START=$(date +%s)

    python3 generate_tool_calls.py \
        --endpoint "http://localhost:${LLAMA_PORT}/v1/chat/completions" \
        --per-scenario "$PER_SCENARIO" \
        --output "$OUT_FILE" \
        >> /home/ubuntu/gen-tool-calls.log 2>&1

    END=$(date +%s)
    COUNT=$(wc -l < "$OUT_FILE" 2>/dev/null || echo 0)
    echo "  Done: ${COUNT} samples in $((END - START))s"

    kill "$LLAMA_PID" 2>/dev/null || true

    echo "[9] Uploading output to S3..."
    S3_OUT="s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}"
    if [ -f "$OUT_FILE" ]; then
        aws s3 cp "$OUT_FILE" "${S3_OUT}/tool_calls.jsonl" \
            --region "$EC2_REGION" --no-progress
        echo "  Uploaded tool_calls.jsonl (${COUNT} samples)"
    else
        echo "  [warn] No output file found"
    fi

    echo ""
    echo "=== Done: ${COUNT} tool-call samples ==="
    echo "Output at: ${S3_OUT}/tool_calls.jsonl"
    echo "Terminating instance..."

    INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id \
        -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT http://169.254.169.254/latest/api/token \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')")
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$EC2_REGION"
}

# ============================================================
# MODE: --userdata
# ============================================================

mode_userdata() {
    cat <<USERDATA
#!/bin/bash
# EC2 user-data bootstrap for tool-call gen
set -euo pipefail

LOG=/home/ubuntu/gen.log
touch \$LOG && chown ubuntu:ubuntu \$LOG
echo "=== tool-calls-gen bootstrap \$(date) ===" >> \$LOG

cd /home/ubuntu
aws s3 cp s3://${S3_BUCKET}/${S3_JOB_PREFIX}/ec2-tool-calls-gen.sh ./ec2-tool-calls-gen.sh >> \$LOG 2>&1
chmod +x ec2-tool-calls-gen.sh
chown ubuntu:ubuntu ec2-tool-calls-gen.sh

sudo -u ubuntu -H bash /home/ubuntu/ec2-tool-calls-gen.sh --run >> \$LOG 2>&1
USERDATA
}

# ============================================================
# MODE: --launch
# ============================================================

mode_launch() {
    echo "=== Launching ${EC2_INSTANCE_TYPE} spot instance for tool-call gen ==="

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

    USERDATA_B64=$(mode_userdata | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$EC2_AMI" \
        --instance-type "$EC2_INSTANCE_TYPE" \
        --key-name "$EC2_KEY_NAME" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"DeleteOnTermination\":true,\"VolumeSize\":${EC2_STORAGE_GB},\"VolumeType\":\"gp3\",\"Iops\":3000,\"Throughput\":125}}]" \
        --network-interfaces "[{\"AssociatePublicIpAddress\":true,\"DeviceIndex\":0,\"Groups\":[\"$SG_ID\"]}]" \
        --iam-instance-profile "Arn=$EC2_IAM_PROFILE" \
        --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$EC2_MAX_SPOT_PRICE\"}}" \
        --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
        --tag-specifications "[{\"ResourceType\":\"instance\",\"Tags\":[{\"Key\":\"Name\",\"Value\":\"tool-calls-gen\"}]}]" \
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
    echo ""
    echo "=== Launch complete ==="
    echo "  SSH: ssh -i ~/.ssh/mad-lab-key.pem ubuntu@${PUBLIC_IP}"
    echo "  Log: ssh ... tail -f /home/ubuntu/gen.log"
    echo "  Output: s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/tool_calls.jsonl"
}

# ============================================================
# DISPATCH
# ============================================================

case "${1:-}" in
    --package)  mode_package ;;
    --launch)   mode_launch ;;
    --run)      mode_run ;;
    --userdata) mode_userdata ;;
    *)
        echo "Usage: $0 --package | --launch | --run | --userdata"
        echo ""
        echo "  --package   Bundle pipeline + stage to S3"
        echo "  --launch    Spin up spot instance"
        echo "  --run       Run on EC2 (called by user-data)"
        echo "  --userdata  Print bootstrap script"
        exit 1
        ;;
esac
