#!/bin/bash
set -euo pipefail

# --- CONTEXT ---
S3_FILES_BUCKET="mad-lab-files"
S3_MOUNT_POINT="$HOME/s3/mad-lab-files"
JOBS_DIR="$S3_MOUNT_POINT/quant-jobs"
OUTPUT_DIR="$S3_MOUNT_POINT/quant-output"
LLAMA_CPP_REPO="https://github.com/kmbandy/llama.cpp"
MOUNTPOINT_S3_VERSION="1.22.2"

# --- HELPERS ---

ensure_pyyaml() {
    if ! python3 -c "import yaml" &>/dev/null; then
        echo "Installing pyyaml..."
        python3 -m pip install pyyaml -q 2>/dev/null || pip install pyyaml -q
    fi
}

parse_yaml() {
    python3 - <<EOF
import yaml

with open('quant.yaml', 'r') as f:
    config = yaml.safe_load(f)

for key, value in config.items():
    if isinstance(value, list):
        print(f"export CONFIG_{key.upper()}='{' '.join(value)}'")
    elif isinstance(value, bool):
        print(f"export CONFIG_{key.upper()}={'true' if value else 'false'}")
    else:
        print(f"export CONFIG_{key.upper()}='{value}'")
EOF
}

setup_s3_mount() {
    if ! command -v mount-s3 &>/dev/null; then
        echo "  Installing mountpoint-s3 ${MOUNTPOINT_S3_VERSION}..."
        wget -q "https://s3.amazonaws.com/mountpoint-s3-release/${MOUNTPOINT_S3_VERSION}/x86_64/mount-s3.deb"
        sudo apt-get install -y ./mount-s3.deb -qq
        rm mount-s3.deb
    fi
    mkdir -p "$S3_MOUNT_POINT"
    if ! mountpoint -q "$S3_MOUNT_POINT"; then
        echo "  Mounting s3://$S3_FILES_BUCKET..."
        mount-s3 "$S3_FILES_BUCKET" "$S3_MOUNT_POINT"
    fi
}

# --- MODE: --package ---

mode_package() {
    echo "[1/6] Reading and validating quant.yaml..."

    if [ ! -f "quant.yaml" ]; then
        echo "FAIL: quant.yaml not found"
        exit 1
    fi

    ensure_pyyaml
    eval $(parse_yaml)

    if [ -d "$CONFIG_MODEL_DIR" ] && ls "$CONFIG_MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
        echo "PASS: model_dir '$CONFIG_MODEL_DIR' exists and contains .safetensors files"
    else
        echo "FAIL: model_dir '$CONFIG_MODEL_DIR' is invalid or has no .safetensors"
        exit 1
    fi

    CALIBRATION_FILE="imatrix/${CONFIG_IMATRIX_CALIBRATION:-calibration.txt}"
    if [ "$CONFIG_IMATRIX" = "true" ]; then
        if [ -f "./$CALIBRATION_FILE" ]; then
            echo "PASS: $CALIBRATION_FILE exists"
        else
            echo "FAIL: $CALIBRATION_FILE not found"
            exit 1
        fi
    fi

    echo "[2/6] Creating staging directory..."
    STAGING_DIR="quant-job-$CONFIG_NAME"
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"

    cp quant.yaml "$STAGING_DIR/"
    cp "$0" "$STAGING_DIR/quant.sh"

    if [ "$CONFIG_IMATRIX" = "true" ]; then
        cp -r imatrix "$STAGING_DIR/"
    fi

    echo "[3/6] Pre-downloading Python wheels..."
    mkdir -p "$STAGING_DIR/wheels"
    if pip download -q -d "$STAGING_DIR/wheels/" huggingface-hub gguf pyyaml transformers accelerate 2>/dev/null; then
        echo "  Bundled $(ls "$STAGING_DIR/wheels/" | wc -l) wheel packages"
    else
        echo "  WARNING: wheel pre-download failed — EC2 will install from PyPI"
        rm -rf "$STAGING_DIR/wheels"
    fi

    echo "[4/6] Creating tarball..."
    TARBALL="$STAGING_DIR.tar.gz"
    tar -czf "$TARBALL" "$STAGING_DIR"
    du -sh "$TARBALL"

    echo "[5/6] Staging to S3..."
    # Use aws s3 directly — mountpoint-s3 doesn't allow overwrite/delete by default
    aws s3 cp "$TARBALL" "s3://${S3_FILES_BUCKET}/quant-jobs/$TARBALL"
    aws s3 cp "$0" "s3://${S3_FILES_BUCKET}/quant-jobs/quant.sh"
    echo "  Tarball + quant.sh staged"

    # Sync model to S3 (aws s3 sync skips files that already exist)
    S3_MODEL_PREFIX="s3://${S3_FILES_BUCKET}/models/$CONFIG_NAME"
    EXISTING=0
    EXISTING=$(aws s3 ls "${S3_MODEL_PREFIX}/" 2>/dev/null | wc -l) || true
    if [ "$EXISTING" -eq 0 ]; then
        echo "  Uploading model to S3 (this may take a while)..."
        aws s3 sync "$CONFIG_MODEL_DIR/" "${S3_MODEL_PREFIX}/" \
            --exclude '*/.cache/*' \
            --no-progress
        echo "  Model staged at ${S3_MODEL_PREFIX}"
    else
        echo "  Model already in S3, skipping upload"
    fi

    echo "[6/6] Cleaning up..."
    rm -rf "$STAGING_DIR"
    rm -f "$TARBALL"

    echo ""
    echo "Job staged. On EC2 run:"
    echo "  bash quant.sh --run $CONFIG_NAME"
    echo ""
    echo "Or get user-data bootstrap:"
    echo "  bash quant.sh --userdata $CONFIG_NAME"
}

# --- MODE: --run ---

mode_run() {
    JOB_NAME=$1
    TARBALL="quant-job-$JOB_NAME.tar.gz"

    echo "[1/11] Unpacking job..."
    aws s3 cp "s3://${S3_FILES_BUCKET}/quant-jobs/$TARBALL" . --no-progress
    tar -xzf "$TARBALL"
    cd "quant-job-$JOB_NAME"

    ensure_pyyaml
    eval $(parse_yaml)
    CALIBRATION_FILE="imatrix/${CONFIG_IMATRIX_CALIBRATION:-calibration.txt}"

    echo "[1b/11] Pulling model from S3..."
    S3_MODEL_PREFIX="s3://${S3_FILES_BUCKET}/models/$JOB_NAME"
    if ! aws s3 ls "${S3_MODEL_PREFIX}/" &>/dev/null; then
        echo "ERROR: Model not found in S3 at ${S3_MODEL_PREFIX}"
        exit 1
    fi
    mkdir -p model
    aws s3 sync "${S3_MODEL_PREFIX}/" model/ --no-progress
    echo "  Model ready ($(du -sh model/ | cut -f1))"

    echo "[2/11] Installing system deps..."
    MISSING_PKGS=()
    command -v git    >/dev/null || MISSING_PKGS+=(git)
    command -v cmake  >/dev/null || MISSING_PKGS+=(cmake)
    command -v rsync  >/dev/null || MISSING_PKGS+=(rsync)
    dpkg -s build-essential &>/dev/null || MISSING_PKGS+=(build-essential)
    dpkg -s python3-venv    &>/dev/null || MISSING_PKGS+=(python3-venv)

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo "  Installing: ${MISSING_PKGS[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq --no-install-recommends "${MISSING_PKGS[@]}"
    else
        echo "  All system deps present, skipping apt"
    fi

    echo "[3/11] Setting up Python venv..."
    python3 -m venv /tmp/quant-venv
    source /tmp/quant-venv/bin/activate

    if [ -d "wheels" ] && [ "$(ls -A wheels/ 2>/dev/null)" ]; then
        echo "  Installing from bundled wheels..."
        pip install -q --no-index --find-links wheels/ huggingface-hub gguf pyyaml transformers accelerate 2>/dev/null || \
            pip install -q huggingface-hub gguf pyyaml transformers accelerate
    else
        echo "  Installing from PyPI..."
        pip install -q huggingface-hub gguf pyyaml transformers accelerate
    fi

    echo "[4/11] Mounting NVMe instance storage..."
    # Find instance store NVMe — the non-root disk (root is whichever has '/' in its partitions)
    ROOT_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | while read dev; do
        lsblk -no MOUNTPOINT "$dev" 2>/dev/null | grep -q '^/$' && echo "$dev" && break
    done)
    NVME_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | while read dev; do
        [ "$dev" != "$ROOT_DEV" ] && echo "$dev" && break
    done)
    if [ -z "$NVME_DEV" ]; then
        echo "ERROR: No instance store NVMe found"
        lsblk
        exit 1
    fi
    echo "  Instance store device: $NVME_DEV"

    # If already mounted (e.g. DLAMI pre-mounts at /opt/dlami/nvme), use that mount point
    EXISTING_MOUNT=$(lsblk -no MOUNTPOINT "$NVME_DEV" 2>/dev/null | grep -v '^$' | head -1)
    if [ -n "$EXISTING_MOUNT" ]; then
        WORK_DIR="$EXISTING_MOUNT"
        echo "  Already mounted at $WORK_DIR — using it"
    else
        if ! blkid "$NVME_DEV" | grep -q ext4; then
            sudo mkfs.ext4 -F "$NVME_DEV"
        fi
        WORK_DIR="/mnt/nvme"
        sudo mkdir -p "$WORK_DIR"
        sudo mount "$NVME_DEV" "$WORK_DIR"
    fi
    sudo chmod 777 "$WORK_DIR"

    echo "[5/11] Setting up llama.cpp..."
    # Add common CUDA bin paths to PATH
    for cuda_path in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -d "$cuda_path" ] && export PATH="$cuda_path:$PATH"
    done

    # If GPU requested and nvcc missing, try to install cuda-toolkit via apt
    if [ "$CONFIG_GPU" = "true" ] && ! command -v nvcc &>/dev/null; then
        echo "  nvcc not found — attempting cuda-toolkit install via apt..."
        CUDA_PKG=$(apt-cache search cuda-toolkit 2>/dev/null | grep -oP 'cuda-toolkit-[0-9]+-[0-9]+' | sort -V | tail -1)
        if [ -n "$CUDA_PKG" ]; then
            sudo apt-get install -y --no-install-recommends "$CUDA_PKG" -qq
            for cuda_path in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
                [ -d "$cuda_path" ] && export PATH="$cuda_path:$PATH"
            done
        else
            echo "  WARNING: no cuda-toolkit package found — falling back to CPU build"
        fi
    fi

    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo "cpu")
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' | tr '.' '_' || echo "none")
    CACHE_KEY="sm${GPU_ARCH}_cuda${CUDA_VER}"
    LLAMA_BIN_S3="s3://${S3_FILES_BUCKET}/llama-builds/${CACHE_KEY}"

    echo "  GPU arch: sm${GPU_ARCH}  CUDA: ${CUDA_VER}  cache key: ${CACHE_KEY}"

    # Always clone for Python scripts (fast, depth=1)
    if [ ! -d "$WORK_DIR/llama.cpp" ]; then
        echo "  Cloning llama.cpp..."
        git clone --depth=1 --branch master "$LLAMA_CPP_REPO" "$WORK_DIR/llama.cpp"
    fi
    mkdir -p "$WORK_DIR/llama.cpp/build/bin"

    CACHE_HIT=false
    if aws s3 ls "${LLAMA_BIN_S3}/llama-quantize" &>/dev/null; then
        echo "  Cache hit — restoring binaries..."
        aws s3 cp "${LLAMA_BIN_S3}/llama-quantize" "$WORK_DIR/llama.cpp/build/bin/llama-quantize" --no-progress
        aws s3 cp "${LLAMA_BIN_S3}/llama-imatrix"  "$WORK_DIR/llama.cpp/build/bin/llama-imatrix"  --no-progress
        chmod +x "$WORK_DIR/llama.cpp/build/bin/llama-quantize"
        chmod +x "$WORK_DIR/llama.cpp/build/bin/llama-imatrix"
        echo "  Skipping build — using cached binaries"
        CACHE_HIT=true
    fi

    if [ "$CACHE_HIT" = "false" ]; then
        echo "  Cache miss — building llama.cpp (this takes 15-20 min)..."
        cd "$WORK_DIR/llama.cpp"
        START_BUILD=$(date +%s)
        if [ "$CONFIG_GPU" = "true" ]; then
            cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF \
                  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
        else
            cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
                  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
        fi
        cmake --build build --target llama-quantize llama-imatrix -j "$(nproc)"
        END_BUILD=$(date +%s)
        echo "  Build time: $((END_BUILD - START_BUILD))s"

        echo "  Caching binaries → S3 (${CACHE_KEY})..."
        aws s3 cp build/bin/llama-quantize "${LLAMA_BIN_S3}/llama-quantize" --no-progress
        aws s3 cp build/bin/llama-imatrix  "${LLAMA_BIN_S3}/llama-imatrix"  --no-progress
        echo "  Cached — future runs will skip the build"
        cd "$OLDPWD"
    fi

    echo "[6/11] Converting to F16 GGUF..."
    START_CONVERT=$(date +%s)
    F16_OUT="$WORK_DIR/$CONFIG_OUTPUT_PREFIX-f16.gguf"
    python3 "$WORK_DIR/llama.cpp/convert_hf_to_gguf.py" \
        model/ \
        --outtype f16 \
        --outfile "$F16_OUT"
    END_CONVERT=$(date +%s)
    echo "  F16 size: $(du -sh "$F16_OUT" | cut -f1)  time: $((END_CONVERT - START_CONVERT))s"

    IMATRIX_FILE=""
    if [ "$CONFIG_IMATRIX" = "true" ]; then
        echo "[7/11] Generating imatrix..."
        START_IMATRIX=$(date +%s)
        IMATRIX_FILE="$WORK_DIR/$CONFIG_OUTPUT_PREFIX.imatrix"
        "$WORK_DIR/llama.cpp/build/bin/llama-imatrix" \
            -m "$F16_OUT" \
            -f "$CALIBRATION_FILE" \
            -o "$IMATRIX_FILE" \
            --chunks 128 \
            -ngl 999
        END_IMATRIX=$(date +%s)
        echo "  Imatrix time: $((END_IMATRIX - START_IMATRIX))s"
    else
        echo "[7/11] Skipping imatrix (not requested)"
    fi

    echo "[8/11] Quantize loop..."
    RESULTS=""
    for QUANT in $CONFIG_QUANTS; do
        echo "  Quantizing to $QUANT..."
        START_Q=$(date +%s)
        OUT_FILE="$WORK_DIR/$CONFIG_OUTPUT_PREFIX-$QUANT.gguf"

        IMATRIX_ARG=""
        if [ "$CONFIG_IMATRIX" = "true" ]; then
            IMATRIX_ARG="--imatrix $IMATRIX_FILE"
        fi

        "$WORK_DIR/llama.cpp/build/bin/llama-quantize" \
            --allow-requantize \
            $IMATRIX_ARG \
            "$F16_OUT" \
            "$OUT_FILE" \
            "$QUANT" \
            "$(nproc)"

        END_Q=$(date +%s)
        ELAPSED=$((END_Q - START_Q))
        SIZE=$(du -sh "$OUT_FILE" | cut -f1)
        echo "  Done: $SIZE in ${ELAPSED}s"
        RESULTS+="$QUANT | $SIZE | ${ELAPSED}s\n"
    done

    echo "[9/11] Cleanup..."
    rm -f "$F16_OUT"
    [ -n "$IMATRIX_FILE" ] && rm -f "$IMATRIX_FILE"

    echo "[10/11] Writing output to S3..."
    for QUANT in $CONFIG_QUANTS; do
        OUT_FILE="$WORK_DIR/$CONFIG_OUTPUT_PREFIX-$QUANT.gguf"
        aws s3 cp "$OUT_FILE" "s3://${S3_FILES_BUCKET}/quant-output/$CONFIG_OUTPUT_PREFIX-$QUANT.gguf" --no-progress
        echo "  Uploaded: $CONFIG_OUTPUT_PREFIX-$QUANT.gguf ($(du -sh "$OUT_FILE" | cut -f1))"
    done

    echo "[11/11] Summary"
    echo "----------------------------------------"
    printf "%-12s | %-8s | %s\n" "quant" "size" "time"
    echo "----------------------------------------"
    echo -e "$RESULTS"
    echo "All done. Terminate this instance now."
}

# --- MODE: --userdata ---

mode_userdata() {
    JOB_NAME=${1:-"<job_name>"}
    cat <<USERDATA
#!/bin/bash
# EC2 user-data bootstrap for quant job: $JOB_NAME
set -euo pipefail

# Pre-create log as ubuntu so the job can write to it
LOG=/home/ubuntu/quant.log
touch \$LOG
chown ubuntu:ubuntu \$LOG
echo "=== quant bootstrap \$(date) ===" >> \$LOG

# Pull quant.sh from S3 and run as ubuntu
cd /home/ubuntu
aws s3 cp s3://${S3_FILES_BUCKET}/quant-jobs/quant.sh ./quant.sh >> \$LOG 2>&1
chmod +x quant.sh
chown ubuntu:ubuntu quant.sh
sudo -u ubuntu -H bash /home/ubuntu/quant.sh --run $JOB_NAME >> \$LOG 2>&1
USERDATA
}

# --- MODE: --publish ---

mode_publish() {
    if [ ! -f "quant.yaml" ]; then
        echo "FAIL: quant.yaml not found"
        exit 1
    fi

    ensure_pyyaml
    eval $(parse_yaml)

    echo "[1/4] Resolving HF token..."
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "  Using HF_TOKEN from environment"
    else
        HF_TOKEN=$(aws ssm get-parameter \
            --name "/mad-lab/hf-token" \
            --with-decryption \
            --query 'Parameter.Value' \
            --output text 2>/dev/null) || true
        if [ -z "$HF_TOKEN" ]; then
            echo "ERROR: No HF token found. Set HF_TOKEN env var or store in SSM at /mad-lab/hf-token"
            exit 1
        fi
        echo "  Retrieved token from SSM"
    fi

    echo "[2/4] Installing huggingface-hub CLI..."
    if ! command -v hf &>/dev/null; then
        pip install -q huggingface-hub
        export PATH="$PATH:$HOME/.local/bin"
    fi

    echo "[3/4] Pulling GGUFs from S3..."
    mkdir -p publish-tmp
    for QUANT in $CONFIG_QUANTS; do
        FILE="$CONFIG_OUTPUT_PREFIX-$QUANT.gguf"
        echo "  Downloading $FILE..."
        aws s3 cp "s3://${S3_FILES_BUCKET}/quant-output/$FILE" "publish-tmp/$FILE" --no-progress
    done

    # Copy calibration data if imatrix was used
    CALIBRATION_FILE="imatrix/${CONFIG_IMATRIX_CALIBRATION:-calibration.txt}"
    if [ "$CONFIG_IMATRIX" = "true" ] && [ -f "$CALIBRATION_FILE" ]; then
        cp "$CALIBRATION_FILE" publish-tmp/imatrix-calibration.txt
    fi

    echo "[4/4] Uploading to HuggingFace..."
    HF_REPO="${CONFIG_HF_REPO:-mad-lab-ai/${CONFIG_OUTPUT_PREFIX}-GGUF}"
    for QUANT in $CONFIG_QUANTS; do
        FILE="publish-tmp/$CONFIG_OUTPUT_PREFIX-$QUANT.gguf"
        echo "  Uploading $QUANT..."
        HF_TOKEN="$HF_TOKEN" hf upload \
            "$HF_REPO" \
            "$FILE" \
            "$CONFIG_OUTPUT_PREFIX-$QUANT.gguf" \
            --repo-type model \
            --commit-message "Add $QUANT quant"
    done

    echo ""
    echo "Published to: https://huggingface.co/$HF_REPO"

    rm -rf publish-tmp
}

# --- MODE: --launch ---

mode_launch() {
    if [ ! -f "quant.yaml" ]; then
        echo "FAIL: quant.yaml not found"
        exit 1
    fi

    ensure_pyyaml
    eval $(parse_yaml)

    echo "[1/4] Resolving caller IP..."
    MY_IP=$(curl -sf https://checkip.amazonaws.com || curl -sf https://ifconfig.me)
    if [ -z "$MY_IP" ]; then
        echo "ERROR: Could not determine public IP"
        exit 1
    fi
    echo "  Caller IP: $MY_IP"

    echo "[2/4] Setting up security group..."
    SG_NAME="quant-job-sg"
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$CONFIG_EC2_VPC_ID" \
        --query 'SecurityGroups[0].GroupId' --output text --region "$CONFIG_EC2_REGION" 2>/dev/null)

    if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
        echo "  Creating security group..."
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SG_NAME" \
            --description "quant job SSH access" \
            --vpc-id "$CONFIG_EC2_VPC_ID" \
            --region "$CONFIG_EC2_REGION" \
            --query 'GroupId' --output text)
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr "${MY_IP}/32" \
            --region "$CONFIG_EC2_REGION"
        echo "  Created: $SG_ID (SSH from $MY_IP)"
    else
        echo "  Reusing: $SG_ID"
        # Ensure current IP is allowed
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr "${MY_IP}/32" \
            --region "$CONFIG_EC2_REGION" 2>/dev/null || true
    fi

    echo "[3/4] Launching spot instance..."
    USERDATA_B64=$(mode_userdata "$CONFIG_NAME" | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$CONFIG_EC2_AMI" \
        --instance-type "$CONFIG_EC2_INSTANCE_TYPE" \
        --key-name "$CONFIG_EC2_KEY_NAME" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"DeleteOnTermination\":true,\"VolumeSize\":${CONFIG_EC2_STORAGE_GB},\"VolumeType\":\"gp3\",\"Iops\":3000,\"Throughput\":125}}]" \
        --network-interfaces "[{\"AssociatePublicIpAddress\":true,\"DeviceIndex\":0,\"Groups\":[\"$SG_ID\"]}]" \
        --iam-instance-profile "Arn=$CONFIG_EC2_IAM_PROFILE" \
        --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$CONFIG_EC2_MAX_SPOT_PRICE\"}}" \
        --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
        --tag-specifications "[{\"ResourceType\":\"instance\",\"Tags\":[{\"Key\":\"Name\",\"Value\":\"quant-$CONFIG_NAME\"}]}]" \
        --user-data "$USERDATA_B64" \
        --region "$CONFIG_EC2_REGION" \
        --query 'Instances[0].InstanceId' --output text)

    echo "  Instance: $INSTANCE_ID"

    echo "[4/4] Waiting for public IP..."
    PUBLIC_IP=""
    while [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "None" ]; do
        sleep 3
        PUBLIC_IP=$(aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text --region "$CONFIG_EC2_REGION" 2>/dev/null)
    done

    echo ""
    echo "Instance running: $PUBLIC_IP"
    echo "Monitor:"
    echo "  ssh -i ~/.ssh/${CONFIG_EC2_KEY_NAME}.pem ubuntu@$PUBLIC_IP 'tail -f ~/quant.log'"
    echo ""
    echo "SSH:"
    echo "  ssh -i ~/.ssh/${CONFIG_EC2_KEY_NAME}.pem ubuntu@$PUBLIC_IP"
}

# --- MAIN ---

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 --package                # validate, bundle, and stage job to S3"
    echo "  $0 --launch                 # launch EC2 spot instance and run job"
    echo "  $0 --publish                # upload GGUFs from S3 to HuggingFace"
    echo "  $0 --run <job_name>         # run job (called automatically on EC2)"
    echo "  $0 --userdata <job_name>    # print EC2 user-data bootstrap script"
    exit 1
fi

case "$1" in
    --package)
        mode_package
        ;;
    --launch)
        mode_launch
        ;;
    --publish)
        mode_publish
        ;;
    --run)
        if [ -z "${2:-}" ]; then
            echo "Error: --run requires a job name"
            exit 1
        fi
        mode_run "$2"
        ;;
    --userdata)
        if [ -z "${2:-}" ]; then
            echo "Error: --userdata requires a job name"
            exit 1
        fi
        mode_userdata "$2"
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
esac
