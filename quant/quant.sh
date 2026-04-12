#!/bin/bash
set -euo pipefail

# --- CONTEXT ---
S3_FILES_BUCKET="mad-lab-files"
S3_MOUNT_POINT="$HOME/s3/mad-lab-files"
JOBS_DIR="$S3_MOUNT_POINT/quant-jobs"
OUTPUT_DIR="$S3_MOUNT_POINT/quant-output"
LLAMA_CPP_REPO="https://github.com/kmbandy/llama.cpp"

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
import sys

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
    echo "[0/11] Setting up S3 mount..."
    if ! command -v mount-s3 &> /dev/null; then
        echo "  Installing mountpoint-s3..."
        wget -q https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
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
    echo "[1/5] Reading and validating quant.yaml..."
    
    if [ ! -f "quant.yaml" ]; then
        echo "FAIL: quant.yaml not found"
        exit 1
    fi

    ensure_pyyaml
    # Source variables from yaml
    eval $(parse_yaml)

    # Validate model_dir
    if [ -d "$CONFIG_MODEL_DIR" ] && ls "$CONFIG_MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
        echo "PASS: model_dir '$CONFIG_MODEL_DIR' exists and contains .safetensors files"
    else
        echo "FAIL: model_dir '$CONFIG_MODEL_DIR' is invalid or has no .safetensors"
        exit 1
    fi

    # Validate imatrix
    if [ "$CONFIG_IMATRIX" = "true" ]; then
        if [ -f "./imatrix/calibration.txt" ]; then
            echo "PASS: imatrix/calibration.txt exists"
        else
            echo "FAIL: imatrix/calibration.txt not found"
            exit 1
        fi
    fi

    echo "[2/5] Creating staging directory..."
    STAGING_DIR="quant-job-$CONFIG_NAME"
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"
    
    cp quant.yaml "$STAGING_DIR/"
    cp "$0" "$STAGING_DIR/quant.sh"
    cp -r "$CONFIG_MODEL_DIR" "$STAGING_DIR/model"
    
    if [ "$CONFIG_IMATRIX" = "true" ]; then
        cp -r imatrix "$STAGING_DIR/"
    fi

    echo "[3/5] Creating tarball..."
    TARBALL="$STAGING_DIR.tar.gz"
    tar -czf "$TARBALL" "$STAGING_DIR"
    du -sh "$TARBALL"

    echo "[4/5] Copying tarball to S3 mount..."
    mkdir -p "$JOBS_DIR"
    cp "$TARBALL" "$JOBS_DIR/"
    echo "Job staged. On EC2 run: bash quant.sh --run $CONFIG_NAME"

    echo "[5/5] Cleaning up local staging dir..."
    rm -rf "$STAGING_DIR"
}

# --- MODE: --run ---

mode_run() {
    JOB_NAME=$1
    TARBALL="quant-job-$JOB_NAME.tar.gz"

    setup_s3_mount

    echo "[1/11] Unpacking..."
    cp "$JOBS_DIR/$TARBALL" .
    tar -xzf "$TARBALL"
    cd "quant-job-$JOB_NAME"

    # Load config from unpacked dir
    ensure_pyyaml
    eval $(parse_yaml)

    echo "[2/11] Installing system deps..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq git cmake build-essential python3-pip python3-venv

    if [ "$CONFIG_GPU" = "true" ]; then
        if ! command -v nvcc &> /dev/null; then
            sudo apt-get install -y -qq nvidia-cuda-toolkit
        fi
    fi

    echo "[3/11] Setting up Python venv..."
    python3 -m venv /tmp/quant-venv
    source /tmp/quant-venv/bin/activate
    pip install -q huggingface-hub gguf pyyaml

    echo "[4/11] Mounting NVMe instance storage..."
    NVME_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | head -1)
    if [ -z "$NVME_DEV" ]; then
        echo "ERROR: No NVMe device found"
        exit 1
    fi
    if ! blkid "$NVME_DEV" | grep -q ext4; then
        sudo mkfs.ext4 -F "$NVME_DEV"
    fi
    WORK_DIR="/mnt/nvme"
    sudo mkdir -p "$WORK_DIR"
    if ! mountpoint -q "$WORK_DIR"; then
        sudo mount "$NVME_DEV" "$WORK_DIR"
    fi
    sudo chmod 777 "$WORK_DIR"

    echo "[5/11] Building llama.cpp..."
    if [ ! -d "$WORK_DIR/llama.cpp" ]; then
        git clone --depth=1 "$LLAMA_CPP_REPO" "$WORK_DIR/llama.cpp"
    fi
    cd "$WORK_DIR/llama.cpp"
    
    if [ "$CONFIG_GPU" = "true" ]; then
        cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF \
              -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
    else
        cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
              -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
    fi
    cmake --build build --target llama-quantize llama-imatrix -j "$(nproc)"
    cd "$OLDPWD"

    echo "[6/11] Converting to F16 GGUF..."
    START_CONVERT=$(date +%s)
    F16_OUT="$WORK_DIR/$CONFIG_OUTPUT_PREFIX-f16.gguf"
    python3 "$WORK_DIR/llama.cpp/convert_hf_to_gguf.py" \
        model/ \
        --outtype f16 \
        --outfile "$F16_OUT"
    END_CONVERT=$(date +%s)
    echo "F16 Size: $(du -sh "$F16_OUT" | cut -f1)"
    echo "Convert time: $((END_CONVERT - START_CONVERT))s"

    IMATRIX_FILE=""
    if [ "$CONFIG_IMATRIX" = "true" ]; then
        echo "[7/11] Generating imatrix..."
        START_IMATRIX=$(date +%s)
        IMATRIX_FILE="$WORK_DIR/$CONFIG_OUTPUT_PREFIX.imatrix"
        "$WORK_DIR/llama.cpp/build/bin/llama-imatrix" \
            -m "$F16_OUT" \
            -f imatrix/calibration.txt \
            -o "$IMATRIX_FILE" \
            --chunks 128
        END_IMATRIX=$(date +%s)
        echo "Imatrix time: $((END_IMATRIX - START_IMATRIX))s"
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
            $IMATRIX_ARG \
            "$F16_OUT" \
            "$OUT_FILE" \
            "$QUANT" \
            "$(nproc)"
        
        END_Q=$(date +%s)
        ELAPSED=$((END_Q - START_Q))
        SIZE=$(du -sh "$OUT_FILE" | cut -f1)
        echo "  Done: $SIZE, ${ELAPSED}s"
        RESULTS+="$QUANT | $SIZE | ${ELAPSED}s\n"
    done

    echo "[9/11] Cleanup..."
    rm -f "$F16_OUT"
    if [ -n "$IMATRIX_FILE" ]; then
        rm -f "$IMATRIX_FILE"
    fi

    echo "[10/11] Writing output to S3..."
    mkdir -p "$OUTPUT_DIR"
    for QUANT in $CONFIG_QUANTS; do
        OUT_FILE="$WORK_DIR/$CONFIG_OUTPUT_PREFIX-$QUANT.gguf"
        cp "$OUT_FILE" "$OUTPUT_DIR/"
        echo "Uploaded: $CONFIG_OUTPUT_PREFIX-$QUANT.gguf ($(du -sh "$OUT_FILE" | cut -f1))"
    done

    echo "[11/11] Summary"
    echo "----------------------------------------"
    echo -e "quant type | size | time"
    echo -e "$RESULTS"
    echo "----------------------------------------"
    echo "All done. Terminate this instance now."
}

# --- MAIN ---

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 --package"
    echo "  $0 --run <job_name>"
    exit 1
fi

case "$1" in
    --package)
        mode_package
        ;;
    --run)
        if [ -z "${2:-}" ]; then
            echo "Error: --run requires a job name"
            exit 1
        fi
        mode_run "$2"
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
esac
