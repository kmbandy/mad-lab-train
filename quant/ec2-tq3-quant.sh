#!/bin/bash
# EC2 TQ3_1S quantization script
# Usage: bash ec2-tq3-quant.sh <hf_repo> <output_name>
# Example: bash ec2-tq3-quant.sh moonshotai/Kimi-Linear-48B-A3B-Base kimi-linear-48b-tq3_1s
#
# Tested on: i3en.6xlarge, Ubuntu 22.04, 192GB RAM
# Requires: HF_TOKEN env var (if model is gated), AWS credentials with S3 write access

set -euo pipefail

HF_REPO="${1:-moonshotai/Kimi-Linear-48B-A3B-Base}"
OUTPUT_NAME="${2:-kimi-linear-48b-tq3_1s}"
S3_BUCKET="mad-lab-models"
WORK_DIR="/mnt/nvme"

echo "=== TQ3_1S Quant Pipeline ==="
echo "Model  : $HF_REPO"
echo "Output : ${OUTPUT_NAME}.gguf"
echo "S3     : s3://${S3_BUCKET}/${OUTPUT_NAME}.gguf"
echo ""

# ---------------------------------------------------------------------------
# 1. Mount NVMe instance storage (i3en-specific — drives come unformatted)
# ---------------------------------------------------------------------------
echo "[1/6] Mounting NVMe..."
NVME_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk" {print $1}' | grep nvme | head -1)
if [ -z "$NVME_DEV" ]; then
    echo "ERROR: No NVMe device found"
    exit 1
fi
echo "  Device: $NVME_DEV"
if ! blkid "$NVME_DEV" | grep -q ext4; then
    echo "  Formatting $NVME_DEV as ext4..."
    mkfs.ext4 -F "$NVME_DEV"
fi
mkdir -p "$WORK_DIR"
if ! mountpoint -q "$WORK_DIR"; then
    mount "$NVME_DEV" "$WORK_DIR"
fi
echo "  Mounted at $WORK_DIR ($(df -h "$WORK_DIR" | tail -1 | awk '{print $4}') free)"

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "[2/6] Installing dependencies..."
apt-get update -qq
apt-get install -y -qq git cmake build-essential python3-pip python3-venv awscli

python3 -m venv "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
pip install -q huggingface-hub "transformers>=4.57.1" "torch~=2.6.0" sentencepiece protobuf tiktoken gguf

# Build llama.cpp (CPU-only, optimized)
if [ ! -f "$WORK_DIR/llama.cpp/build/bin/llama-quantize" ]; then
    echo "  Building llama.cpp..."
    cd "$WORK_DIR"
    git clone --depth=1 https://github.com/kmbandy/llama.cpp
    cd llama.cpp
    cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target llama-quantize llama-gguf-split -j "$(nproc)"
else
    echo "  llama.cpp already built, skipping"
fi
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# 3. Download model from HuggingFace
# ---------------------------------------------------------------------------
echo "[3/6] Downloading $HF_REPO..."
MODEL_DIR="$WORK_DIR/model"
mkdir -p "$MODEL_DIR"

python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN")
snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$MODEL_DIR",
    token=token,
    ignore_patterns=["*.md", "*.txt", "original/*"],
)
print("Download complete.")
PYEOF

du -sh "$MODEL_DIR"

# ---------------------------------------------------------------------------
# 4. Convert to F16 GGUF
# ---------------------------------------------------------------------------
echo "[4/6] Converting to F16 GGUF..."
F16_PATH="$WORK_DIR/${OUTPUT_NAME}-f16.gguf"
python3 "$WORK_DIR/llama.cpp/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outtype f16 \
    --outfile "$F16_PATH"
echo "  F16 size: $(du -sh "$F16_PATH" | cut -f1)"

# ---------------------------------------------------------------------------
# 5. Quantize to TQ3_1S
# ---------------------------------------------------------------------------
echo "[5/6] Quantizing to TQ3_1S..."
TQ3_PATH="$WORK_DIR/${OUTPUT_NAME}.gguf"
START_T=$(date +%s)
"$WORK_DIR/llama.cpp/build/bin/llama-quantize" \
    --allow-requantize \
    "$F16_PATH" \
    "$TQ3_PATH" \
    TQ3_1S \
    "$(nproc)"
END_T=$(date +%s)
echo "  Quantization time: $((END_T - START_T))s"
echo "  TQ3_1S size: $(du -sh "$TQ3_PATH" | cut -f1)"

# Free up F16 to save space before S3 upload
rm -f "$F16_PATH"

# ---------------------------------------------------------------------------
# 6. Upload to S3
# ---------------------------------------------------------------------------
echo "[6/6] Uploading to s3://${S3_BUCKET}/${OUTPUT_NAME}.gguf ..."
aws s3 cp "$TQ3_PATH" "s3://${S3_BUCKET}/${OUTPUT_NAME}.gguf" \
    --storage-class STANDARD_IA \
    --no-progress
echo ""
echo "=== Done ==="
echo "s3://${S3_BUCKET}/${OUTPUT_NAME}.gguf"
echo "Terminate this instance now."
