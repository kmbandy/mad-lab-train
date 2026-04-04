#!/usr/bin/env bash
# Overnight Qwen3-1.7B fine-tune chain
# Runs: data gen → technical expert → sentiment expert
# Start with: bash run_overnight.sh 2>&1 | tee overnight.log

set -e
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Overnight fine-tune chain starting ==="

# ── Free VRAM: stop tier-1 servers ────────────────────────────────────────────
log "Stopping tier-1 llama servers..."
systemctl --user stop llama-server-phi.service llama-server-qwen-sentiment.service || true
sleep 3

# ── Generate data ──────────────────────────────────────────────────────────────
log "Generating technical data..."
python3 generate_technical_data.py

log "Generating sentiment data..."
python3 generate_sentiment_data.py

# ── Technical fine-tune ────────────────────────────────────────────────────────
log "=== Starting TECHNICAL fine-tune ==="
python3 qwen3_finetune.py --domain technical
log "=== TECHNICAL fine-tune complete ==="

# ── Sentiment fine-tune ────────────────────────────────────────────────────────
log "=== Starting SENTIMENT fine-tune ==="
python3 qwen3_finetune.py --domain sentiment
log "=== SENTIMENT fine-tune complete ==="

log "=== All done. Both adapters in models/ ==="
log "Next step: mergekit MOEification (see moe_config.yaml)"
