#!/bin/bash
# Waits for stats + math ZIM extractions to finish, preps dataset, then kicks off training.
# Run in background — safe to leave overnight.

LOG="/home/kmbandy/mad-lab-mcp/datasets/quant_train_launcher.log"
DATASETS_DIR="/home/kmbandy/mad-lab-mcp/datasets"
TRAINING_DIR="/home/kmbandy/mad-lab-dnd/training"
PYTHON="$HOME/axolotl-env/bin/python3"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== quant-train-launcher started ==="

# -----------------------------------------------------------------------
# 1. Wait for stats + math extractions to complete
# -----------------------------------------------------------------------
log "Waiting for stats and math extractions to finish..."

while true; do
    stats_running=$(pgrep -f "extract_zim_so.py.*stats" | wc -l)
    math_running=$(pgrep  -f "extract_zim_so.py.*math"  | wc -l)

    if [ "$stats_running" -eq 0 ] && [ "$math_running" -eq 0 ]; then
        log "Both extractions done."
        break
    fi

    log "Still running — stats_procs=$stats_running math_procs=$math_running — sleeping 60s..."
    sleep 60
done

# Quick sanity check
stats_count=$(wc -l < "$DATASETS_DIR/stats_accepted.jsonl" 2>/dev/null || echo 0)
math_count=$(wc -l  < "$DATASETS_DIR/math_accepted.jsonl"  2>/dev/null || echo 0)
quant_count=$(wc -l < "$DATASETS_DIR/quant_so_accepted.jsonl" 2>/dev/null || echo 0)
log "Sample counts — quant: $quant_count  stats: $stats_count  math: $math_count"

# -----------------------------------------------------------------------
# 2. Prep dataset
# -----------------------------------------------------------------------
log "Running prep_quant_dataset.py..."
$PYTHON "$DATASETS_DIR/../bin/prep_quant_dataset.py" 2>&1 | tee -a "$LOG"

if [ $? -ne 0 ]; then
    log "ERROR: prep_quant_dataset.py failed — aborting."
    exit 1
fi

# -----------------------------------------------------------------------
# 3. Run training
# -----------------------------------------------------------------------
log "Starting QLoRA training..."
cd "$TRAINING_DIR" || exit 1

$PYTHON pipeline/train.py \
    --config run_stock_analyst.yaml \
    --theme  themes/stock_analyst \
    2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log "=== Training complete! Adapter saved to ~/models/nemotron-4b-quant-lora ==="
else
    log "=== Training exited with code $EXIT_CODE — check log for details ==="
fi
