# mad-lab-train

All fine-tuning pipelines for the mad-lab fleet.

## Structure

```
qwen3/      — Qwen3 MoE quant stack fine-tune (stock analyst)
nemotron/   — Nemotron/SSM fine-tune pipeline (EC2 + Kaggle)
npc/        — SmolLM3 NPC character fine-tune (axolotl)
shared/     — Dataset prep scripts shared across pipelines
datasets/   — Processed training/eval JSONL files
```

## Origins

| Directory | Moved from |
|-----------|-----------|
| `qwen3/` | `mad-lab-scripts/quant-finetune/` |
| `nemotron/` | `kaggle-finetune/` |
| `npc/` | `mad-lab-dnd/training/` |
| `shared/` | `mad-lab-mcp/bin/prep_*.py`, `extract_zim_so.py`, `generate_tool_calls*.py` |
| `datasets/` | `mad-lab-mcp/datasets/*.jsonl` |
