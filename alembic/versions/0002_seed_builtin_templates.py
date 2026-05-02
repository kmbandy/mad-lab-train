"""seed builtin templates

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-02
"""
import json
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TEMPLATES = [
    {
        "name": "gguf_quant",
        "label": "GGUF Quant",
        "description": "Dataset prep and GGUF quantization with optional imatrix calibration.",
        "chain": [
            {"stage_type": "dataset_prep", "defaults": {"train_split": 0.9, "deduplicate": True}},
            {"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M"], "imatrix": False}},
        ],
    },
    {
        "name": "full_finetune",
        "label": "Full Finetune",
        "description": "Dataset prep → data generation → QLoRA finetune → GGUF quant.",
        "chain": [
            {"stage_type": "dataset_prep", "defaults": {"train_split": 0.9, "deduplicate": True}},
            {"stage_type": "data_gen", "defaults": {"samples": 5000, "temperature": 0.85}},
            {"stage_type": "finetune", "defaults": {"mode": "standard", "lora": {"r": 16, "alpha": 32}}},
            {"stage_type": "merge", "defaults": {"mode": "adapter"}},
            {"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M", "Q5_K_M"]}},
        ],
    },
    {
        "name": "from_scratch",
        "label": "From Scratch",
        "description": "Dataset prep → data generation → pretrain from custom architecture → GGUF quant.",
        "chain": [
            {"stage_type": "dataset_prep", "defaults": {"train_split": 0.9, "deduplicate": True}},
            {"stage_type": "data_gen", "defaults": {"samples": 50000, "temperature": 0.85}},
            {"stage_type": "pretrain", "defaults": {"multi_gpu": False, "deepspeed_zero_stage": 2}},
            {"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M", "Q5_K_M"]}},
        ],
    },
    {
        "name": "prune",
        "label": "Prune",
        "description": "Dataset prep → data generation → structural prune → healing finetune → GGUF quant.",
        "chain": [
            {"stage_type": "dataset_prep", "defaults": {"train_split": 0.9, "deduplicate": True}},
            {"stage_type": "data_gen", "defaults": {"samples": 2000, "temperature": 0.85}},
            {"stage_type": "prune", "defaults": {"method": "wanda", "pruning_ratio": 0.2}},
            {"stage_type": "finetune", "defaults": {"mode": "healing", "lora": {"r": 16, "alpha": 32}}},
            {"stage_type": "merge", "defaults": {"mode": "adapter"}},
            {"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M", "Q5_K_M"]}},
        ],
    },
    {
        "name": "merge",
        "label": "Merge",
        "description": "Dataset prep → data generation → model merge → eval gate → prune → healing finetune → GGUF quant.",
        "chain": [
            {"stage_type": "dataset_prep", "defaults": {"train_split": 0.9, "deduplicate": True}},
            {"stage_type": "data_gen", "defaults": {"samples": 2000, "temperature": 0.85}},
            {"stage_type": "merge", "defaults": {"mode": "slerp", "merge_ratio": 0.5}},
            {"stage_type": "eval", "defaults": {"benchmarks": [{"type": "perplexity"}]}},
            {"stage_type": "prune", "defaults": {"method": "wanda", "pruning_ratio": 0.2}},
            {"stage_type": "finetune", "defaults": {"mode": "healing", "lora": {"r": 16, "alpha": 32}}},
            {"stage_type": "merge", "defaults": {"mode": "adapter"}},
            {"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M", "Q5_K_M"]}},
        ],
    },
    {
        "name": "eval_standalone",
        "label": "Eval Standalone",
        "description": "Run benchmarks against an existing model.",
        "chain": [
            {"stage_type": "eval", "defaults": {"benchmarks": [{"type": "perplexity"}, {"type": "mmlu"}]}},
        ],
    },
]


def upgrade() -> None:
    for t in _TEMPLATES:
        op.execute(
            f"""
            INSERT INTO templates (name, label, description, chain, is_builtin)
            VALUES (
                {_q(t['name'])},
                {_q(t['label'])},
                {_q(t['description'])},
                {_q(json.dumps(t['chain']))}::jsonb,
                TRUE
            )
            ON CONFLICT (name) DO NOTHING;
            """
        )


def downgrade() -> None:
    op.execute("DELETE FROM templates WHERE is_builtin = TRUE")


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"
