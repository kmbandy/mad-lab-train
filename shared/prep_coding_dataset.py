#!/usr/bin/env python3
"""Prepare the eng-2 coding fine-tune dataset.

Sources (in priority order):
  - stackoverflow_accepted.jsonl  — score >= MIN_SO_SCORE, capped at SO_CAP
  - softwareengineering_accepted.jsonl — score >= MIN_SE_SCORE
  - tool_calls.jsonl              — synthetic tool-calling examples (all kept)

Adds a coding system prompt to all SO/SE samples.
Shuffles and splits into train/eval.

Outputs:
  /home/kmbandy/mad-lab-mcp/datasets/coding_train.jsonl
  /home/kmbandy/mad-lab-mcp/datasets/coding_eval.jsonl

Also copies outputs to /home/kmbandy/kaggle-finetune/eng2-dataset/ for Kaggle upload.
"""

import json
import random
import shutil
from pathlib import Path

DATASETS_DIR = Path("/home/kmbandy/mad-lab-mcp/datasets")
KAGGLE_DIR   = Path("/home/kmbandy/kaggle-finetune/eng2-dataset")

SOURCES = {
    "stackoverflow": {
        "path": DATASETS_DIR / "stackoverflow_accepted.jsonl",
        "min_score": 10,   # high bar — 2.2M lines, need aggressive filter
        "cap": 20_000,
    },
    "softwareengineering": {
        "path": DATASETS_DIR / "softwareengineering_accepted.jsonl",
        "min_score": 5,
        "cap": None,       # only ~103K, keep what passes score filter
    },
    "tool_calls": {
        "path": DATASETS_DIR / "tool_calls.jsonl",
        "min_score": None, # no score filter — generated data
        "cap": None,
    },
}

SYSTEM_PROMPT = (
    "You are mad-lab-nanobot-eng-2, an expert software engineering assistant. "
    "You help with debugging, code review, implementing features, and explaining "
    "technical concepts clearly. You write clean, idiomatic code and give precise, "
    "well-reasoned answers."
)

EVAL_FRACTION = 0.05   # 5% eval — we want most data in training
SEED = 42
TRAIN_OUT = DATASETS_DIR / "coding_train.jsonl"
EVAL_OUT  = DATASETS_DIR / "coding_eval.jsonl"


def load_source(name: str, cfg: dict) -> list[dict]:
    path = cfg["path"]
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {name}")
        return []

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue

            min_score = cfg.get("min_score")
            if min_score is not None and s.get("q_score", 0) < min_score:
                continue

            samples.append(s)
            cap = cfg.get("cap")
            if cap and len(samples) >= cap:
                break

    print(f"  {name}: {len(samples):,} samples (min_score={cfg.get('min_score', 'n/a')})")
    return samples


def add_system_prompt(sample: dict, system_prompt: str) -> dict:
    """Insert system turn at the front of conversations if not already present."""
    convos = sample.get("conversations", [])
    if convos and convos[0].get("from") == "system":
        # Already has system prompt (tool_calls data) — leave it
        return sample
    new_convos = [{"from": "system", "value": system_prompt}] + convos
    return {**sample, "conversations": new_convos}


def main():
    random.seed(SEED)

    print("Loading sources...")
    all_samples: list[dict] = []
    for name, cfg in SOURCES.items():
        samples = load_source(name, cfg)
        # Add system prompt to SO/SE data; tool_calls already have it
        if name != "tool_calls":
            samples = [add_system_prompt(s, SYSTEM_PROMPT) for s in samples]
        all_samples.extend(samples)

    print(f"\nTotal before shuffle: {len(all_samples):,}")
    random.shuffle(all_samples)

    n_eval = max(200, int(len(all_samples) * EVAL_FRACTION))
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    print(f"Train: {len(train_samples):,} | Eval: {len(eval_samples):,}")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    with open(EVAL_OUT, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote:\n  {TRAIN_OUT}\n  {EVAL_OUT}")

    # Copy to Kaggle dataset dir
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRAIN_OUT, KAGGLE_DIR / "coding_train.jsonl")
    shutil.copy(EVAL_OUT,  KAGGLE_DIR / "coding_eval.jsonl")
    print(f"Copied to {KAGGLE_DIR}")


if __name__ == "__main__":
    main()
