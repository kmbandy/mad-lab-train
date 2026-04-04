#!/usr/bin/env python3
"""
Merge and filter ZIM-extracted SE datasets for the Nemotron-4B quant fine-tune.

Sources:
  - quant.stackexchange.com  — all samples (most relevant, no score filter)
  - stats.stackexchange.com  — score >= MIN_SCORE
  - math.stackexchange.com   — score >= MIN_SCORE, capped at MATH_CAP

Adds a system prompt turn to every sample so the model learns its identity.
Outputs train.jsonl + eval.jsonl to DATASETS_DIR.
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("/home/kmbandy/mad-lab-mcp/datasets")

SOURCES = {
    "quant": {
        "path": DATASETS_DIR / "quant_so_accepted.jsonl",
        "min_score": 0,   # keep everything — only 11K samples
        "cap": None,
    },
    "stats": {
        "path": DATASETS_DIR / "stats_accepted.jsonl",
        "min_score": 3,
        "cap": None,
    },
    "math": {
        "path": DATASETS_DIR / "math_accepted.jsonl",
        "min_score": 3,
        "cap": 9000,      # math is large — cap to avoid drowning finance signal
    },
}

SYSTEM_PROMPT = (
    "You are a quantitative financial analyst with deep expertise in statistical methods, "
    "mathematical modeling, and market data interpretation. "
    "You reason carefully and provide precise, well-grounded analysis."
)

EVAL_FRACTION = 0.1
SEED = 42

TRAIN_OUT = DATASETS_DIR / "quant_train.jsonl"
EVAL_OUT  = DATASETS_DIR / "quant_eval.jsonl"

# ---------------------------------------------------------------------------

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

            # Score filter
            if s.get("q_score", 0) < cfg["min_score"]:
                continue

            samples.append(s)

    print(f"  {name}: {len(samples):,} after score filter (min_score={cfg['min_score']})")

    # Cap
    if cfg["cap"] and len(samples) > cfg["cap"]:
        random.shuffle(samples)
        samples = samples[: cfg["cap"]]
        print(f"  {name}: capped to {cfg['cap']:,}")

    return samples


def add_system_prompt(sample: dict) -> dict:
    """Prepend system turn to conversations list."""
    convs = sample.get("conversations", [])
    if convs and convs[0].get("from") == "system":
        return sample   # already has one
    return {
        **sample,
        "conversations": [{"from": "system", "value": SYSTEM_PROMPT}] + convs,
    }


def main() -> None:
    random.seed(SEED)

    all_samples = []
    for name, cfg in SOURCES.items():
        samples = load_source(name, cfg)
        # Tag source for debugging
        for s in samples:
            s["source"] = name
        all_samples.extend(samples)

    print(f"\nTotal before shuffle: {len(all_samples):,}")

    random.shuffle(all_samples)

    # Add system prompt
    all_samples = [add_system_prompt(s) for s in all_samples]

    # Train / eval split
    n_eval = max(1, int(len(all_samples) * EVAL_FRACTION))
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    print(f"Train: {len(train_samples):,}  |  Eval: {len(eval_samples):,}")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_OUT, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(EVAL_OUT, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved:")
    print(f"  {TRAIN_OUT}")
    print(f"  {EVAL_OUT}")


if __name__ == "__main__":
    main()
