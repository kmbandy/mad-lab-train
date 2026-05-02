#!/usr/bin/env python3
"""
Build Python-focused imatrix calibration.txt from HuggingFace datasets.
Targets ~5MB of Python code and instruction content.

Usage:
    python3 build_python_calibration.py -o imatrix/python-calibration.txt
"""

import argparse
import random
from pathlib import Path
from typing import Generator

TARGET_BYTES = 5 * 1024 * 1024  # 5MB


def iter_ajibawa_python(max_samples: int) -> Generator[str, None, None]:
    """ajibawa-2023/Python-Code-23k-ShareGPT — sharegpt format Python coding QA."""
    from datasets import load_dataset
    ds = load_dataset("ajibawa-2023/Python-Code-23k-ShareGPT", split="train")
    samples = list(ds)
    random.shuffle(samples)
    for row in samples[:max_samples]:
        convos = row["conversations"] if isinstance(row, dict) else []
        parts = [t["value"] for t in convos if isinstance(t, dict) and t.get("value", "").strip()]
        if parts:
            yield "\n\n".join(parts)


def iter_iamtarun_python(max_samples: int) -> Generator[str, None, None]:
    """iamtarun/python_code_instructions_18k_alpaca — instruction/output pairs."""
    from datasets import load_dataset
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    samples = list(ds)
    random.shuffle(samples)
    for row in samples[:max_samples]:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if instruction and output:
            yield f"{instruction}\n\n{output}"


def iter_flytech_python(max_samples: int) -> Generator[str, None, None]:
    """flytech/python-codes-25k — Python code snippets with descriptions."""
    from datasets import load_dataset
    ds = load_dataset("flytech/python-codes-25k", split="train")
    samples = list(ds)
    random.shuffle(samples)
    for row in samples[:max_samples]:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "") or row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if output:
            text = f"{instruction}\n\n{output}" if instruction else output
            yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="imatrix/python-calibration.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading Python coding datasets from HuggingFace...")

    sources = [
        ("ajibawa Python-Code-23k", iter_ajibawa_python(2500)),
        ("iamtarun python_code_instructions", iter_iamtarun_python(2000)),
        ("flytech python-codes-25k", iter_flytech_python(1500)),
    ]

    all_samples: list[str] = []
    for name, gen in sources:
        samples = list(gen)
        print(f"  {name}: {len(samples)} samples")
        all_samples.extend(samples)

    random.shuffle(all_samples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    count = 0
    with open(out_path, "w") as f:
        for text in all_samples:
            if not text.strip():
                continue
            f.write(text + "\n\n")
            written += len(text.encode())
            count += 1
            if written >= TARGET_BYTES:
                break

    print(f"Wrote {count:,} samples, {written / 1024 / 1024:.1f}MB → {out_path}")


if __name__ == "__main__":
    main()
