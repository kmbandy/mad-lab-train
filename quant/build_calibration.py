#!/usr/bin/env python3
"""
Build imatrix calibration.txt from corpus JSONL files.
Extracts conversation text, targets ~5MB of GPU-domain content.

Usage:
    python3 build_calibration.py <input.jsonl> [input2.jsonl ...] -o calibration.txt
"""

import argparse
import json
import random
from pathlib import Path

TARGET_BYTES = 5 * 1024 * 1024  # 5MB is plenty for imatrix


def extract_text(record: dict) -> str | None:
    convos = record.get("conversations", [])
    if not convos:
        return None
    parts = [turn["value"] for turn in convos if turn.get("value", "").strip()]
    return "\n\n".join(parts) if parts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("-o", "--output", default="calibration.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    records = []
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(records):,} records from {path}")

    random.shuffle(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    count = 0
    with open(out_path, "w") as f:
        for record in records:
            text = extract_text(record)
            if not text:
                continue
            f.write(text + "\n\n")
            written += len(text.encode())
            count += 1
            if written >= TARGET_BYTES:
                break

    print(f"Wrote {count:,} samples, {written / 1024 / 1024:.1f}MB → {out_path}")


if __name__ == "__main__":
    main()
