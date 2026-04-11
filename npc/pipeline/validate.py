#!/usr/bin/env python3
"""
Generalized staged validator — fast-filter + cross-review.

Prompts, reviewer config, and field names all come from the theme.
Works for any domain (D&D NPC, stock analyst, etc.)

Usage:
    python3 pipeline/validate.py --config run.yaml --theme themes/dnd_npc \
        --pass fast-filter --source writer

    python3 pipeline/validate.py --config run.yaml --theme themes/dnd_npc \
        --pass cross-review --source writer --reviewer qwen
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

# -------------------------------------------------
# Retry helper for synchronous OpenAI calls
# -------------------------------------------------

def _with_retry_sync(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a synchronous function with exponential backoff.

    fn: a zero‑argument callable to execute.
    max_retries: number of attempts (including the first).
    base_delay: initial backoff in seconds; doubles each retry.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [error] validate call failed: {e}", file=sys.stderr)
                return None
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {e} — retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)

from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Template
from openai import OpenAI

# Reuse Theme from generate.py
sys.path.insert(0, str(Path(__file__).parent))
from generate import Theme


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_score(text: str) -> Optional[float]:
    m = re.search(r"\b(0\.\d+|1\.0+|0|1)\b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def parse_review(text: str) -> tuple[Optional[float], str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    score = parse_score(lines[0]) if lines else None
    reason = lines[1] if len(lines) > 1 else ""
    return score, reason


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 80,
) -> Optional[str]:
    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None

    return _with_retry_sync(_call)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Pass 1 — Fast filter
# ---------------------------------------------------------------------------

def run_fast_filter(
    samples: list[dict],
    theme: Theme,
    client: OpenAI,
    model: str,
    threshold: float,
) -> list[dict]:
    """Score samples with the fast filter model. Returns samples that pass threshold."""
    filter_cfg = theme.cfg.get("fast_filter", {})
    system_key  = filter_cfg.get("system_prompt", "fast_filter_system")
    prompt_key  = filter_cfg.get("prompt_template", "fast_filter_prompt")
    score_field = filter_cfg.get("score_field", "score")
    max_tokens  = filter_cfg.get("max_tokens", 10)
    rate_limit_sleep = filter_cfg.get("rate_limit_sleep", 0.2)

    system_prompt   = theme.prompt(system_key)
    prompt_template = Template(theme.prompt(prompt_key))

    print(f"\n--- Fast filter: {len(samples)} samples (threshold={threshold}) ---")
    passed = []

    for i, sample in enumerate(samples):
        user_prompt = prompt_template.render(**sample)
        raw = call_model(client, model, system_prompt, user_prompt, max_tokens=max_tokens)
        score = parse_score(raw or "")

        if score is None:
            print(f"  [{i+1:4d}/{len(samples)}] skip (unparseable: {raw!r})")
            time.sleep(rate_limit_sleep)
            continue

        sample[f"fast_filter_{score_field}"] = score
        if score >= threshold:
            passed.append(sample)

        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            rate = len(passed) / (i + 1) * 100
            label = sample.get("character") or sample.get("id") or ""
            print(f"  [{i+1:4d}/{len(samples)}] pass rate {rate:.0f}% — last: {score:.2f} {label}")

        time.sleep(rate_limit_sleep)

    print(f"  Done: {len(passed)}/{len(samples)} passed")
    return passed


# ---------------------------------------------------------------------------
# Pass 2 — Cross-review
# ---------------------------------------------------------------------------

def run_cross_review(
    samples: list[dict],
    theme: Theme,
    client: OpenAI,
    model: str,
    reviewer_key: str,
    threshold: float,
) -> list[dict]:
    """Have a reviewer model score filtered samples. Returns samples that pass threshold."""
    rev_cfg = theme.reviewer_cfg(reviewer_key)
    system_prompt = theme.prompt(
        rev_cfg["prompt_file"].replace("prompts/", "").replace(".txt", "")
    )

    # Cross-review prompt template — themes can override, default key is "cross_review_prompt"
    review_prompt_key = theme.cfg.get("cross_review", {}).get("prompt_template", "cross_review_prompt")
    # Fall back to a generic template if the theme doesn't define one
    try:
        prompt_template = Template(theme.prompt(review_prompt_key))
    except FileNotFoundError:
        prompt_template = Template(
            "Review this sample:\n\n"
            "{% for key, value in sample.items() %}{{ key }}: {{ value }}\n{% endfor %}"
        )

    pass_field   = rev_cfg.get("pass_field", "score")
    max_tokens   = theme.cfg.get("cross_review", {}).get("max_tokens", 80)
    rate_limit_sleep = theme.cfg.get("cross_review", {}).get("rate_limit_sleep", 0.3)

    print(f"\n--- Cross-review by {reviewer_key}: {len(samples)} samples (threshold={threshold}) ---")
    passed = []

    for i, sample in enumerate(samples):
        user_prompt = prompt_template.render(sample=sample, **sample)
        raw = call_model(client, model, system_prompt, user_prompt, max_tokens=max_tokens)
        score, reason = parse_review(raw or "") if raw else (None, "")

        if score is None:
            print(f"  [{i+1:4d}/{len(samples)}] skip (unparseable: {raw!r})")
            time.sleep(rate_limit_sleep)
            continue

        sample[f"review_{reviewer_key}_{pass_field}"] = score
        sample[f"review_{reviewer_key}_reason"] = reason

        if score >= threshold:
            passed.append(sample)

        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            rate = len(passed) / (i + 1) * 100
            print(f"  [{i+1:4d}/{len(samples)}] pass rate {rate:.0f}% — {score:.2f} {reason[:55]}")

        time.sleep(rate_limit_sleep)

    print(f"  Done: {len(passed)}/{len(samples)} passed")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged validation — run one pass at a time to match hardware availability"
    )
    parser.add_argument("--config",   required=True, help="Run config yaml")
    parser.add_argument("--theme",    required=True, help="Path to theme directory")
    parser.add_argument("--pass",     dest="stage", required=True,
                        choices=["fast-filter", "cross-review"])
    parser.add_argument("--source",   required=True,
                        help="Generator key whose samples to validate (e.g. writer)")
    parser.add_argument("--reviewer", help="Reviewer key (required for cross-review)")
    parser.add_argument("--input",    default=None,
                        help="Override input file path")
    args = parser.parse_args()

    if args.stage == "cross-review" and not args.reviewer:
        parser.error("--reviewer is required for --pass cross-review")
    if args.stage == "cross-review" and args.reviewer == args.source:
        parser.error("--reviewer cannot be the same model as --source")

    # Load theme + run config
    theme_dir = Path(args.theme)
    if not theme_dir.is_absolute():
        theme_dir = Path(__file__).parent.parent / args.theme
    theme = Theme(theme_dir)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / args.config
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)

    data_dir         = Path(run_cfg["output_dir"])
    fast_threshold   = run_cfg.get("min_quality_score", 0.6)
    review_threshold = run_cfg.get("cross_review_threshold", 0.8)

    if args.stage == "fast-filter":
        raw_path = Path(args.input) if args.input else data_dir / f"raw_{args.source}.jsonl"
        out_path = data_dir / f"filtered_{args.source}.jsonl"

        if not raw_path.exists():
            print(f"Error: {raw_path} not found")
            sys.exit(1)

        samples = load_jsonl(raw_path)
        print(f"Loaded {len(samples)} raw samples from {raw_path.name}")

        client = OpenAI(base_url=run_cfg["fast_filter_api_base"], api_key="unused")
        passed = run_fast_filter(samples, theme, client, run_cfg["fast_filter_model"], fast_threshold)

        save_jsonl(passed, out_path)
        print(f"\nSaved {len(passed)} filtered samples → {out_path}")

    elif args.stage == "cross-review":
        in_path  = Path(args.input) if args.input else data_dir / f"filtered_{args.source}.jsonl"
        out_path = data_dir / f"validated_{args.source}.jsonl"

        if not in_path.exists():
            print(f"Error: {in_path} not found — run fast-filter first")
            sys.exit(1)

        samples = load_jsonl(in_path)
        print(f"Loaded {len(samples)} filtered samples from {in_path.name}")

        rev_cfg   = theme.reviewer_cfg(args.reviewer)
        api_key   = f"{args.reviewer}_api_base"
        model_key = f"{args.reviewer}_model"
        client = OpenAI(base_url=run_cfg[api_key], api_key="unused")
        model  = run_cfg[model_key]

        passed = run_cross_review(samples, theme, client, model, args.reviewer, review_threshold)

        save_jsonl(passed, out_path)
        print(f"\nSaved {len(passed)} validated samples → {out_path}")


if __name__ == "__main__":
    main()
