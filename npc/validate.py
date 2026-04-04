#!/usr/bin/env python3
"""
Staged validation pipeline for synthetic NPC training data.

Designed around the constraint that Writer and Opus Distill are both 27B models
on the same 6900 XT — only one can be loaded at a time. OmniCoder (GTX 1070)
runs the fast filter independently.

Recommended run order (★ = can run in parallel with the line above it):

  Stage 1 — Generate writer samples:
    [Writer loaded on 6900 XT]
    python3 generate.py --model writer

  Stage 2 — Generate opus samples + fast filter writer simultaneously:
    [Swap to Opus Distill on 6900 XT]   terminal A: python3 generate.py --model opus
    [OmniCoder on GTX 1070]           ★ terminal B: python3 validate.py --pass fast-filter --source writer

  Stage 3 — Drummer cross-reviews filtered writer + fast filter raw opus simultaneously:
    [Drummer fine-tune on 6900 XT]      terminal A: python3 validate.py --pass cross-review --source writer --reviewer drummer
    [OmniCoder on GTX 1070]           ★ terminal B: python3 validate.py --pass fast-filter --source opus

  Stage 4 — Writer cross-reviews filtered opus samples:
    [Swap back to Writer on 6900 XT]
    python3 validate.py --pass cross-review --source opus --reviewer writer

Usage:
    python3 validate.py --pass fast-filter --source writer
    python3 validate.py --pass fast-filter --source opus
    python3 validate.py --pass cross-review --source writer --reviewer drummer
    python3 validate.py --pass cross-review --source opus   --reviewer writer
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FAST_FILTER_SYSTEM = """You are a quality scorer for D&D roleplay writing.

Score the NPC response below from 0.0 to 1.0.

Scoring guide:
  1.0 — Perfect: vivid action, natural dialogue, clear character voice, stays in the moment
  0.8 — Good: solid format, decent voice, minor issues
  0.6 — Acceptable: correct format, but flat or generic
  0.4 — Poor: format problems, out of character, or too long
  0.0 — Fail: no format, broken, or gibberish

Respond with ONLY a decimal number, nothing else. Example: 0.8"""

FAST_FILTER_PROMPT = """Rate this NPC response:

Scene: {scene}
Mood: {mood}
Player action: {player_action}

Response:
{response}

Score (0.0-1.0):"""

DRUMMER_REVIEW_SYSTEM = """You are reviewing synthetic D&D NPC training data for creative writing quality.

Scoring guide:
  1.0 — Exceptional: genuinely surprising voice, vivid physicality, every word earns its place
  0.8 — Good: solid craft, distinct character voice, fits the scene and mood well
  0.6 — Passable: correct format but generic, forgettable, or leans on clichés
  0.4 — Weak: flat voice, misses the mood, or action line adds nothing
  0.0 — Fail: wrong format, out of character, or broken

A good NPC response should feel earned — not just technically correct but genuinely in-character and immersive. Generic fantasy filler should score 0.6 or below.

Respond with exactly two lines:
Line 1: A decimal score from 0.0 to 1.0
Line 2: One sentence explaining your rating"""

OPUS_REVIEW_SYSTEM = """You are reviewing synthetic D&D NPC training data for lore consistency and character accuracy.

Evaluate whether the NPC response fits the scene, mood, and player action described.
Check: Does the character's reaction make sense? Is the voice consistent? Does it avoid generic fantasy clichés?

Respond with exactly two lines:
Line 1: A decimal score from 0.0 to 1.0
Line 2: One sentence explaining your rating"""

WRITER_REVIEW_SYSTEM = """You are reviewing synthetic D&D NPC training data for creative quality and natural dialogue.

Evaluate whether the NPC response has genuine character voice and feels like something a real person would say.
Check: Does the dialogue feel natural or stilted? Does the action line add something? Does it avoid purple prose?

Respond with exactly two lines:
Line 1: A decimal score from 0.0 to 1.0
Line 2: One sentence explaining your rating"""

QWEN_REVIEW_SYSTEM = """You are evaluating synthetic D&D NPC training data for overall quality.

Assess whether the NPC response is believable, fits the scene and mood, and has a distinct character voice.
Consider: Is the reaction appropriate? Does the dialogue feel natural? Is it specific to this character or could any NPC have said it?

Scoring guide:
  1.0 — Excellent: specific, vivid, fully in-character
  0.8 — Good: fits the scene well, clear voice, minor issues at most
  0.6 — Mediocre: technically correct but generic or interchangeable
  0.4 — Poor: misses the mood, weak voice, or format problems
  0.0 — Fail: broken, out of character, or wrong format

Respond with exactly two lines:
Line 1: A decimal score from 0.0 to 1.0
Line 2: One sentence explaining your rating"""

CROSS_REVIEW_PROMPT = """Review this NPC training sample:

Character: {character}
Scene: {scene}
Mood: {mood}
Player action: {player_action}

NPC response:
{response}"""

REVIEWER_CONFIGS = {
    "drummer": (DRUMMER_REVIEW_SYSTEM, "drummer_api_base",      "drummer_model"),
    "opus":    (OPUS_REVIEW_SYSTEM,    "opus_distill_api_base", "opus_distill_model"),
    "qwen":    (QWEN_REVIEW_SYSTEM,    "qwen_api_base",         "qwen_model"),
    "writer":  (WRITER_REVIEW_SYSTEM,  "writer_api_base",       "writer_model"),
}

# ---------------------------------------------------------------------------
# Helpers
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


def call_model(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 80,
) -> Optional[str]:
    try:
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
    except Exception as e:
        print(f"  [error] model call failed: {e}", file=sys.stderr)
        return None


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Pass 1 — Fast filter (OmniCoder on GTX 1070)
# ---------------------------------------------------------------------------

def run_fast_filter(samples: list[dict], client: OpenAI, model: str, threshold: float) -> list[dict]:
    print(f"\n--- Fast filter: {len(samples)} samples (threshold={threshold}) ---")
    passed = []
    for i, sample in enumerate(samples):
        prompt = FAST_FILTER_PROMPT.format(
            scene=sample["scene"],
            mood=sample["mood"],
            player_action=sample["player_action"],
            response=sample["response"],
        )
        raw = call_model(client, model, FAST_FILTER_SYSTEM, prompt, max_tokens=10)
        score = parse_score(raw or "")

        if score is None:
            print(f"  [{i+1:4d}/{len(samples)}] skip (unparseable: {raw!r})")
            continue

        sample["fast_filter_score"] = score
        if score >= threshold:
            passed.append(sample)

        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            rate = len(passed) / (i + 1) * 100
            print(f"  [{i+1:4d}/{len(samples)}] pass rate {rate:.0f}% — last: {score:.2f} {sample['character']}")

        time.sleep(0.2)

    print(f"  Done: {len(passed)}/{len(samples)} passed")
    return passed


# ---------------------------------------------------------------------------
# Pass 2 — Cross-review (27B model on 6900 XT)
# ---------------------------------------------------------------------------

def run_cross_review(
    samples: list[dict],
    client: OpenAI,
    model: str,
    system: str,
    reviewer_name: str,
    threshold: float,
) -> list[dict]:
    print(f"\n--- Cross-review by {reviewer_name}: {len(samples)} samples (threshold={threshold}) ---")
    passed = []
    for i, sample in enumerate(samples):
        prompt = CROSS_REVIEW_PROMPT.format(
            character=sample["character"],
            scene=sample["scene"],
            mood=sample["mood"],
            player_action=sample["player_action"],
            response=sample["response"],
        )
        raw = call_model(client, model, system, prompt, max_tokens=80)
        score, reason = parse_review(raw or "") if raw else (None, "")

        if score is None:
            print(f"  [{i+1:4d}/{len(samples)}] skip (unparseable: {raw!r})")
            continue

        sample[f"review_{reviewer_name}_score"] = score
        sample[f"review_{reviewer_name}_reason"] = reason

        if score >= threshold:
            passed.append(sample)

        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            rate = len(passed) / (i + 1) * 100
            print(f"  [{i+1:4d}/{len(samples)}] pass rate {rate:.0f}% — {score:.2f} {reason[:55]}")

        time.sleep(0.3)

    print(f"  Done: {len(passed)}/{len(samples)} passed")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged validation — run one pass at a time to match hardware availability"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pass", dest="stage", required=True,
                        choices=["fast-filter", "cross-review"],
                        help="Which validation stage to run")
    parser.add_argument("--source", required=True,
                        choices=["writer", "opus", "drummer"],
                        help="Which model's samples to validate")
    parser.add_argument("--reviewer", choices=["writer", "opus", "drummer", "qwen"],
                        help="Required for --pass cross-review: which model does the reviewing")
    parser.add_argument("--input", default=None,
                        help="Override input file for cross-review (default: filtered_{source}.jsonl)")
    args = parser.parse_args()

    if args.stage == "cross-review" and not args.reviewer:
        parser.error("--reviewer is required for --pass cross-review")
    if args.stage == "cross-review" and args.reviewer == args.source:
        parser.error("--reviewer cannot be the same model as --source")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["output_dir"])
    fast_threshold = cfg["min_quality_score"]
    review_threshold = cfg["cross_review_threshold"]

    if args.stage == "fast-filter":
        raw_path = data_dir / f"raw_{args.source}.jsonl"
        out_path = data_dir / f"filtered_{args.source}.jsonl"

        if not raw_path.exists():
            print(f"Error: {raw_path} not found — run generate.py --model {args.source} first")
            sys.exit(1)

        samples = load_jsonl(raw_path)
        print(f"Loaded {len(samples)} raw samples from {raw_path.name}")

        client = OpenAI(base_url=cfg["fast_filter_api_base"], api_key="unused")
        passed = run_fast_filter(samples, client, cfg["fast_filter_model"], fast_threshold)

        save_jsonl(passed, out_path)
        print(f"\nSaved {len(passed)} filtered samples → {out_path}")

    elif args.stage == "cross-review":
        filtered_path = Path(args.input) if args.input else data_dir / f"filtered_{args.source}.jsonl"
        out_path      = data_dir / f"validated_{args.source}.jsonl"

        if not filtered_path.exists():
            print(f"Error: {filtered_path} not found — run --pass fast-filter --source {args.source} first")
            sys.exit(1)

        system, api_key, model_key = REVIEWER_CONFIGS[args.reviewer]
        client = OpenAI(base_url=cfg[api_key], api_key="unused")
        model  = cfg[model_key]

        samples = load_jsonl(filtered_path)
        print(f"Loaded {len(samples)} filtered samples from {filtered_path.name}")

        passed = run_cross_review(samples, client, model, system, args.reviewer, review_threshold)

        save_jsonl(passed, out_path)
        print(f"\nSaved {len(passed)} validated samples → {out_path}")


if __name__ == "__main__":
    main()
