#!/usr/bin/env python3
"""
Final dataset builder — merges validated synthetic samples + HuggingFace datasets,
converts to ChatML format for axolotl fine-tuning.

Input:  data/validated_writer.jsonl + data/validated_opus.jsonl
        HuggingFace datasets (downloaded on demand)
Output: data/dataset.jsonl       (full ChatML dataset)
        data/train.jsonl         (90% split)
        data/eval.jsonl          (10% split)

ChatML format (axolotl sharegpt):
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt",   "value": "..."}
        ]
    }

Usage:
    python3 dataset.py [--config config.yaml] [--eval-split 0.1] [--no-shuffle]
    python3 dataset.py --no-hf        # skip HF mixing, synthetic only
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# System prompt — matches the NPC AGENTS.md voice and format expectations
# ---------------------------------------------------------------------------

NPC_SYSTEM_PROMPT = """You are an NPC actor in an ongoing D&D campaign. You embody characters as directed by the Dungeon Master.

You receive scene context and respond in character as the specified NPC. You do not narrate the world — that is the DM's job. You speak, react, and act as the character would.

Always structure your response using this format:
*Physical action or reaction in italics.*
"Spoken dialogue in quotes."

Rules:
- 2-4 sentences maximum — one beat, one reaction, stop
- Speak and act only as the character — no narration, no meta-commentary
- Match the character's voice, vocabulary, and emotional register
- Never use < > or > notation for actions"""

# ---------------------------------------------------------------------------
# Synthetic sample conversion
# ---------------------------------------------------------------------------

def to_chatml(sample: dict) -> dict:
    """Convert a validated synthetic sample to ChatML conversation."""
    human_turn = (
        f"[CHARACTER: {sample['character']}] "
        f"[SCENE: {sample['scene']}] "
        f"[MOOD: {sample['mood']}] "
        f"[PLAYER ACTION: {sample['player_action']}]"
    )
    return {
        "conversations": [
            {"from": "system", "value": NPC_SYSTEM_PROMPT},
            {"from": "human",  "value": human_turn},
            {"from": "gpt",    "value": sample["response"]},
        ],
        "_meta": {
            "id":           sample.get("id", ""),
            "source":       "synthetic",
            "source_model": sample.get("model", ""),
            "category":     sample.get("category", ""),
            "character":    sample.get("character", ""),
            "kiwix_ref":    sample.get("kiwix_ref", ""),
            "fast_filter_score":    sample.get("fast_filter_score"),
            "review_opus_score":    sample.get("review_opus_score"),
            "review_opus_reason":   sample.get("review_opus_reason", ""),
            "review_writer_score":  sample.get("review_writer_score"),
            "review_writer_reason": sample.get("review_writer_reason", ""),
        },
    }


# ---------------------------------------------------------------------------
# HuggingFace dataset loaders
# ---------------------------------------------------------------------------

def load_pippa(max_samples: int) -> list[dict]:
    """Load PygmalionAI/PIPPA from local chatml jsonl file."""
    import json as _json
    local_path = Path(__file__).parent / "data" / "pippa_chatml.jsonl"
    print(f"  Loading PIPPA from {local_path} (up to {max_samples} samples)...")

    samples = []
    with open(local_path) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                row = _json.loads(line)
            except Exception:
                continue

            messages = row.get("messages", [])
            if not messages:
                continue

            # Extract system, first user, first assistant turn
            system_msg = ""
            human_msg  = None
            bot_msg    = None
            for m in messages:
                role    = m.get("role", "")
                content = m.get("content", "").strip()
                if role == "user" and human_msg is None:
                    # system is sometimes prepended into the first user message
                    if content.startswith("system"):
                        parts = content.split("\n", 1)
                        system_msg = parts[0].replace("system", "").strip()
                        human_msg  = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        human_msg = content
                elif role == "assistant" and human_msg is not None:
                    bot_msg = content
                    break

            if not human_msg or not bot_msg:
                continue
            if len(bot_msg) < 30 or len(bot_msg) > 800:
                continue

            system = system_msg if system_msg else NPC_SYSTEM_PROMPT
            samples.append({
                "conversations": [
                    {"from": "system", "value": system},
                    {"from": "human",  "value": human_msg},
                    {"from": "gpt",    "value": bot_msg},
                ],
                "_meta": {
                    "id": "",
                    "source": "hf_pippa",
                    "source_model": "pippa",
                    "category": "npc_dialogue",
                    "character": "",
                    "kiwix_ref": "",
                    "fast_filter_score": None,
                    "review_opus_score": None,
                    "review_opus_reason": "",
                    "review_writer_score": None,
                    "review_writer_reason": "",
                },
            })

    print(f"    Loaded {len(samples)} PIPPA samples.")
    return samples


def load_claude_multiround(max_samples: int) -> list[dict]:
    """Load Norquinal/claude_multiround_chat_30k — Claude multi-turn ShareGPT format."""
    from datasets import load_dataset
    print(f"  Downloading Norquinal/claude_multiround_chat_30k (up to {max_samples} samples)...")
    ds = load_dataset("Norquinal/claude_multiround_chat_30k", split="train", trust_remote_code=True)

    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break

        convos = row.get("conversations", [])
        if len(convos) < 2:
            continue

        # Extract system + first human/gpt pair
        system_val = NPC_SYSTEM_PROMPT
        turns = []
        for turn in convos:
            role = turn.get("from", "")
            val  = turn.get("value", "").strip()
            if role == "system":
                # Keep original system if it's roleplay-flavored, otherwise replace
                if any(kw in val.lower() for kw in ["character", "roleplay", "persona", "npc", "fantasy", "adventure"]):
                    system_val = val
            elif role in ("human", "gpt") and val:
                turns.append({"from": role, "value": val})

        if len(turns) < 2:
            continue

        # Only keep first human→gpt exchange
        human_turn = next((t for t in turns if t["from"] == "human"), None)
        gpt_turn   = next((t for t in turns if t["from"] == "gpt"), None)
        if not human_turn or not gpt_turn:
            continue

        if len(gpt_turn["value"]) < 30 or len(gpt_turn["value"]) > 800:
            continue

        samples.append({
            "conversations": [
                {"from": "system", "value": system_val},
                {"from": "human",  "value": human_turn["value"]},
                {"from": "gpt",    "value": gpt_turn["value"]},
            ],
            "_meta": {
                "id": "",
                "source": "hf_claude_multiround",
                "source_model": "claude_multiround",
                "category": "npc_dialogue",
                "character": "",
                "kiwix_ref": "",
                "fast_filter_score": None,
                "review_opus_score": None,
                "review_opus_reason": "",
                "review_writer_score": None,
                "review_writer_reason": "",
            },
        })

    print(f"    Loaded {len(samples)} claude_multiround samples.")
    return samples


HF_LOADERS = {
    "PygmalionAI/PIPPA":                    load_pippa,
    "Norquinal/claude_multiround_chat_30k": load_claude_multiround,
}


def load_hf_datasets(hf_cfg: list[dict], hf_target_total: int, n_synthetic: int) -> list[dict]:
    """Download and mix HF datasets to fill out to target total."""
    hf_budget = max(0, hf_target_total - n_synthetic)
    if hf_budget == 0:
        print("  Synthetic samples already meet target — skipping HF datasets.")
        return []

    print(f"\n  HF budget: {hf_budget} samples (target {hf_target_total} - {n_synthetic} synthetic)")

    all_hf = []
    total_weight = sum(d.get("weight", 1.0) for d in hf_cfg)

    for ds_cfg in hf_cfg:
        repo       = ds_cfg["repo"]
        max_samp   = ds_cfg.get("max_samples", 500)
        weight     = ds_cfg.get("weight", 1.0)
        quota      = int(hf_budget * weight / total_weight)

        if repo not in HF_LOADERS:
            print(f"  [warn] No loader for {repo} — skipping")
            continue

        samples = HF_LOADERS[repo](max_samp)
        random.shuffle(samples)
        samples = samples[:quota]
        print(f"    Using {len(samples)} from {repo} (quota: {quota})")
        all_hf.extend(samples)

    return all_hf


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def print_stats(samples: list[dict], label: str) -> None:
    print(f"\n  {label}: {len(samples)} samples")
    categories = Counter(s["_meta"]["category"] for s in samples)
    sources    = Counter(s["_meta"]["source"] for s in samples)

    print("  Sources:")
    for src, count in sources.most_common():
        bar = "█" * (count * 30 // len(samples))
        print(f"    {src:<30} {count:4d}  {bar}")

    print("  Categories:")
    for cat, count in categories.most_common():
        bar = "█" * (count * 30 // len(samples))
        print(f"    {cat:<25} {count:4d}  {bar}")

    ff_scores = [s["_meta"]["fast_filter_score"] for s in samples if s["_meta"]["fast_filter_score"] is not None]
    if ff_scores:
        print(f"  Fast filter scores: min={min(ff_scores):.2f} avg={sum(ff_scores)/len(ff_scores):.2f} max={max(ff_scores):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-hf",      action="store_true", help="Skip HF dataset mixing")
    parser.add_argument("--extra-dirs", nargs="*", default=[],
                        help="Additional output dirs to load validated samples from (e.g. data/r2)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_dir     = Path(cfg["output_dir"])
    dataset_path = Path(cfg["final_dataset"])
    train_path   = dataset_path.parent / "train.jsonl"
    eval_path    = dataset_path.parent / "eval.jsonl"

    # ── Load synthetic samples ────────────────────────────────────────────────
    print("\nLoading synthetic samples...")
    all_synthetic: list[dict] = []
    search_dirs = [data_dir] + [Path(d) for d in args.extra_dirs]
    for d in search_dirs:
        for source in ("writer", "opus", "drummer"):
            path = d / f"validated_{source}.jsonl"
            if path.exists():
                raw = load_jsonl(path)
                converted = [to_chatml(s) for s in raw]
                all_synthetic.extend(converted)
                print(f"  Loaded {len(converted):4d} samples from {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")

    print(f"  Total synthetic: {len(all_synthetic)}")

    # ── Load HF datasets ──────────────────────────────────────────────────────
    all_hf: list[dict] = []
    if not args.no_hf and cfg.get("hf_datasets"):
        print("\nLoading HuggingFace datasets...")
        all_hf = load_hf_datasets(
            cfg["hf_datasets"],
            cfg.get("hf_target_total", 1500),
            len(all_synthetic),
        )

    # ── Merge ─────────────────────────────────────────────────────────────────
    combined = all_synthetic + all_hf

    if not combined:
        print("Error: no samples found. Run the pipeline first.")
        return

    # ── Shuffle & split ───────────────────────────────────────────────────────
    if not args.no_shuffle:
        random.shuffle(combined)

    n_eval  = max(1, int(len(combined) * args.eval_split))
    n_train = len(combined) - n_eval
    train   = combined[:n_train]
    eval_   = combined[n_train:]

    # ── Save ──────────────────────────────────────────────────────────────────
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(combined, dataset_path)
    save_jsonl(train,    train_path)
    save_jsonl(eval_,    eval_path)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_stats(combined, "Full dataset")
    print(f"\n  Saved:")
    print(f"    {dataset_path}  ({len(combined)} samples)")
    print(f"    {train_path}    ({n_train} train)")
    print(f"    {eval_path}     ({n_eval} eval)")
    print()


if __name__ == "__main__":
    main()
