#!/usr/bin/env python3
"""
Generalized dataset builder.

Merges validated synthetic samples from any theme with optional HuggingFace
dataset mixing. System prompt and human-turn format come from the theme.

Usage:
    python3 pipeline/dataset.py --config run.yaml --theme themes/dnd_npc
    python3 pipeline/dataset.py --config run.yaml --theme themes/dnd_npc --no-hf
    python3 pipeline/dataset.py --config run.yaml --theme themes/dnd_npc \
        --extra-dirs data/r1 data/r2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from jinja2 import Template

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate import Theme

# -------------------------------------------------
# HF dataset loading (config‑driven)
# -------------------------------------------------
import json
from pathlib import Path
from typing import Any
from schema import HFDatasetLoader

def _load_rows(repo: str, split: str, max_samples: int, local_file: str | None) -> list[dict]:
    """Pure I/O – load raw rows from a local JSONL or HF dataset."""
    if local_file and Path(local_file).exists():
        print(f"  Loading {repo} from local file {local_file}…")
        rows: list[dict] = []
        with open(local_file) as f:
            for line in f:
                if len(rows) >= max_samples:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    else:
        print(f"  Downloading {repo} from HuggingFace…")
        from datasets import load_dataset
        ds = load_dataset(repo, split=split, trust_remote_code=True)
        return list(ds)[:max_samples]

def _detect_format(rows: list[dict]) -> str:
    """Detect dataset format (sharegpt, chatml, alpaca)."""
    if not rows:
        return "sharegpt"
    row = rows[0]
    if "conversations" in row:
        return "sharegpt"
    if "messages" in row:
        return "chatml"
    if "instruction" in row or "output" in row:
        return "alpaca"
    return "sharegpt"

def _load_sharegpt(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    ds_cfg = theme.cfg.get("dataset", {})
    out_cfg = theme.cfg.get("output", {})
    sys_role = out_cfg.get("system_role", "system")
    user_role = out_cfg.get("user_role", "human")
    asst_role = out_cfg.get("assistant_role", "gpt")
    system_prompt = theme.prompt(
        ds_cfg.get("system_prompt", "prompts/system_prompt")
        .replace("prompts/", "").replace(".txt", "")
    )
    conv_field = cfg.column_map.get("conversations", "conversations")
    samples: list[dict] = []
    for row in rows:
        convos = row.get(conv_field, [])
        if not isinstance(convos, list) or len(convos) < 2:
            continue
        system_val = system_prompt
        human_val: str | None = None
        asst_val: str | None = None
        for turn in convos:
            role = turn.get("from") or turn.get("role", "")
            val = (turn.get("value") or turn.get("content", "")).strip()
            if role in ("system",) and not human_val:
                if any(kw in val.lower() for kw in ["character", "roleplay", "persona", "analyst", "assistant", "npc"]):
                    system_val = val
            elif role in ("human", "user") and human_val is None:
                if val.startswith("system") and "\n" in val:
                    parts = val.split("\n", 1)
                    system_val = parts[0].replace("system", "").strip() or system_val
                    human_val = parts[1].strip()
                else:
                    human_val = val
            elif role in ("gpt", "assistant") and human_val is not None:
                asst_val = val
                break
        if not human_val or not asst_val:
            continue
        if len(asst_val) < 20 or len(asst_val) > 1200:
            continue
        samples.append({
            "conversations": [
                {"from": sys_role, "value": system_val},
                {"from": user_role, "value": human_val},
                {"from": asst_role, "value": asst_val},
            ],
            "_meta": {
                "id": "",
                "source": f"hf_{cfg.repo.split('/')[-1].lower()}",
                "source_model": cfg.repo,
                "category": "",
            },
        })
    return samples

def _load_alpaca(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    ds_cfg = theme.cfg.get("dataset", {})
    out_cfg = theme.cfg.get("output", {})
    sys_role = out_cfg.get("system_role", "system")
    user_role = out_cfg.get("user_role", "human")
    asst_role = out_cfg.get("assistant_role", "gpt")
    system_prompt = theme.prompt(
        ds_cfg.get("system_prompt", "prompts/system_prompt")
        .replace("prompts/", "").replace(".txt", "")
    )
    sys_col = cfg.column_map.get("system", "system_prompt")
    human_col = cfg.column_map.get("human", "instruction")
    input_col = cfg.column_map.get("input", "input")
    asst_col = cfg.column_map.get("assistant", "output")
    samples: list[dict] = []
    for row in rows:
        system_val = row.get(sys_col, system_prompt)
        human_val = row.get(human_col) or row.get(input_col)
        asst_val = row.get(asst_col)
        if not human_val or not asst_val:
            continue
        human_val = str(human_val).strip()
        asst_val = str(asst_val).strip()
        if len(asst_val) < 20 or len(asst_val) > 1200:
            continue
        samples.append({
            "conversations": [
                {"from": sys_role, "value": system_val},
                {"from": user_role, "value": human_val},
                {"from": asst_role, "value": asst_val},
            ],
            "_meta": {
                "id": "",
                "source": f"hf_{cfg.repo.split('/')[-1].lower()}",
                "source_model": cfg.repo,
                "category": "",
            },
        })
    return samples

def _load_chatml(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    ds_cfg = theme.cfg.get("dataset", {})
    out_cfg = theme.cfg.get("output", {})
    sys_role = out_cfg.get("system_role", "system")
    user_role = out_cfg.get("user_role", "human")
    asst_role = out_cfg.get("assistant_role", "gpt")
    system_prompt = theme.prompt(
        ds_cfg.get("system_prompt", "prompts/system_prompt")
        .replace("prompts/", "").replace(".txt", "")
    )
    msgs_field = cfg.column_map.get("messages", "messages")
    samples: list[dict] = []
    for row in rows:
        msgs = row.get(msgs_field, [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        system_val = system_prompt
        human_val: str | None = None
        asst_val: str | None = None
        for turn in msgs:
            role = turn.get("role") or turn.get("from", "")
            content = turn.get("content") or turn.get("value", "")
            if not role:
                continue
            role = role.lower()
            if role == "system" and not human_val:
                if any(kw in content.lower() for kw in ["character", "roleplay", "persona", "analyst", "assistant", "npc"]):
                    system_val = content
            elif role in ("user", "human") and human_val is None:
                human_val = content
            elif role in ("assistant", "gpt") and human_val is not None:
                asst_val = content
                break
        if not human_val or not asst_val:
            continue
        if len(asst_val) < 20 or len(asst_val) > 1200:
            continue
        samples.append({
            "conversations": [
                {"from": sys_role, "value": system_val},
                {"from": user_role, "value": human_val},
                {"from": asst_role, "value": asst_val},
            ],
            "_meta": {
                "id": "",
                "source": f"hf_{cfg.repo.split('/')[-1].lower()}",
                "source_model": cfg.repo,
                "category": "",
            },
        })
    return samples

def load_hf_dataset(cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    fmt = cfg.format
    rows = _load_rows(cfg.repo, cfg.split, cfg.max_samples, cfg.local_file)
    if fmt == "auto":
        fmt = _detect_format(rows[:5] if rows else [])
    if fmt == "sharegpt":
        return _load_sharegpt(rows, cfg, theme)
    if fmt == "alpaca":
        return _load_alpaca(rows, cfg, theme)
    if fmt == "chatml":
        return _load_chatml(rows, cfg, theme)
    print(f"  [warn] Unknown format '{fmt}' for {cfg.repo} — skipping")
    return []


# ---------------------------------------------------------------------------
# Synthetic sample → conversation
# ---------------------------------------------------------------------------

def to_conversation(sample: dict, theme: Theme) -> dict:
    """
    Convert a validated synthetic sample to ShareGPT conversation format.
    System prompt and human turn template come from the theme.
    """
    ds_cfg = theme.cfg.get("dataset", {})

    system_prompt = theme.prompt(
        ds_cfg.get("system_prompt", "prompts/system_prompt")
        .replace("prompts/", "").replace(".txt", "")
    )
    human_tmpl = Template(theme.prompt(
        ds_cfg.get("human_turn", "prompts/human_turn")
        .replace("prompts/", "").replace(".txt", "")
    ))
    response_field = ds_cfg.get("response_field", "response")

    out_cfg   = theme.cfg.get("output", {})
    sys_role  = out_cfg.get("system_role", "system")
    user_role = out_cfg.get("user_role", "human")
    asst_role = out_cfg.get("assistant_role", "gpt")

    human_turn = human_tmpl.render(**sample).strip()
    response   = sample.get(response_field, "")

    return {
        "conversations": [
            {"from": sys_role,  "value": system_prompt},
            {"from": user_role, "value": human_turn},
            {"from": asst_role, "value": response},
        ],
        "_meta": {
            "id":           sample.get("id", ""),
            "source":       "synthetic",
            "source_model": sample.get("model", ""),
            "category":     sample.get("category", ""),
            **{k: sample.get(k) for k in sample
               if k.startswith(("fast_filter_", "review_"))},
        },
    }





def load_hf_datasets(hf_cfgs: list[dict], hf_target: int,
                     n_synthetic: int, theme: Theme) -> list[dict]:
    """Mix HF datasets to fill up to target total."""
    hf_budget = max(0, hf_target - n_synthetic)
    if hf_budget == 0:
        print("  Synthetic samples already meet target — skipping HF datasets.")
        return []

    print(f"\n  HF budget: {hf_budget} (target {hf_target} - {n_synthetic} synthetic)")

    all_hf: list[dict] = []
    total_weight = sum(d.get("weight", 1.0) for d in hf_cfgs)

    for ds_cfg in hf_cfgs:
        repo       = ds_cfg["repo"]
        split      = ds_cfg.get("split", "train")
        max_samp   = ds_cfg.get("max_samples", 500)
        weight     = ds_cfg.get("weight", 1.0)
        local_file = ds_cfg.get("local_file")
        quota      = int(hf_budget * weight / total_weight)

        loader_cfg = HFDatasetLoader(
            repo=repo,
            split=split,
            max_samples=max_samp,
            weight=weight,
            filter=ds_cfg.get("filter", {}),
            column_map=ds_cfg.get("column_map", {}),
            format=ds_cfg.get("format", "auto"),
            local_file=local_file,
        )
        samples = load_hf_dataset(loader_cfg, theme)
        random.shuffle(samples)
        samples = samples[:quota]
        print(f"    Using {len(samples)} from {repo} (quota: {quota})")
        all_hf.extend(samples)

    return all_hf


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(samples: list[dict]) -> None:
    print(f"\n  Total: {len(samples)} samples")
    sources    = Counter(s["_meta"].get("source", "?") for s in samples)
    categories = Counter(s["_meta"].get("category", "?") for s in samples)

    print("  Sources:")
    for src, count in sources.most_common():
        bar = "█" * (count * 30 // max(len(samples), 1))
        print(f"    {src:<35} {count:4d}  {bar}")

    print("  Categories:")
    for cat, count in categories.most_common():
        bar = "█" * (count * 30 // max(len(samples), 1))
        print(f"    {cat:<25} {count:4d}  {bar}")

    ff_scores = [
        s["_meta"].get(k) for s in samples
        for k in s["_meta"] if k.startswith("fast_filter_") and s["_meta"].get(k) is not None
    ]
    if ff_scores:
        print(f"  Fast filter scores: min={min(ff_scores):.2f} "
              f"avg={sum(ff_scores)/len(ff_scores):.2f} max={max(ff_scores):.2f}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Run config yaml")
    parser.add_argument("--theme",      required=True, help="Path to theme directory")
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-hf",      action="store_true", help="Skip HF dataset mixing")
    parser.add_argument("--extra-dirs", nargs="*", default=[],
                        help="Additional dirs to load validated samples from")
    args = parser.parse_args()

    # ---- Load theme + run config ----
    theme_dir = Path(args.theme)
    if not theme_dir.is_absolute():
        theme_dir = Path(__file__).parent.parent / args.theme
    theme = Theme(theme_dir)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / args.config
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)

    data_dir  = Path(run_cfg["output_dir"])
    out_dir   = Path(run_cfg.get("final_dataset", str(data_dir / "dataset.jsonl"))).parent

    generator_keys = theme.cfg.get("dataset", {}).get("generator_keys", ["writer", "opus", "drummer"])

    # ---- Load synthetic samples ----
    print("\nLoading synthetic samples...")
    all_synthetic: list[dict] = []
    search_dirs = [data_dir] + [Path(d) for d in args.extra_dirs]

    for d in search_dirs:
        for key in generator_keys:
            path = d / f"validated_{key}.jsonl"
            if path.exists():
                raw = load_jsonl(path)
                converted = [to_conversation(s, theme) for s in raw]
                all_synthetic.extend(converted)
                print(f"  {len(converted):4d} samples ← {path}")

    print(f"  Total synthetic: {len(all_synthetic)}")

    # ---- Load HF datasets ----
    all_hf: list[dict] = []
    hf_cfgs = theme.cfg.get("hf_datasets", [])
    if not args.no_hf and hf_cfgs:
        print("\nLoading HuggingFace datasets...")
        all_hf = load_hf_datasets(
            hf_cfgs,
            run_cfg.get("hf_target_total", 1500),
            len(all_synthetic),
            theme,
        )

    # ---- Merge + split ----
    combined = all_synthetic + all_hf
    if not combined:
        print("Error: no samples found. Run the pipeline first.")
        return

    if not args.no_shuffle:
        random.shuffle(combined)

    n_eval  = max(1, int(len(combined) * args.eval_split))
    n_train = len(combined) - n_eval

    # ---- Save ----
    dataset_path = out_dir / "dataset.jsonl"
    train_path   = out_dir / "train.jsonl"
    eval_path    = out_dir / "eval.jsonl"

    save_jsonl(combined, dataset_path)
    save_jsonl(combined[:n_train], train_path)
    save_jsonl(combined[n_train:], eval_path)

    print_stats(combined)
    print(f"\n  Saved:")
    print(f"    {dataset_path}  ({len(combined)} total)")
    print(f"    {train_path}    ({n_train} train)")
    print(f"    {eval_path}     ({n_eval} eval)")


if __name__ == "__main__":
    main()
