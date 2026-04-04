#!/usr/bin/env python3
"""
Pipeline orchestrator — runs stages end-to-end for any theme.

Reads which stages to run from the config or command line.
Each stage delegates to the appropriate pipeline module.

Usage:
    # Full pipeline
    python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc

    # Specific stages only
    python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc \\
        --stages generate validate dataset

    # Resume after partial run (skip completed stages)
    python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc \\
        --stages dataset train

    # Skip HF dataset mixing
    python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc \\
        --no-hf

Stages (in order):
    generate    — generate raw synthetic samples (all generators in theme)
    validate    — fast-filter + cross-review validation
    dataset     — merge synthetic + HF → train/eval split
    train       — QLoRA fine-tune

Config sections per stage:
    generate:   samples_<key>, concurrency, <key>_api_base, <key>_model
    validate:   fast_filter_*, cross_review_threshold, min_quality_score
    dataset:    hf_target_total, output_dir
    train:      theme finetune.yaml (or run config [finetune] block)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Stage runners — each calls the relevant pipeline/*.py module
# ---------------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).parent


def run_stage(stage: str, cmd: list[str], dry_run: bool = False) -> int:
    """Run a pipeline subprocess, stream output, return exit code."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  [dry-run] skipping execution")
        return 0

    start = time.time()
    result = subprocess.run(cmd, stdout=None, stderr=None)  # inherit stdio
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n  [stage={stage}] done in {minutes}m{seconds}s — exit {result.returncode}")
    return result.returncode


def stage_generate(args: argparse.Namespace, run_cfg: dict,
                   theme_path: Path, config_path: Path) -> int:
    """
    Run generate.py for each generator defined in the theme.
    Skips generators that already have enough output (optional --force).
    """
    import sys
    sys.path.insert(0, str(PIPELINE_DIR))
    from generate import Theme

    theme = Theme(theme_path)
    generators = theme.cfg.get("generators", {})

    if not generators:
        print("  No generators defined in theme — skipping generate stage.")
        return 0

    rc = 0
    for model_key, gen_cfg in generators.items():
        # Skip if output already exists and --no-regen
        if args.no_regen:
            output_dir = Path(run_cfg.get("output_dir", "data/pipeline_out"))
            if not output_dir.is_absolute():
                output_dir = config_path.parent / output_dir
            out_file = output_dir / f"raw_{model_key}.jsonl"
            if out_file.exists() and out_file.stat().st_size > 0:
                print(f"  Skipping {model_key} (output exists: {out_file})")
                continue

        n_samples_key = f"samples_{model_key}"
        n = run_cfg.get(n_samples_key, run_cfg.get("samples_per_model", 300))

        cmd = [
            sys.executable, str(PIPELINE_DIR / "generate.py"),
            "--config", str(config_path),
            "--theme", str(theme_path),
            "--model", model_key,
            "--n-samples", str(n),
        ]
        if args.concurrency:
            cmd += ["--concurrency", str(args.concurrency)]

        result = run_stage(f"generate:{model_key}", cmd, dry_run=args.dry_run)
        if result != 0:
            print(f"  ERROR: generate:{model_key} failed (exit {result})")
            if not args.keep_going:
                return result
            rc = result

    return rc


def stage_validate(args: argparse.Namespace, run_cfg: dict,
                   theme_path: Path, config_path: Path) -> int:
    """
    Run validate.py for each generator key:
      1. fast-filter pass
      2. cross-review pass (for each reviewer)
    """
    sys.path.insert(0, str(PIPELINE_DIR))
    from generate import Theme

    theme = Theme(theme_path)
    generators = theme.cfg.get("generators", {})
    reviewers  = theme.cfg.get("reviewers", {})

    rc = 0
    for model_key in generators:
        # Fast filter
        cmd = [
            sys.executable, str(PIPELINE_DIR / "validate.py"),
            "--config", str(config_path),
            "--theme", str(theme_path),
            "--pass", "fast-filter",
            "--source", model_key,
        ]
        result = run_stage(f"validate:fast-filter:{model_key}", cmd, dry_run=args.dry_run)
        if result != 0:
            print(f"  ERROR: fast-filter:{model_key} failed")
            if not args.keep_going:
                return result
            rc = result

        # Cross-review (each reviewer)
        for reviewer_key in reviewers:
            cmd = [
                sys.executable, str(PIPELINE_DIR / "validate.py"),
                "--config", str(config_path),
                "--theme", str(theme_path),
                "--pass", "cross-review",
                "--source", model_key,
                "--reviewer", reviewer_key,
            ]
            result = run_stage(
                f"validate:cross-review:{model_key}:{reviewer_key}",
                cmd, dry_run=args.dry_run,
            )
            if result != 0:
                print(f"  ERROR: cross-review:{model_key}:{reviewer_key} failed")
                if not args.keep_going:
                    return result
                rc = result

    return rc


def stage_dataset(args: argparse.Namespace, run_cfg: dict,
                  theme_path: Path, config_path: Path) -> int:
    """Build train/eval dataset from validated samples + optional HF mixing."""
    cmd = [
        sys.executable, str(PIPELINE_DIR / "dataset.py"),
        "--config", str(config_path),
        "--theme", str(theme_path),
    ]
    if args.no_hf:
        cmd.append("--no-hf")
    if args.eval_split is not None:
        cmd += ["--eval-split", str(args.eval_split)]
    if args.extra_dirs:
        cmd += ["--extra-dirs"] + args.extra_dirs

    return run_stage("dataset", cmd, dry_run=args.dry_run)


def stage_train(args: argparse.Namespace, run_cfg: dict,
                theme_path: Path, config_path: Path) -> int:
    """Run QLoRA fine-tune via pipeline/train.py."""
    cmd = [
        sys.executable, str(PIPELINE_DIR / "train.py"),
        "--config", str(config_path),
        "--theme", str(theme_path),
    ]
    return run_stage("train", cmd, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGES = {
    "generate": stage_generate,
    "validate": stage_validate,
    "dataset":  stage_dataset,
    "train":    stage_train,
}

ALL_STAGES = list(STAGES.keys())  # default execution order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline orchestrator")
    parser.add_argument("--config",      required=True, help="Run config yaml")
    parser.add_argument("--theme",       required=True, help="Path to theme directory")
    parser.add_argument("--stages",      nargs="*",     default=None,
                        choices=list(STAGES.keys()) + ["all"],
                        help=f"Stages to run (default: all). Options: {', '.join(ALL_STAGES)}")
    parser.add_argument("--no-hf",       action="store_true", help="Skip HF mixing in dataset stage")
    parser.add_argument("--no-regen",    action="store_true",
                        help="Skip generate if output already exists")
    parser.add_argument("--keep-going",  action="store_true",
                        help="Continue past stage failures")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--eval-split",  type=float, default=None,
                        help="Eval fraction for dataset stage (default: 0.1)")
    parser.add_argument("--extra-dirs",  nargs="*", default=[],
                        help="Extra dirs to load validated samples from (dataset stage)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override concurrency for generate stage")
    parser.add_argument("--validate-config", action="store_true",
                        help="Validate all config files and exit without running stages")
    args = parser.parse_args()

    # ---- Resolve paths ----
    base = Path(__file__).parent.parent  # mad-lab-dnd/training/

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base / args.config

    theme_path = Path(args.theme)
    if not theme_path.is_absolute():
        theme_path = base / args.theme

    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)
    if not theme_path.exists():
        print(f"Error: theme directory not found: {theme_path}")
        sys.exit(1)

    with open(config_path) as f:
        run_cfg = yaml.safe_load(f)

    # ---- Validate configs (always) ----
    sys.path.insert(0, str(PIPELINE_DIR))
    try:
        from schema import load_run_config, load_theme_config, load_finetune_config
        load_run_config(config_path)
        load_theme_config(theme_path)
        load_finetune_config(theme_path, run_cfg)
        if args.validate_config:
            print("All configs valid.")
            sys.exit(0)
    except ImportError:
        pass  # schema.py not yet available

    # ---- Determine stages ----
    if args.stages is None or args.stages == ["all"]:
        stages = ALL_STAGES
    else:
        stages = args.stages

    # ---- Banner ----
    print(f"\n{'='*60}")
    print(f"  mad-lab pipeline")
    print(f"  config : {config_path}")
    print(f"  theme  : {theme_path}")
    print(f"  stages : {' → '.join(stages)}")
    if args.dry_run:
        print(f"  MODE   : DRY RUN")
    print(f"{'='*60}")

    # ---- Run stages ----
    overall_start = time.time()
    failed = []

    for stage in stages:
        fn = STAGES[stage]
        rc = fn(args, run_cfg, theme_path, config_path)
        if rc != 0:
            failed.append(stage)
            if not args.keep_going:
                print(f"\n  Pipeline aborted at stage '{stage}' (exit {rc})")
                print(f"  Run with --keep-going to continue past failures")
                sys.exit(rc)

    # ---- Summary ----
    elapsed = time.time() - overall_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  Pipeline complete — {minutes}m{seconds}s")

    if failed:
        print(f"  Failed stages: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"  All stages passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
