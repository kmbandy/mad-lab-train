#!/usr/bin/env python3
"""
Fine-tuning dispatcher. Reads finetune.yaml and routes to the correct backend.

Backends: local (CUDA/ROCm/CPU), kaggle, aws, gcp, lambda_labs

Usage:
    python3 pipeline/train.py --config run_dnd_npc.yaml --theme themes/dnd_npc
    python3 pipeline/train.py --config run_dnd_npc.yaml --theme themes/dnd_npc --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import load_run_config, load_finetune_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune dispatcher")
    parser.add_argument("--config",  required=True, help="Run config yaml")
    parser.add_argument("--theme",   required=True, help="Path to theme directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit without training")
    args = parser.parse_args()

    # Resolve paths
    cwd         = Path.cwd()
    config_path = Path(args.config) if Path(args.config).is_absolute() else cwd / args.config
    theme_dir   = Path(args.theme)  if Path(args.theme).is_absolute()  else cwd / args.theme

    run_cfg = load_run_config(config_path)
    ft_cfg  = load_finetune_config(theme_dir, run_cfg.model_dump())

    # Resolve data paths
    from pathlib import Path as P
    output_dir = P(run_cfg.output_dir)
    train_data = ft_cfg.train_data or str(output_dir / "train.jsonl")
    eval_data  = ft_cfg.eval_data  or str(output_dir / "eval.jsonl")

    print(f"[train] backend    : {ft_cfg.training_env}")
    print(f"[train] base_model : {ft_cfg.base_model}")
    print(f"[train] train_data : {train_data}")
    print(f"[train] eval_data  : {eval_data}")
    print(f"[train] output_dir : {ft_cfg.output_dir}")

    if args.dry_run:
        print("[train] dry-run — skipping training")
        return

    from backends.factory import get_backend
    backend = get_backend(ft_cfg, run_cfg.model_dump(), theme_dir)
    backend.run(train_data, eval_data)


if __name__ == "__main__":
    main()
