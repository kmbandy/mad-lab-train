"""Kaggle training backend — extracted from pipeline/kaggle_train.py."""

from __future__ import annotations

import sys
from pathlib import Path

from .base import TrainingBackend


class KaggleBackend(TrainingBackend):
    """Upload data to Kaggle, generate and push a training kernel, poll for completion."""

    def run(self, train_data: str, eval_data: str) -> None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from kaggle_train import run as kaggle_run
        kaggle_run(
            self.ft_cfg.__dict__ if hasattr(self.ft_cfg, "__dict__") else dict(self.ft_cfg),
            self.run_cfg,
            train_data,
            eval_data,
            self.theme_dir,
        )
