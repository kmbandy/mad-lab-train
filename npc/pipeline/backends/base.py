"""Abstract base class for all training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TrainingBackend(ABC):
    """Each backend takes a resolved FinetuneConfig and runs training."""

    def __init__(self, ft_cfg, run_cfg: dict, theme_dir: Path):
        self.ft_cfg    = ft_cfg
        self.run_cfg   = run_cfg
        self.theme_dir = theme_dir

    @abstractmethod
    def run(self, train_data: str, eval_data: str) -> None:
        """Execute training. Blocks until complete."""
        ...
