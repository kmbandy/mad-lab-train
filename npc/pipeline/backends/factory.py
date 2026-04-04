"""Backend factory — maps training_env to the right TrainingBackend class."""

from __future__ import annotations

from pathlib import Path

from .base import TrainingBackend


def get_backend(ft_cfg, run_cfg: dict, theme_dir: Path) -> TrainingBackend:
    env = ft_cfg.training_env
    if env == "local":
        from .local import LocalBackend
        return LocalBackend(ft_cfg, run_cfg, theme_dir)
    elif env == "kaggle":
        from .kaggle import KaggleBackend
        return KaggleBackend(ft_cfg, run_cfg, theme_dir)
    elif env == "aws":
        from .aws import AWSBackend
        return AWSBackend(ft_cfg, run_cfg, theme_dir)
    elif env == "gcp":
        from .gcp import GCPBackend
        return GCPBackend(ft_cfg, run_cfg, theme_dir)
    elif env == "lambda_labs":
        from .lambda_labs import LambdaLabsBackend
        return LambdaLabsBackend(ft_cfg, run_cfg, theme_dir)
    else:
        raise ValueError(
            f"Unknown training_env: '{env}'. "
            f"Valid options: local, kaggle, aws, gcp, lambda_labs"
        )
