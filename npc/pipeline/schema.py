"""
Config schema validation using Pydantic v2.

Validates run configs, theme configs, and finetune configs at load time,
producing clear error messages for missing or invalid fields.

Usage:
    from schema import load_run_config, load_theme_config, load_finetune_config
    run_cfg  = load_run_config(Path("run_dnd_npc.yaml"))
    theme    = load_theme_config(Path("themes/dnd_npc"))
    ft_cfg   = load_finetune_config(Path("themes/dnd_npc"))
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    output_dir: str
    chromadb_path: str = ""
    kiwix_base: str = ""
    samples_per_model: int = 300
    concurrency: int = 4
    min_quality_score: float = 0.6
    cross_review_threshold: float = 0.8
    hf_target_total: int = 1500
    fast_filter_api_base: str = "http://127.0.0.1:8080/v1"
    fast_filter_model: str = "omnicoder"

    # Allow extra fields for {key}_api_base / {key}_model per generator
    model_config = {"extra": "allow"}

    def endpoint(self, key: str) -> tuple[str, str]:
        """Return (api_base, model_id) for a generator or reviewer key."""
        extra = self.__pydantic_extra__ or {}
        api_base = extra.get(f"{key}_api_base") or getattr(self, f"{key}_api_base", None)
        model_id = extra.get(f"{key}_model")    or getattr(self, f"{key}_model", None)
        if not api_base:
            raise ValueError(f"Missing '{key}_api_base' in run config")
        if not model_id:
            raise ValueError(f"Missing '{key}_model' in run config")
        return api_base, model_id


# ---------------------------------------------------------------------------
# Theme sub-models
# ---------------------------------------------------------------------------

class GeneratorConfig(BaseModel):
    role: Literal["primary", "secondary"] = "primary"
    prompt_file: str
    temperature: float = 0.85
    max_tokens: int = 300
    no_think: bool = False


class ReviewerConfig(BaseModel):
    prompt_file: str
    pass_field: str = "score"
    pass_threshold: Optional[float] = None


class CategoryConfig(BaseModel):
    name: str
    description: str = ""
    weight: float

    @field_validator("weight")
    @classmethod
    def weight_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("category weight must be > 0")
        return v


class KiwixConfig(BaseModel):
    enabled: bool = True
    topics: list[str] = Field(default_factory=list)
    max_chars: int = 800


class LoreConfig(BaseModel):
    kiwix: KiwixConfig = Field(default_factory=KiwixConfig)
    static_file: Optional[str] = None


class HFDatasetLoader(BaseModel):
    """
    Configurable HF dataset loader entry from theme.yaml hf_datasets list.
    column_map defines how to extract system/human/assistant from a given dataset.
    format hint lets the loader skip auto-detection.
    """
    repo: str
    split: str = "train"
    local_file: Optional[str] = None
    max_samples: int = 500
    weight: float = 1.0
    filter: dict[str, Any] = Field(default_factory=dict)

    # column_map keys: conversations | messages | system | human | input | assistant
    column_map: dict[str, str] = Field(default_factory=dict)

    # Format hint: sharegpt | alpaca | chatml | auto
    format: Literal["sharegpt", "alpaca", "chatml", "auto"] = "auto"


class DatasetConfig(BaseModel):
    system_prompt: str
    human_turn: str
    response_field: str = "response"
    generator_keys: list[str] = Field(default_factory=list)


class FastFilterConfig(BaseModel):
    system_prompt: str = "fast_filter_system"
    prompt_template: str = "fast_filter_prompt"
    score_field: str = "score"
    max_tokens: int = 10
    rate_limit_sleep: float = 0.2


class CrossReviewConfig(BaseModel):
    prompt_template: str = "cross_review_prompt"
    max_tokens: int = 80
    rate_limit_sleep: float = 0.3


class ValidationConfig(BaseModel):
    must_match: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)
    max_lines: int = 20


class OutputConfig(BaseModel):
    format: Literal["sharegpt", "alpaca", "completion"] = "sharegpt"
    chat_template: Literal["chatml", "llama3", "mistral", "auto"] = "chatml"
    system_role: str = "system"
    user_role: str = "human"
    assistant_role: str = "gpt"


class ThemeConfig(BaseModel):
    name: str
    description: str = ""
    generators: dict[str, GeneratorConfig]
    reviewers: dict[str, ReviewerConfig] = Field(default_factory=dict)
    categories: list[CategoryConfig]
    lore: LoreConfig = Field(default_factory=LoreConfig)
    hf_datasets: list[HFDatasetLoader] = Field(default_factory=list)
    dataset: DatasetConfig
    fast_filter: FastFilterConfig = Field(default_factory=FastFilterConfig)
    cross_review: CrossReviewConfig = Field(default_factory=CrossReviewConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ThemeConfig":
        total = sum(c.weight for c in self.categories)
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Category weights sum to {total:.3f} — must sum to ~1.0. "
                f"Categories: {[c.name for c in self.categories]}"
            )
        return self

    @model_validator(mode="after")
    def generator_keys_exist(self) -> "ThemeConfig":
        missing = [k for k in self.dataset.generator_keys if k not in self.generators]
        if missing:
            raise ValueError(
                f"dataset.generator_keys references undefined generators: {missing}. "
                f"Defined generators: {list(self.generators.keys())}"
            )
        return self


# ---------------------------------------------------------------------------
# Finetune config
# ---------------------------------------------------------------------------

class KaggleBackendConfig(BaseModel):
    dataset_id: str
    kernel_id: str
    model_hf_id: str
    model_patches: Optional[str] = None
    sequence_len: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    num_epochs: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    extra_pip: list[str] = Field(default_factory=list)


class AWSBackendConfig(BaseModel):
    instance_type: str = "ml.g4dn.xlarge"
    role_arn: str
    s3_bucket: str
    image_uri: Optional[str] = None
    spot: bool = True


class GCPBackendConfig(BaseModel):
    project: str
    region: str = "us-central1"
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    gcs_bucket: str


class LambdaLabsBackendConfig(BaseModel):
    instance_type: str = "gpu_1x_a10"
    ssh_key_name: str
    api_key_env: str = "LAMBDA_API_KEY"


class FinetuneConfig(BaseModel):
    base_model: str
    output_dir: str
    train_data: Optional[str] = None
    eval_data: Optional[str] = None

    # Training hyperparams
    num_epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    sequence_len: int = 2048
    gradient_checkpointing: bool = False
    packing: bool = False
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"

    # Logging/saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"
    wandb_project: Optional[str] = None
    dataloader_num_workers: int = 0

    # Backend selection
    training_env: Literal["local", "kaggle", "aws", "gcp", "lambda_labs"] = "local"
    kaggle: Optional[KaggleBackendConfig] = None
    aws: Optional[AWSBackendConfig] = None
    gcp: Optional[GCPBackendConfig] = None
    lambda_labs: Optional[LambdaLabsBackendConfig] = None

    @model_validator(mode="after")
    def backend_config_present(self) -> "FinetuneConfig":
        checks = {
            "kaggle":      self.kaggle,
            "aws":         self.aws,
            "gcp":         self.gcp,
            "lambda_labs": self.lambda_labs,
        }
        env = self.training_env
        if env in checks and checks[env] is None:
            raise ValueError(
                f"training_env: '{env}' requires a [{env}] section in finetune.yaml"
            )
        return self


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _expand_env(obj: Any) -> Any:
    """Recursively expand $VAR and ~/path in all string values."""
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    elif isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_run_config(path: Path) -> RunConfig:
    with open(path) as f:
        raw = _expand_env(yaml.safe_load(f) or {})
    try:
        return RunConfig(**raw)
    except Exception as e:
        raise SystemExit(f"\nRun config error in {path}:\n  {e}\n")


def load_theme_config(theme_dir: Path) -> ThemeConfig:
    path = theme_dir / "theme.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        return ThemeConfig(**raw)
    except Exception as e:
        raise SystemExit(f"\nTheme config error in {path}:\n  {e}\n")


def load_finetune_config(theme_dir: Path, run_cfg_raw: dict | None = None) -> FinetuneConfig:
    path = theme_dir / "finetune.yaml"
    if path.exists():
        with open(path) as f:
            raw = _expand_env(yaml.safe_load(f) or {})
    else:
        raw = (run_cfg_raw or {}).get("finetune", {})
    try:
        return FinetuneConfig(**raw)
    except Exception as e:
        raise SystemExit(f"\nFinetune config error in {path}:\n  {e}\n")
