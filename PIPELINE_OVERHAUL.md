# Pipeline Overhaul — Implementation Plan

**Goal**: Make the pipeline fully config-driven and environment-agnostic.
A single `run_X.yaml` + `themes/X/` directory should be sufficient to run any
fine-tune on any hardware (local CUDA, local ROCm, Kaggle, AWS, GCP, Lambda Labs)
with no code changes required for new domains or new training backends.

**Working directory for all changes**: `npc/` (the generalized pipeline lives here)

**Do not break existing functionality.** All current themes (dnd_npc, stock_analyst)
must continue to work after every change. Test with `--dry-run` after each step.

---

## Step 1 — Centralize ALL trainer args in `finetune.yaml`

**File**: `npc/pipeline/train.py`

**Problem**: Several `SFTConfig` arguments are hardcoded in `train.py` and cannot
be overridden from config:
```python
weight_decay=0.01
max_grad_norm=1.0
gradient_checkpointing=False
packing=False
lr_scheduler_type="cosine"
logging_steps=10
eval_steps=100
save_steps=100
save_total_limit=2
load_best_model_at_end=True
metric_for_best_model="eval_loss"
report_to="none"
dataloader_num_workers=0
seed=42
```

Also `LoraConfig` has hardcoded:
```python
target_modules="all-linear"
bias="none"
task_type="CAUSAL_LM"
```

**Changes to `npc/pipeline/train.py`**:

Replace the hardcoded SFTConfig block with config-driven values. Add a
`_sft_args()` helper that reads from `ft_cfg` with defaults:

```python
def _sft_args(ft_cfg: dict, hw: HardwareProfile, output_dir: str, sequence_len: int) -> SFTConfig:
    """Build SFTConfig from finetune.yaml, falling back to sensible defaults."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=ft_cfg.get("num_epochs", 3),
        per_device_train_batch_size=ft_cfg.get("micro_batch_size", 1),
        per_device_eval_batch_size=ft_cfg.get("micro_batch_size", 1),
        gradient_accumulation_steps=ft_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=ft_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=ft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=ft_cfg.get("warmup_steps", 20),
        weight_decay=ft_cfg.get("weight_decay", 0.01),
        max_grad_norm=ft_cfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=ft_cfg.get("gradient_checkpointing", False),
        fp16=hw.fp16,
        bf16=hw.bf16,
        logging_steps=ft_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=ft_cfg.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=ft_cfg.get("save_steps", 100),
        save_total_limit=ft_cfg.get("save_total_limit", 2),
        load_best_model_at_end=ft_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=ft_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=ft_cfg.get("report_to", "none"),
        dataloader_num_workers=ft_cfg.get("dataloader_num_workers", 0),
        seed=ft_cfg.get("seed", 42),
        max_length=sequence_len,
        dataset_text_field="text",
        packing=ft_cfg.get("packing", False),
    )
```

Add a `_lora_config()` helper:

```python
def _lora_config(ft_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=ft_cfg.get("lora_r", 16),
        lora_alpha=ft_cfg.get("lora_alpha", 32),
        lora_dropout=ft_cfg.get("lora_dropout", 0.05),
        target_modules=ft_cfg.get("lora_target_modules", "all-linear"),
        bias=ft_cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )
```

**Changes to `npc/themes/dnd_npc/finetune.yaml`** — add optional override section
as comments showing every field that can now be set:

```yaml
# All fields below are optional (defaults shown)
# weight_decay: 0.01
# max_grad_norm: 1.0
# gradient_checkpointing: false
# packing: false
# lr_scheduler_type: cosine    # cosine | linear | constant
# logging_steps: 10
# eval_steps: 100
# save_steps: 100
# save_total_limit: 2
# seed: 42
# lora_target_modules: all-linear
# report_to: none              # none | wandb | tensorboard
# wandb_project: my-project    # only used if report_to: wandb
```

Same comments added to `npc/themes/stock_analyst/finetune.yaml`.

**Optional W&B integration** — after `report_to` is configurable, add to `train.py`:

```python
if ft_cfg.get("report_to") == "wandb":
    import os
    os.environ.setdefault("WANDB_PROJECT", ft_cfg.get("wandb_project", "mad-lab-train"))
```

---

## Step 2 — Config schema validation

**New file**: `npc/pipeline/schema.py`

Create Pydantic v2 models for all config files. These are used to validate at
load time and produce clear error messages.

```python
"""Config schema validation using Pydantic v2."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Run config ──────────────────────────────────────────────────────────────

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

    # Dynamic: {key}_api_base and {key}_model — validated against theme generators/reviewers
    model_config = {"extra": "allow"}

    def endpoint(self, key: str) -> tuple[str, str]:
        """Return (api_base, model_id) for a generator or reviewer key."""
        api_base = getattr(self, f"{key}_api_base", None) or self.__pydantic_extra__.get(f"{key}_api_base")
        model_id = getattr(self, f"{key}_model", None)     or self.__pydantic_extra__.get(f"{key}_model")
        if not api_base:
            raise ValueError(f"Missing {key}_api_base in run config")
        if not model_id:
            raise ValueError(f"Missing {key}_model in run config")
        return api_base, model_id


# ── Theme sub-models ─────────────────────────────────────────────────────────

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
    Configurable HF dataset loader — no more hardcoded PIPPA/claude_multiround.
    column_map defines how to extract system/human/assistant from arbitrary datasets.
    """
    repo: str
    split: str = "train"
    local_file: Optional[str] = None
    max_samples: int = 500
    weight: float = 1.0
    filter: dict[str, Any] = Field(default_factory=dict)

    # Column mapping — keys are role names, values are source column names or paths
    # Examples:
    #   ShareGPT format:  column_map: {conversations: conversations}
    #   Alpaca format:    column_map: {system: system_prompt, human: instruction, assistant: output}
    #   ChatML format:    column_map: {messages: messages}
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
            raise ValueError(f"Category weights sum to {total:.3f}, expected ~1.0")
        return self

    @model_validator(mode="after")
    def generator_keys_exist(self) -> "ThemeConfig":
        missing = [k for k in self.dataset.generator_keys if k not in self.generators]
        if missing:
            raise ValueError(f"dataset.generator_keys references undefined generators: {missing}")
        return self


# ── Finetune config ──────────────────────────────────────────────────────────

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
    image_uri: Optional[str] = None   # defaults to AWS DLC image
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
    api_key_env: str = "LAMBDA_API_KEY"   # env var holding the key


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
        env = self.training_env
        if env == "kaggle"      and self.kaggle is None:
            raise ValueError("training_env: kaggle requires a [kaggle] section")
        if env == "aws"         and self.aws is None:
            raise ValueError("training_env: aws requires an [aws] section")
        if env == "gcp"         and self.gcp is None:
            raise ValueError("training_env: gcp requires a [gcp] section")
        if env == "lambda_labs" and self.lambda_labs is None:
            raise ValueError("training_env: lambda_labs requires a [lambda_labs] section")
        return self


# ── Loader helpers ────────────────────────────────────────────────────────────

def load_run_config(path: Path) -> RunConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        return RunConfig(**raw)
    except Exception as e:
        raise SystemExit(f"Run config error in {path}:\n  {e}")


def load_theme_config(theme_dir: Path) -> ThemeConfig:
    import yaml
    path = theme_dir / "theme.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        return ThemeConfig(**raw)
    except Exception as e:
        raise SystemExit(f"Theme config error in {path}:\n  {e}")


def load_finetune_config(theme_dir: Path, run_cfg_raw: dict | None = None) -> FinetuneConfig:
    import yaml
    path = theme_dir / "finetune.yaml"
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = (run_cfg_raw or {}).get("finetune", {})
    try:
        return FinetuneConfig(**raw)
    except Exception as e:
        raise SystemExit(f"Finetune config error in {path}:\n  {e}")
```

**Update `npc/pipeline/generate.py`**:
- In `Theme.__init__`, after loading `theme.yaml`, add:
```python
from schema import load_theme_config
self.validated = load_theme_config(theme_dir)  # raises on invalid config
```
This means any pipeline invocation validates the theme config at startup.

**Add `--validate-config` flag to `npc/pipeline/run.py`**:
```python
parser.add_argument("--validate-config", action="store_true",
    help="Validate all config files and exit without running stages")
```
In `main()`, after loading configs, add:
```python
if args.validate_config:
    from pipeline.schema import load_run_config, load_theme_config, load_finetune_config
    load_run_config(config_path)
    load_theme_config(theme_path)
    load_finetune_config(theme_path)
    print("All configs valid.")
    sys.exit(0)
```

---

## Step 3 — Config-driven HF dataset loaders

**File**: `npc/pipeline/dataset.py`

**Problem**: `load_hf_generic_sharegpt()` is one function that hardcodes PIPPA-style
packed system prompt parsing. Adding a new dataset format requires editing the code.

**Changes**:

1. Replace `load_hf_generic_sharegpt()` with a dispatcher `load_hf_dataset()` that
reads `column_map` and `format` from the `HFDatasetLoader` config and routes to
the appropriate loader:

```python
def load_hf_dataset(cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    """
    Load and normalize one HF dataset entry based on its column_map + format.
    Falls back to auto-detection if format is 'auto'.
    """
    fmt = cfg.format

    # Load raw rows
    rows = _load_rows(cfg.repo, cfg.split, cfg.max_samples, cfg.local_file)

    # Detect format if auto
    if fmt == "auto":
        fmt = _detect_format(rows[:5] if rows else [])

    # Route to loader
    if fmt == "sharegpt":
        return _load_sharegpt(rows, cfg, theme)
    elif fmt == "chatml":
        return _load_chatml(rows, cfg, theme)
    elif fmt == "alpaca":
        return _load_alpaca(rows, cfg, theme)
    else:
        print(f"  [warn] Unknown format '{fmt}' for {cfg.repo} — skipping")
        return []
```

2. Add `_load_rows(repo, split, max_samples, local_file)` — pure I/O, no parsing:
```python
def _load_rows(repo: str, split: str, max_samples: int, local_file: str | None) -> list[dict]:
    if local_file and Path(local_file).exists():
        rows = []
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
        from datasets import load_dataset
        ds = load_dataset(repo, split=split, trust_remote_code=True)
        return list(ds)[:max_samples]
```

3. Add `_detect_format(sample_rows)`:
```python
def _detect_format(rows: list[dict]) -> str:
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
```

4. Add `_load_sharegpt(rows, cfg, theme)` — replaces old generic loader but uses
`column_map` for flexibility. If `column_map` has a `conversations` key, use
that field name; otherwise default to `conversations`:
```python
def _load_sharegpt(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    conv_field = cfg.column_map.get("conversations", "conversations")
    # ... rest of existing ShareGPT normalization logic
    # (copy the working part of load_hf_generic_sharegpt, remove PIPPA-specific hacks)
```

5. Add `_load_alpaca(rows, cfg, theme)` — for datasets with instruction/input/output:
```python
def _load_alpaca(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    system_col     = cfg.column_map.get("system", "system_prompt")
    human_col      = cfg.column_map.get("human", "instruction")
    input_col      = cfg.column_map.get("input", "input")
    assistant_col  = cfg.column_map.get("assistant", "output")
    # Build ShareGPT-style conversations from alpaca fields
```

6. Add `_load_chatml(rows, cfg, theme)` — for datasets with `messages` list of
`{role, content}` dicts:
```python
def _load_chatml(rows: list[dict], cfg: HFDatasetLoader, theme: Theme) -> list[dict]:
    msg_field = cfg.column_map.get("messages", "messages")
    # Normalize role names: user→human, assistant→gpt if needed by theme output config
```

7. **Update `load_hf_datasets()`** to use the new dispatcher. Replace:
```python
samples = load_hf_generic_sharegpt(repo, split, max_samp, local_file, theme)
```
With:
```python
from schema import HFDatasetLoader as HFLoader
loader_cfg = HFLoader(**ds_cfg)
samples = load_hf_dataset(loader_cfg, theme)
```

**Update `npc/themes/dnd_npc/theme.yaml`** — add `column_map` and `format` to each
`hf_datasets` entry (this makes the existing behavior explicit in config):

```yaml
hf_datasets:
  - repo: "PygmalionAI/PIPPA"
    split: "train"
    local_file: null
    max_samples: 800
    weight: 0.6
    format: sharegpt
    column_map:
      conversations: conversations
    filter:
      min_turns: 2
      require_system: false

  - repo: "Norquinal/claude_multiround_chat_30k"
    split: "train"
    local_file: null
    max_samples: 500
    weight: 0.4
    format: sharegpt
    column_map:
      conversations: conversations
    filter:
      min_turns: 2
      require_system: false
```

---

## Step 4 — Training backend plugin architecture

**New file**: `npc/pipeline/backends/__init__.py` (empty)

**New file**: `npc/pipeline/backends/base.py`

```python
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

    def _resolve_data(self, train_data: str, eval_data: str) -> tuple[str, str]:
        """Resolve data paths relative to run config output_dir if not absolute."""
        output_dir = Path(self.run_cfg.get("output_dir", "."))
        train = Path(train_data) if train_data else output_dir / "train.jsonl"
        eval_ = Path(eval_data)  if eval_data  else output_dir / "eval.jsonl"
        return str(train), str(eval_)
```

**New file**: `npc/pipeline/backends/local.py`

Extract all local training logic from `npc/pipeline/train.py` `main()` into a
`LocalBackend` class. `train.py` becomes a thin dispatcher that instantiates
the right backend:

```python
"""Local GPU training backend (CUDA / ROCm / CPU)."""

from __future__ import annotations
from pathlib import Path
from .base import TrainingBackend


class LocalBackend(TrainingBackend):

    def run(self, train_data: str, eval_data: str) -> None:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import json, sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from hardware import detect as detect_hardware
        from generate import Theme
        # ... (move all existing local training logic from train.py here)
        # Use self.ft_cfg instead of ft_cfg, self.run_cfg instead of run_cfg
```

**New file**: `npc/pipeline/backends/kaggle.py`

Move `npc/pipeline/kaggle_train.py` logic into a `KaggleBackend` class:

```python
"""Kaggle training backend."""

from __future__ import annotations
from .base import TrainingBackend


class KaggleBackend(TrainingBackend):

    def run(self, train_data: str, eval_data: str) -> None:
        # Move all logic from kaggle_train.run() here
        # Use self.ft_cfg.kaggle for Kaggle-specific settings
        pass
```

**New file**: `npc/pipeline/backends/aws.py`

```python
"""AWS SageMaker training backend."""

from __future__ import annotations
import os
import json
import subprocess
from pathlib import Path
from .base import TrainingBackend


class AWSBackend(TrainingBackend):
    """
    Uploads training data to S3, launches a SageMaker training job,
    then polls until completion and downloads the adapter.

    Requires:
      - AWS CLI configured (aws configure or IAM role)
      - boto3 installed (pip install boto3)
      - An S3 bucket specified in finetune.yaml [aws] section
    """

    def run(self, train_data: str, eval_data: str) -> None:
        import boto3
        aws_cfg = self.ft_cfg.aws

        print(f"[aws] Uploading training data to s3://{aws_cfg.s3_bucket}/...")
        self._upload_data(train_data, eval_data, aws_cfg.s3_bucket)

        print(f"[aws] Launching SageMaker training job...")
        job_name = self._launch_job(aws_cfg)

        print(f"[aws] Waiting for job '{job_name}'...")
        self._wait_for_job(job_name)

        print(f"[aws] Downloading adapter from S3...")
        self._download_adapter(job_name, aws_cfg.s3_bucket)

    def _upload_data(self, train_data: str, eval_data: str, bucket: str) -> None:
        import boto3
        s3 = boto3.client("s3")
        for local, key in [(train_data, "data/train.jsonl"), (eval_data, "data/eval.jsonl")]:
            if Path(local).exists():
                s3.upload_file(local, bucket, key)
                print(f"  Uploaded {local} → s3://{bucket}/{key}")

    def _launch_job(self, aws_cfg) -> str:
        import boto3, time
        sm = boto3.client("sagemaker")
        ft = self.ft_cfg
        job_name = f"mad-lab-train-{ft.base_model.split('/')[-1]}-{int(time.time())}"

        # Build hyperparameters dict from ft_cfg
        hyperparameters = {
            "base_model":           ft.base_model,
            "num_epochs":           str(ft.num_epochs),
            "micro_batch_size":     str(ft.micro_batch_size),
            "gradient_accumulation_steps": str(ft.gradient_accumulation_steps),
            "learning_rate":        str(ft.learning_rate),
            "lora_r":               str(ft.lora_r),
            "lora_alpha":           str(ft.lora_alpha),
            "sequence_len":         str(ft.sequence_len),
        }

        # Use AWS Deep Learning Container image if not specified
        image_uri = aws_cfg.image_uri or self._default_dlc_image()

        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={"TrainingImage": image_uri, "TrainingInputMode": "File"},
            RoleArn=aws_cfg.role_arn,
            InputDataConfig=[
                {"ChannelName": "train", "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{aws_cfg.s3_bucket}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }}},
            ],
            OutputDataConfig={"S3OutputPath": f"s3://{aws_cfg.s3_bucket}/output/"},
            ResourceConfig={
                "InstanceType": aws_cfg.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 86400},
            HyperParameters=hyperparameters,
            EnableManagedSpotTraining=aws_cfg.spot,
        )
        return job_name

    def _wait_for_job(self, job_name: str) -> None:
        import boto3, time
        sm = boto3.client("sagemaker")
        while True:
            response = sm.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]
            print(f"  [{job_name}] status: {status}")
            if status in ("Completed", "Failed", "Stopped"):
                if status != "Completed":
                    raise RuntimeError(f"SageMaker job failed: {status}")
                return
            time.sleep(30)

    def _download_adapter(self, job_name: str, bucket: str) -> None:
        import boto3
        s3 = boto3.client("sagemaker")
        response = s3.describe_training_job(TrainingJobName=job_name)
        model_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Download and extract tar.gz
        import subprocess
        subprocess.run(["aws", "s3", "cp", model_uri, str(output_dir / "model.tar.gz")], check=True)
        subprocess.run(["tar", "-xzf", str(output_dir / "model.tar.gz"), "-C", str(output_dir)], check=True)
        print(f"  Adapter saved to {output_dir}")

    def _default_dlc_image(self) -> str:
        # AWS Deep Learning Container for PyTorch training
        # Update the image tag as needed for latest PyTorch version
        return "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
```

**New file**: `npc/pipeline/backends/gcp.py`

```python
"""Google Cloud Vertex AI training backend."""

from __future__ import annotations
from .base import TrainingBackend


class GCPBackend(TrainingBackend):
    """
    Uploads training data to GCS, launches a Vertex AI CustomJob,
    polls until completion, downloads the adapter.

    Requires:
      - gcloud CLI authenticated (gcloud auth application-default login)
      - google-cloud-aiplatform installed (pip install google-cloud-aiplatform)
    """

    def run(self, train_data: str, eval_data: str) -> None:
        from google.cloud import aiplatform, storage
        gcp_cfg = self.ft_cfg.gcp

        aiplatform.init(project=gcp_cfg.project, location=gcp_cfg.region)

        print(f"[gcp] Uploading training data to gs://{gcp_cfg.gcs_bucket}/...")
        self._upload_data(train_data, eval_data, gcp_cfg.gcs_bucket)

        print(f"[gcp] Launching Vertex AI CustomJob...")
        job = self._launch_job(gcp_cfg)

        print(f"[gcp] Waiting for job '{job.display_name}'...")
        job.wait()  # blocks until complete

        print(f"[gcp] Downloading adapter from GCS...")
        self._download_adapter(gcp_cfg.gcs_bucket)

    def _upload_data(self, train_data: str, eval_data: str, bucket: str) -> None:
        from google.cloud import storage
        client = storage.Client()
        bkt = client.bucket(bucket)
        for local, blob_name in [(train_data, "data/train.jsonl"), (eval_data, "data/eval.jsonl")]:
            if Path(local).exists():
                bkt.blob(blob_name).upload_from_filename(local)
                print(f"  Uploaded {local} → gs://{bucket}/{blob_name}")

    def _launch_job(self, gcp_cfg):
        from google.cloud import aiplatform
        ft = self.ft_cfg
        worker_spec = {
            "machine_spec": {
                "machine_type": gcp_cfg.machine_type,
                "accelerator_type": gcp_cfg.accelerator_type,
                "accelerator_count": gcp_cfg.accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                # Use a standard PyTorch training image; update URI as needed
                "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
                "command": ["python3", "train_entrypoint.py"],
                "args": [
                    "--base_model",    ft.base_model,
                    "--num_epochs",    str(ft.num_epochs),
                    "--lora_r",        str(ft.lora_r),
                    "--sequence_len",  str(ft.sequence_len),
                    "--output_dir",    f"gs://{gcp_cfg.gcs_bucket}/output/adapter",
                    "--train_data",    f"gs://{gcp_cfg.gcs_bucket}/data/train.jsonl",
                    "--eval_data",     f"gs://{gcp_cfg.gcs_bucket}/data/eval.jsonl",
                ],
            },
        }
        job = aiplatform.CustomJob(
            display_name=f"mad-lab-train-{ft.base_model.split('/')[-1]}",
            worker_pool_specs=[worker_spec],
            staging_bucket=f"gs://{gcp_cfg.gcs_bucket}",
        )
        job.run(sync=False)
        return job

    def _download_adapter(self, bucket: str) -> None:
        import subprocess
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "gsutil", "-m", "cp", "-r",
            f"gs://{bucket}/output/adapter/*", str(output_dir),
        ], check=True)
        print(f"  Adapter saved to {output_dir}")
```

**New file**: `npc/pipeline/backends/lambda_labs.py`

```python
"""Lambda Labs cloud GPU training backend."""

from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from .base import TrainingBackend


class LambdaLabsBackend(TrainingBackend):
    """
    Provisions a Lambda Labs GPU instance, rsync's training data and code,
    runs training over SSH, then rsync's the adapter back.

    Requires:
      - Lambda Labs API key in env var specified by lambda_labs.api_key_env
      - lambdalabs Python SDK: pip install lambdalabs
      - SSH key registered with Lambda Labs account
    """

    def run(self, train_data: str, eval_data: str) -> None:
        import lambdalabs
        ll_cfg = self.ft_cfg.lambda_labs
        api_key = os.environ.get(ll_cfg.api_key_env, "")
        if not api_key:
            raise EnvironmentError(f"Lambda Labs API key not found in env var '{ll_cfg.api_key_env}'")

        client = lambdalabs.Client(api_key=api_key)

        print(f"[lambda] Launching {ll_cfg.instance_type} instance...")
        instance = self._launch_instance(client, ll_cfg)
        ip = instance["ip"]

        try:
            print(f"[lambda] Waiting for SSH on {ip}...")
            self._wait_for_ssh(ip, ll_cfg.ssh_key_name)

            print(f"[lambda] Uploading training data...")
            self._upload_data(ip, train_data, eval_data, ll_cfg.ssh_key_name)

            print(f"[lambda] Running training...")
            self._run_training(ip, ll_cfg.ssh_key_name)

            print(f"[lambda] Downloading adapter...")
            self._download_adapter(ip, ll_cfg.ssh_key_name)

        finally:
            print(f"[lambda] Terminating instance {instance['id']}...")
            client.terminate_instances([instance["id"]])

    def _launch_instance(self, client, ll_cfg) -> dict:
        response = client.launch_instances(
            region_name="us-west-2",          # first available
            instance_type_name=ll_cfg.instance_type,
            ssh_key_names=[ll_cfg.ssh_key_name],
            quantity=1,
        )
        return response["data"]["instance_ids"][0]  # adjust based on SDK response

    def _wait_for_ssh(self, ip: str, ssh_key: str, timeout: int = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-i", f"~/.ssh/{ssh_key}",
                 f"ubuntu@{ip}", "echo ok"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return
            time.sleep(10)
        raise TimeoutError(f"SSH to {ip} timed out after {timeout}s")

    def _upload_data(self, ip: str, train_data: str, eval_data: str, ssh_key: str) -> None:
        ft = self.ft_cfg
        remote = f"ubuntu@{ip}:~/training/"
        subprocess.run(["ssh", "-i", f"~/.ssh/{ssh_key}", f"ubuntu@{ip}",
                        "mkdir -p ~/training/data ~/training/output"], check=True)
        for local in [train_data, eval_data]:
            if Path(local).exists():
                subprocess.run(["rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
                                local, f"{remote}data/"], check=True)
        # Upload the training script
        train_script = Path(__file__).parent.parent / "train.py"
        subprocess.run(["rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
                        str(train_script), f"{remote}"], check=True)

    def _run_training(self, ip: str, ssh_key: str) -> None:
        ft = self.ft_cfg
        cmd = (
            f"cd ~/training && pip install trl peft bitsandbytes transformers -q && "
            f"python3 train.py "
            f"--base_model {ft.base_model} "
            f"--num_epochs {ft.num_epochs} "
            f"--lora_r {ft.lora_r} "
            f"--sequence_len {ft.sequence_len} "
            f"--output_dir ~/training/output/adapter "
            f"--train_data ~/training/data/train.jsonl "
            f"--eval_data ~/training/data/eval.jsonl"
        )
        subprocess.run(
            ["ssh", "-i", f"~/.ssh/{ssh_key}", f"ubuntu@{ip}", cmd],
            check=True,
        )

    def _download_adapter(self, ip: str, ssh_key: str) -> None:
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
            f"ubuntu@{ip}:~/training/output/adapter/", str(output_dir) + "/",
        ], check=True)
        print(f"  Adapter saved to {output_dir}")
```

**New file**: `npc/pipeline/backends/factory.py`

```python
"""Backend factory — maps training_env string to backend class."""

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
        raise ValueError(f"Unknown training_env: '{env}'. Valid: local, kaggle, aws, gcp, lambda_labs")
```

**Rewrite `npc/pipeline/train.py`** to be a thin dispatcher:

```python
#!/usr/bin/env python3
"""
Fine-tuning dispatcher. Reads finetune.yaml and routes to the appropriate backend.

Usage:
    python3 pipeline/train.py --config run.yaml --theme themes/dnd_npc
    python3 pipeline/train.py --config run.yaml --theme themes/stock_analyst --dry-run
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from schema import load_run_config, load_finetune_config
from backends.factory import get_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--theme",   required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve paths
    base = Path(__file__).parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else base / args.config
    theme_dir   = Path(args.theme)  if Path(args.theme).is_absolute()  else base / args.theme

    run_cfg  = load_run_config(config_path)
    ft_cfg   = load_finetune_config(theme_dir, run_cfg.model_dump())

    # Resolve data paths
    output_dir = Path(run_cfg.output_dir)
    train_data = ft_cfg.train_data or str(output_dir / "train.jsonl")
    eval_data  = ft_cfg.eval_data  or str(output_dir / "eval.jsonl")

    print(f"[train] backend:    {ft_cfg.training_env}")
    print(f"[train] base_model: {ft_cfg.base_model}")
    print(f"[train] train_data: {train_data}")
    print(f"[train] eval_data:  {eval_data}")
    print(f"[train] output_dir: {ft_cfg.output_dir}")

    if args.dry_run:
        print("[train] dry-run — skipping training")
        return

    backend = get_backend(ft_cfg, run_cfg.model_dump(), theme_dir)
    backend.run(train_data, eval_data)


if __name__ == "__main__":
    main()
```

---

## Step 5 — Retry + backoff on LLM API calls

**File**: `npc/pipeline/generate.py`

**Problem**: `generate_one()` has a bare `except Exception as e: return None` with no
retry. Network blips, rate limits, and overloaded llama-server instances silently
drop samples.

**Changes**:

1. Add a `_with_retry()` async wrapper around the OpenAI call:

```python
import asyncio

async def _with_retry(coro_fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry an async coroutine with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [error] API call failed after {max_retries} attempts: {e}",
                      file=sys.stderr)
                return None
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {e} — retrying in {delay:.1f}s",
                  file=sys.stderr)
            await asyncio.sleep(delay)
```

2. Update `generate_one()` to use it:

```python
async def generate_one(...) -> Optional[str]:
    async def _call():
        async with semaphore:
            resp = await client.chat.completions.create(...)
            content = resp.choices[0].message.content
            ...
            return content
    return await _with_retry(_call)
```

**File**: `npc/pipeline/validate.py`

Apply the same pattern to `call_model()` — add a synchronous retry wrapper:

```python
import time

def _with_retry_sync(fn, max_retries: int = 3, base_delay: float = 1.0):
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
```

---

## Step 6 — Replace legacy orchestrators

**Files to delete**:
- `npc/run.py`
- `npc/run_r2.py`

These hardcode Kurtis's specific 2-GPU setup (GTX 1070 + RX 480), specific
model swap prompts, and specific config file paths. They predate the generalized
pipeline and are fully superseded by `npc/pipeline/run.py`.

**Before deleting**: verify that all their functionality is covered:

| Legacy feature | Replacement |
|----------------|-------------|
| `run.py` GPU model swap prompts | Use separate run configs per hardware profile |
| `run.py` stage-gating | `pipeline/run.py --stages generate validate dataset train` |
| `run_r2.py` round-2 data mixing | `pipeline/run.py --extra-dirs data/r1 data/r2` |
| `run.py` manual `systemctl` prompts | External script or Makefile target |

Create a `Makefile` (or `run-local.sh`) in `npc/` as a convenience replacement:

```makefile
# npc/Makefile — convenience targets for common pipeline runs

DND_CFG  := run_dnd_npc.yaml
DND_THEME := themes/dnd_npc

SA_CFG   := run_stock_analyst.yaml
SA_THEME  := themes/stock_analyst

PYTHON   := python3

# D&D NPC — full pipeline
dnd:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME)

# D&D NPC — regen only
dnd-gen:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --stages generate validate

# D&D NPC — train only (use existing dataset)
dnd-train:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --stages train

# Stock analyst — full pipeline
stock:
	$(PYTHON) pipeline/run.py --config $(SA_CFG) --theme $(SA_THEME)

# Validate configs only
validate-configs:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --validate-config
	$(PYTHON) pipeline/run.py --config $(SA_CFG) --theme $(SA_THEME) --validate-config
```

---

## Step 7 — Path resolution + env var overrides

**File**: `npc/pipeline/run.py`

**Problem**: All paths (output_dir, chromadb_path, model paths, kiwix_base) are
hardcoded absolute paths that differ between machines. Can't share run configs
across hosts without editing them.

**Changes**:

1. Add a `_expand_env(value: str) -> str` helper that expands `$VAR` and `${VAR}`
in any string config value:

```python
import os

def _expand_env(obj):
    """Recursively expand env vars in all string values of a dict/list."""
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    elif isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
```

2. Apply it after loading both run config and finetune config:

```python
with open(config_path) as f:
    run_cfg = _expand_env(yaml.safe_load(f))
```

3. Document in `run_dnd_npc.yaml` and `run_stock_analyst.yaml` that env vars work:

```yaml
# All paths support $HOME, $USER, and any exported env var.
# Example override: MAD_LAB_MODELS=/mnt/fast-ssd/models python3 pipeline/run.py ...

output_dir: "$HOME/mad-lab-dnd/training/data/pipeline_out"
chromadb_path: "$HOME/.mad-lab-mcp/chromadb"
fast_filter_api_base: "${FAST_FILTER_API_BASE:-http://127.0.0.1:8080/v1}"
```

---

## Step 8 — Dataset deduplication

**File**: `npc/pipeline/dataset.py`

**Problem**: No deduplication. Near-duplicate samples can sneak in when running
multiple generation rounds or mixing HF datasets with synthetic samples.

**Changes**:

Add a `_dedup()` function that removes near-duplicates based on the assistant
response content (the thing actually being trained on):

```python
import hashlib

def _dedup(samples: list[dict], min_length: int = 30) -> list[dict]:
    """
    Remove exact duplicates by hashing assistant response content.
    Also removes samples where the assistant response is under min_length chars.
    Returns deduplicated list (preserves first occurrence).
    """
    seen: set[str] = set()
    out: list[dict] = []

    for s in samples:
        # Extract assistant response from ShareGPT format
        response = ""
        for turn in s.get("conversations", []):
            if turn.get("from") in ("gpt", "assistant"):
                response = turn.get("value", "")
                break

        if len(response) < min_length:
            continue

        key = hashlib.md5(response.strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(s)

    return out
```

Call it in `main()` after merging:

```python
n_before = len(combined)
combined = _dedup(combined)
n_removed = n_before - len(combined)
if n_removed > 0:
    print(f"  Deduplication: removed {n_removed} duplicates ({n_before} → {len(combined)})")
```

Also add train/eval contamination check after the split:

```python
def _check_contamination(train: list[dict], eval_: list[dict]) -> None:
    """Warn if any eval sample appears in train."""
    train_hashes = {
        hashlib.md5(t["value"].strip().lower().encode()).hexdigest()
        for s in train for t in s.get("conversations", [])
        if t.get("from") in ("gpt", "assistant")
    }
    overlap = sum(
        1 for s in eval_ for t in s.get("conversations", [])
        if t.get("from") in ("gpt", "assistant")
        and hashlib.md5(t["value"].strip().lower().encode()).hexdigest() in train_hashes
    )
    if overlap > 0:
        print(f"  [warn] {overlap} eval samples found in train set — check dataset sources")
```

---

## Summary of New Files

```
npc/pipeline/
  schema.py                   ← new: Pydantic config models + loaders
  backends/
    __init__.py               ← new: empty
    base.py                   ← new: abstract TrainingBackend
    factory.py                ← new: get_backend() dispatcher
    local.py                  ← new: extracted from train.py
    kaggle.py                 ← new: extracted from kaggle_train.py
    aws.py                    ← new: SageMaker backend
    gcp.py                    ← new: Vertex AI backend
    lambda_labs.py            ← new: Lambda Labs SSH backend
npc/
  Makefile                    ← new: convenience targets

Files modified:
  pipeline/train.py           ← thin dispatcher, calls backends/factory.py
  pipeline/dataset.py         ← config-driven HF loaders, dedup
  pipeline/generate.py        ← retry/backoff on API calls
  pipeline/validate.py        ← retry/backoff on API calls
  pipeline/run.py             ← --validate-config flag, env var expansion
  themes/dnd_npc/theme.yaml   ← add column_map + format to hf_datasets
  themes/dnd_npc/finetune.yaml← add all SFTConfig/LoRA fields as comments
  themes/stock_analyst/theme.yaml  ← same
  themes/stock_analyst/finetune.yaml ← same

Files deleted:
  npc/run.py                  ← superseded by pipeline/run.py
  npc/run_r2.py               ← superseded by pipeline/run.py --extra-dirs
  npc/pipeline/kaggle_train.py← extracted into backends/kaggle.py
```

---

## Implementation Order

Do these in order. Each step is independently testable with `--dry-run`.

1. **Step 1** — Centralize trainer args (low risk, no behavior change)
2. **Step 2** — Config schema (add `schema.py`, wire into generate.py; existing configs must pass)
3. **Step 8** — Deduplication (add to dataset.py; verify output counts are reasonable)
4. **Step 3** — Config-driven HF loaders (update dataset.py + theme yamls; test with `--no-hf` first)
5. **Step 5** — Retry/backoff (add to generate.py + validate.py; test by temporarily breaking an endpoint)
6. **Step 7** — Path env var expansion (add to run.py; verify existing configs still work)
7. **Step 4** — Backend plugin architecture (biggest change; do last, test local backend first)
8. **Step 6** — Delete legacy files (only after Step 4 is verified working)

---

## Testing Checklist (run after all steps)

```bash
cd npc/

# Validate configs
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc --validate-config
python3 pipeline/run.py --config run_stock_analyst.yaml --theme themes/stock_analyst --validate-config

# Dry run full pipeline
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc --dry-run
python3 pipeline/run.py --config run_stock_analyst.yaml --theme themes/stock_analyst --dry-run

# Dataset stage only (no LLM calls needed)
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc --stages dataset --no-hf

# Train dry run (local backend)
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc --stages train --dry-run
```
