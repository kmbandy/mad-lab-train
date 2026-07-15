import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.models import ExecutionTarget, JobStatus, StageStatus, StageType


class StageConfigSchema(BaseModel):
    stage_type: StageType
    config: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    template_name: str = Field(min_length=1, max_length=200)
    execution_target: ExecutionTarget = ExecutionTarget.local
    ec2_config: dict | None = None
    stages: list[StageConfigSchema] = Field(min_length=1, max_length=50)
    scheduled_for: datetime | None = None
    start_immediately: bool = False
    set_as_next: bool = False

    @field_validator("scheduled_for")
    @classmethod
    def scheduled_time_must_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("scheduled_for must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_stage_chain(self):
        if self.start_immediately and self.scheduled_for:
            raise ValueError("start_immediately and scheduled_for are mutually exclusive")
        if self.set_as_next and self.scheduled_for:
            raise ValueError("set_as_next and scheduled_for are mutually exclusive")

        prior_types: set[StageType] = set()
        for index, stage in enumerate(self.stages):
            _validate_stage_config(index, stage, prior_types)
            prior_types.add(stage.stage_type)
        return self


_MODEL_PRODUCERS = {
    StageType.finetune,
    StageType.pretrain,
    StageType.moeify,
    StageType.quant,
    StageType.merge,
    StageType.prune,
    StageType.convert,
}


def _require(config: dict[str, Any], key: str, stage: StageType, index: int) -> None:
    if config.get(key) in (None, "", []):
        raise ValueError(f"stages[{index}] {stage.value}: {key} is required")


def _validate_stage_config(
    index: int,
    stage: StageConfigSchema,
    prior_types: set[StageType],
) -> None:
    config = stage.config
    stage_type = stage.stage_type
    has_upstream_model = bool(prior_types & _MODEL_PRODUCERS)

    if stage_type == StageType.dataset_prep:
        _require(config, "sources", stage_type, index)
        if not all(isinstance(source, dict) and source.get("type") for source in config["sources"]):
            raise ValueError(f"stages[{index}] dataset_prep: every source needs a type")
    elif stage_type == StageType.data_gen:
        _require(config, "model", stage_type, index)
        _require(config, "workers", stage_type, index)
    elif stage_type == StageType.finetune:
        if not config.get("base_model") and not has_upstream_model:
            _require(config, "base_model", stage_type, index)
        if not config.get("dataset") and StageType.dataset_prep not in prior_types:
            raise ValueError(f"stages[{index}] finetune: dataset is required without dataset_prep")
    elif stage_type == StageType.pretrain:
        _require(config, "architecture", stage_type, index)
        if StageType.dataset_prep not in prior_types and not config.get("dataset"):
            raise ValueError(f"stages[{index}] pretrain: dataset is required without dataset_prep")
    elif stage_type == StageType.moeify:
        if not config.get("base_model") and not has_upstream_model:
            _require(config, "base_model", stage_type, index)
    elif stage_type in {StageType.quant, StageType.prune, StageType.eval}:
        if not config.get("model_path") and not has_upstream_model:
            _require(config, "model_path", stage_type, index)
    elif stage_type == StageType.merge:
        mode = config.get("mode", "adapter")
        if mode == "adapter":
            if not config.get("adapter_path") and StageType.finetune not in prior_types:
                _require(config, "adapter_path", stage_type, index)
            if not config.get("base_model") and StageType.finetune not in prior_types:
                _require(config, "base_model", stage_type, index)
        else:
            _require(config, "base_model", stage_type, index)
            _require(config, "model_b", stage_type, index)
    elif stage_type == StageType.convert:
        if not config.get("input_path") and not has_upstream_model:
            _require(config, "input_path", stage_type, index)
    elif stage_type == StageType.upload:
        if not config.get("source_path") and not has_upstream_model:
            _require(config, "source_path", stage_type, index)
        _require(config, "hf_repo", stage_type, index)


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    template_name: str
    status: JobStatus
    execution_target: ExecutionTarget
    ec2_config: dict | None
    priority: int
    scheduled_for: datetime | None
    created_at: datetime
    queued_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    retain_logs_until: datetime
    error: str | None


class RunUpdate(BaseModel):
    name: str | None = None
    ec2_config: dict | None = None
    stages: list[StageConfigSchema] | None = None


class StageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    sequence: int
    stage_type: StageType
    status: StageStatus
    input_path: str | None
    output_path: str | None
    started_at: datetime | None
    ended_at: datetime | None
    error: str | None


class StageConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    stage_id: uuid.UUID
    stage_type: StageType
    config: dict


class CheckpointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    stage_id: uuid.UUID
    sequence: int
    is_clean: bool
    artifact_path: str
    meta: dict
    created_at: datetime


class RunDetail(BaseModel):
    run: RunResponse
    stages: list[StageResponse]
    configs: list[StageConfigResponse]


class TemplateCreate(BaseModel):
    name: str
    label: str
    description: str = ""
    chain: list[dict]

    @field_validator("chain", mode="before")
    @classmethod
    def canonicalize_chain(cls, value):
        return _canonical_template_chain(value)


class TemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    label: str
    description: str
    chain: list
    is_builtin: bool
    created_at: datetime

    @field_validator("chain", mode="before")
    @classmethod
    def canonicalize_chain(cls, value):
        return _canonical_template_chain(value)


def _canonical_template_chain(value):
    """Expose one template shape even for legacy built-ins seeded with defaults."""
    if not isinstance(value, list):
        return value
    chain = []
    for item in value:
        if not isinstance(item, dict):
            chain.append(item)
            continue
        normalized = dict(item)
        if "config" not in normalized and "defaults" in normalized:
            normalized["config"] = normalized.pop("defaults")
        chain.append(normalized)
    return chain


class PriorityUpdate(BaseModel):
    priority: int


class ScheduleUpdate(BaseModel):
    scheduled_for: datetime | None


class ResumeRequest(BaseModel):
    checkpoint_id: uuid.UUID | None = None


class ErrorResponse(BaseModel):
    error: str
