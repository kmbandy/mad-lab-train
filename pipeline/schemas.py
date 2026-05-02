import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from pipeline.models import ExecutionTarget, JobStatus, StageStatus, StageType


class StageConfigSchema(BaseModel):
    stage_type: StageType
    config: dict


class RunCreate(BaseModel):
    name: str
    template_name: str
    execution_target: ExecutionTarget = ExecutionTarget.local
    ec2_config: dict | None = None
    stages: list[StageConfigSchema]
    scheduled_for: datetime | None = None
    start_immediately: bool = False
    set_as_next: bool = False


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
    metadata: dict
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


class TemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    label: str
    description: str
    chain: list
    is_builtin: bool
    created_at: datetime


class PriorityUpdate(BaseModel):
    priority: int


class ScheduleUpdate(BaseModel):
    scheduled_for: datetime | None


class ResumeRequest(BaseModel):
    checkpoint_id: uuid.UUID | None = None


class ErrorResponse(BaseModel):
    error: str
