import enum
import uuid
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pipeline.db import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class StageStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class StageType(str, enum.Enum):
    dataset_prep = "dataset_prep"
    data_gen = "data_gen"
    finetune = "finetune"
    pretrain = "pretrain"
    quant = "quant"
    merge = "merge"
    prune = "prune"
    eval = "eval"
    convert = "convert"
    upload = "upload"


class ExecutionTarget(str, enum.Enum):
    local = "local"
    ec2 = "ec2"


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("idx_runs_status", "status"),
        Index("idx_runs_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    template_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.pending, nullable=False)
    execution_target: Mapped[ExecutionTarget] = mapped_column(default=ExecutionTarget.local, nullable=False)
    ec2_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    scheduled_for: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, default=_now, nullable=False)
    queued_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    retain_logs_until: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    stages: Mapped[list["Stage"]] = relationship("Stage", back_populates="run", cascade="all, delete-orphan", order_by="Stage.sequence")
    stage_configs: Mapped[list["StageConfig"]] = relationship("StageConfig", back_populates="run", cascade="all, delete-orphan")
    checkpoints: Mapped[list["Checkpoint"]] = relationship("Checkpoint", back_populates="run", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="run", cascade="all, delete-orphan")


class Stage(Base):
    __tablename__ = "stages"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence"),
        Index("idx_stages_run_id", "run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    stage_type: Mapped[StageType] = mapped_column(nullable=False)
    status: Mapped[StageStatus] = mapped_column(default=StageStatus.pending, nullable=False)
    input_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["Run"] = relationship("Run", back_populates="stages")
    config: Mapped["StageConfig | None"] = relationship("StageConfig", back_populates="stage", uselist=False, cascade="all, delete-orphan")
    checkpoints: Mapped[list["Checkpoint"]] = relationship("Checkpoint", back_populates="stage", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="stage", cascade="all, delete-orphan")


class StageConfig(Base):
    __tablename__ = "stage_configs"
    __table_args__ = (
        UniqueConstraint("stage_id"),
        Index("idx_stage_configs_run_id", "run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    stage_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stages.id", ondelete="CASCADE"), nullable=False)
    stage_type: Mapped[StageType] = mapped_column(nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="stage_configs")
    stage: Mapped["Stage"] = relationship("Stage", back_populates="config")


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    __table_args__ = (
        UniqueConstraint("stage_id", "sequence"),
        Index("idx_checkpoints_run_id", "run_id"),
        Index("idx_checkpoints_stage_id", "stage_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    stage_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stages.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    is_clean: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, default=_now, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="checkpoints")
    stage: Mapped["Stage"] = relationship("Stage", back_populates="checkpoints")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("idx_events_run_id", "run_id"),
        Index("idx_events_run_id_ts", "run_id", "ts"),
        Index("idx_events_stage_id", "stage_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    stage_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("stages.id", ondelete="CASCADE"), nullable=True)
    stage_type: Mapped[StageType | None] = mapped_column(nullable=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    ts: Mapped[datetime] = mapped_column(TIMESTAMPTZ, default=_now, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="events")
    stage: Mapped["Stage | None"] = relationship("Stage", back_populates="events")


class Template(Base):
    __tablename__ = "templates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    chain: Mapped[list] = mapped_column(JSONB, nullable=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, default=_now, nullable=False)
