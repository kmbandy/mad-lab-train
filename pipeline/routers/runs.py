import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from pipeline.db import get_db
from pipeline.models import (
    Checkpoint,
    Event,
    JobStatus,
    Run,
    Stage,
    StageConfig,
    StageType,
)
from pipeline.schemas import (
    CheckpointResponse,
    PriorityUpdate,
    ResumeRequest,
    RunCreate,
    RunDetail,
    RunResponse,
    RunUpdate,
    ScheduleUpdate,
    StageConfigResponse,
    StageResponse,
)

router = APIRouter(prefix="/runs", tags=["runs"])

# Retention windows by stage type (days after job ends)
_RETENTION = {
    StageType.pretrain: 14,
    StageType.finetune: 14,
    StageType.prune: 14,
    StageType.data_gen: 7,
    StageType.dataset_prep: 7,
    StageType.quant: 7,
    StageType.merge: 7,
    StageType.convert: 7,
    StageType.eval: 3,
    StageType.upload: 3,
}


def _retention_days(stage_types: list[StageType]) -> int:
    return max((_RETENTION.get(st, 3) for st in stage_types), default=3)


def _compute_retain_until(stage_types: list[StageType]) -> datetime:
    days = _retention_days(stage_types)
    return datetime.now(timezone.utc) + timedelta(days=days)


async def _get_run_or_404(run_id: uuid.UUID, db: AsyncSession) -> Run:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("", status_code=201)
async def create_run(body: RunCreate, db: AsyncSession = Depends(get_db)) -> dict:
    if body.set_as_next and body.scheduled_for:
        raise HTTPException(status_code=400, detail="set_as_next and scheduled_for are mutually exclusive")

    stage_types = [s.stage_type for s in body.stages]

    priority = 100
    if body.set_as_next or body.start_immediately:
        # Demote any existing priority-0 run to priority 1
        result = await db.execute(
            select(Run).where(
                Run.priority == 0,
                Run.status.in_([JobStatus.pending, JobStatus.queued]),
            )
        )
        for existing in result.scalars().all():
            existing.priority = 1
        priority = 0

    run = Run(
        name=body.name,
        template_name=body.template_name,
        execution_target=body.execution_target,
        ec2_config=body.ec2_config,
        priority=priority,
        scheduled_for=body.scheduled_for,
        retain_logs_until=_compute_retain_until(stage_types),
        status=JobStatus.pending if body.scheduled_for else JobStatus.queued,
        queued_at=None if body.scheduled_for else datetime.now(timezone.utc),
    )
    db.add(run)
    await db.flush()

    for i, stage_def in enumerate(body.stages):
        stage = Stage(run_id=run.id, sequence=i, stage_type=stage_def.stage_type)
        db.add(stage)
        await db.flush()
        cfg = StageConfig(run_id=run.id, stage_id=stage.id, stage_type=stage_def.stage_type, config=stage_def.config)
        db.add(cfg)

    await db.commit()
    await db.refresh(run)
    from pipeline.scheduler import wake_scheduler
    wake_scheduler()
    return {"run": RunResponse.model_validate(run)}


@router.get("")
async def list_runs(
    status: JobStatus | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    q = select(Run).order_by(Run.priority.asc(), Run.created_at.desc()).offset(offset).limit(limit)
    if status:
        q = q.where(Run.status == status)
    result = await db.execute(q)
    runs = result.scalars().all()
    return {"runs": [RunResponse.model_validate(r) for r in runs], "total": len(runs)}


@router.get("/{run_id}")
async def get_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(Run)
        .where(Run.id == run_id)
        .options(selectinload(Run.stages), selectinload(Run.stage_configs))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetail(
        run=RunResponse.model_validate(run),
        stages=[StageResponse.model_validate(s) for s in run.stages],
        configs=[StageConfigResponse.model_validate(c) for c in run.stage_configs],
    ).model_dump()


@router.patch("/{run_id}")
async def update_run(run_id: uuid.UUID, body: RunUpdate, db: AsyncSession = Depends(get_db)) -> dict:
    run = await _get_run_or_404(run_id, db)
    if run.status not in (JobStatus.pending, JobStatus.queued):
        raise HTTPException(status_code=409, detail="Can only edit runs in pending or queued state")
    if body.name is not None:
        run.name = body.name
    if body.ec2_config is not None:
        run.ec2_config = body.ec2_config
    await db.commit()
    await db.refresh(run)
    return {"run": RunResponse.model_validate(run)}


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    run = await _get_run_or_404(run_id, db)
    terminal = {JobStatus.pending, JobStatus.queued, JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
    if run.status not in terminal:
        raise HTTPException(status_code=409, detail="Cannot delete a running or paused run")
    await db.delete(run)
    await db.commit()


# ── Control actions ───────────────────────────────────────────────────────────

@router.post("/{run_id}/start")
async def start_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.orchestrator import get_orchestrator
    from pipeline.scheduler import wake_scheduler

    run = await _get_run_or_404(run_id, db)
    if run.status not in (JobStatus.pending, JobStatus.queued):
        raise HTTPException(status_code=409, detail="Run is not in a startable state")
    if get_orchestrator(run_id):
        raise HTTPException(status_code=409, detail="Run already has an active orchestrator")
    run.status = JobStatus.queued
    run.scheduled_for = None
    run.queued_at = run.queued_at or datetime.now(timezone.utc)
    run.priority = 0
    _add_event(run.id, "run_queued", {"source": "manual"}, db)
    await db.commit()
    wake_scheduler()
    return {"status": run.status}


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.orchestrator import get_orchestrator
    from pipeline.scheduler import wake_scheduler

    run = await _get_run_or_404(run_id, db)
    if run.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(status_code=409, detail="Run is already in a terminal state")
    run.status = JobStatus.cancelled
    run.ended_at = datetime.now(timezone.utc)
    _add_event(run.id, "run_cancelled", {}, db)
    await db.commit()
    orch = get_orchestrator(run_id)
    if orch:
        await orch.cancel()
    wake_scheduler()
    return {"status": run.status}


@router.post("/{run_id}/pause")
async def pause_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.orchestrator import get_orchestrator

    run = await _get_run_or_404(run_id, db)
    if run.status != JobStatus.running:
        raise HTTPException(status_code=409, detail="Run is not running")
    _add_event(run.id, "pause_requested", {}, db)
    await db.commit()

    orch = get_orchestrator(run_id)
    if orch:
        await orch.pause()
    return {"status": "pause_requested"}


@router.post("/{run_id}/force-pause")
async def force_pause_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.orchestrator import get_orchestrator

    run = await _get_run_or_404(run_id, db)
    if run.status != JobStatus.running:
        raise HTTPException(status_code=409, detail="Run is not running")
    _add_event(run.id, "force_pause_requested", {}, db)
    await db.commit()

    orch = get_orchestrator(run_id)
    if orch:
        await orch.force_pause()
    return {"status": "force_pause_requested"}


@router.post("/{run_id}/resume")
async def resume_run(run_id: uuid.UUID, body: ResumeRequest, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.orchestrator import get_orchestrator
    from pipeline.scheduler import wake_scheduler

    run = await _get_run_or_404(run_id, db)
    if run.status != JobStatus.paused:
        raise HTTPException(status_code=409, detail="Run is not paused")
    if get_orchestrator(run_id):
        raise HTTPException(status_code=409, detail="Run already has an active orchestrator")

    checkpoint = None
    if body.checkpoint_id:
        checkpoint = await db.get(Checkpoint, body.checkpoint_id)
        if checkpoint is None or checkpoint.run_id != run_id:
            raise HTTPException(status_code=404, detail="Checkpoint not found for this run")
    else:
        result = await db.execute(
            select(Checkpoint)
            .where(Checkpoint.run_id == run_id, Checkpoint.is_clean.is_(True))
            .order_by(Checkpoint.created_at.desc())
            .limit(1)
        )
        checkpoint = result.scalar_one_or_none()

    if checkpoint:
        checkpoint_stage = await db.get(Stage, checkpoint.stage_id)
        result = await db.execute(
            select(Stage).where(
                Stage.run_id == run_id,
                Stage.sequence >= checkpoint_stage.sequence,
            )
        )
        for stage in result.scalars().all():
            stage.status = StageStatus.paused if stage.id == checkpoint.stage_id else StageStatus.pending
            stage.started_at = None
            stage.ended_at = None
            stage.error = None
            stage.output_path = None

    run.status = JobStatus.queued
    run.priority = 0
    run.queued_at = datetime.now(timezone.utc)
    checkpoint_id = str(checkpoint.id) if checkpoint else None
    _add_event(run.id, "run_resumed", {"checkpoint_id": checkpoint_id}, db)
    await db.commit()

    wake_scheduler()
    return {"status": run.status}


@router.post("/{run_id}/priority")
async def set_priority(run_id: uuid.UUID, body: PriorityUpdate, db: AsyncSession = Depends(get_db)) -> dict:
    run = await _get_run_or_404(run_id, db)
    # Check for millisecond collision at priority 0
    if body.priority == 0:
        result = await db.execute(
            select(Run).where(Run.priority == 0, Run.id != run_id, Run.created_at == run.created_at)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Priority collision: identical priority and created_at")
        # Demote existing priority-0 run
        result = await db.execute(select(Run).where(Run.priority == 0, Run.id != run_id))
        for existing in result.scalars().all():
            existing.priority = 1
    run.priority = body.priority
    await db.commit()
    return {"priority": run.priority}


@router.post("/{run_id}/schedule")
async def schedule_run(run_id: uuid.UUID, body: ScheduleUpdate, db: AsyncSession = Depends(get_db)) -> dict:
    from pipeline.scheduler import wake_scheduler

    run = await _get_run_or_404(run_id, db)
    if run.status not in (JobStatus.pending, JobStatus.queued):
        raise HTTPException(status_code=409, detail="Can only schedule pending or queued runs")
    run.scheduled_for = body.scheduled_for
    run.status = JobStatus.pending if body.scheduled_for else JobStatus.queued
    await db.commit()
    wake_scheduler()
    return {"scheduled_for": run.scheduled_for}


# ── Streams, logs, checkpoints ────────────────────────────────────────────────

@router.get("/{run_id}/checkpoints")
async def list_checkpoints(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    await _get_run_or_404(run_id, db)
    result = await db.execute(
        select(Checkpoint).where(Checkpoint.run_id == run_id).order_by(Checkpoint.created_at.asc())
    )
    checkpoints = result.scalars().all()
    return {"checkpoints": [CheckpointResponse.model_validate(c) for c in checkpoints]}


@router.get("/{run_id}/logs")
async def get_logs(
    run_id: uuid.UUID,
    lines: int = Query(100, ge=1, le=5000),
    stage_id: uuid.UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    import os
    from pathlib import Path

    from pipeline.settings import settings

    await _get_run_or_404(run_id, db)
    log_dir = Path(os.path.expanduser(settings.log_dir)) / str(run_id)
    if not log_dir.exists():
        return {"lines": [], "run_id": str(run_id)}

    log_file = log_dir / f"{stage_id}.log" if stage_id else log_dir / "run.log"
    if not log_file.exists():
        # Return last lines from any available log file
        logs = sorted(log_dir.glob("*.log"))
        if not logs:
            return {"lines": [], "run_id": str(run_id)}
        log_file = logs[-1]

    with open(log_file) as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip() for l in all_lines[-lines:]], "run_id": str(run_id)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_event(run_id: uuid.UUID, event_type: str, data: dict, db: AsyncSession) -> None:
    event = Event(run_id=run_id, event_type=event_type, data=data, ts=datetime.now(timezone.utc))
    db.add(event)
