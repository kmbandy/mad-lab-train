"""Database-backed run scheduler and service-start reconciliation."""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from pipeline.orchestrator import RunOrchestrator, get_orchestrator, register_orchestrator

_scheduler_task: asyncio.Task | None = None
_wake_event: asyncio.Event | None = None
logger = logging.getLogger(__name__)


def active_run_count() -> int:
    from pipeline.orchestrator import _registry

    return len(_registry)


async def launch_run(run_id: uuid.UUID) -> bool:
    """Launch a run if it has no live orchestrator and capacity is available."""
    if get_orchestrator(run_id) or active_run_count() > 0:
        return False
    orch = RunOrchestrator(run_id)
    register_orchestrator(run_id, orch)
    orch.launch()
    return True


def wake_scheduler() -> None:
    if _wake_event is not None:
        _wake_event.set()


async def reconcile_interrupted_runs() -> None:
    """Fail runs that lost their in-memory executor during a service restart."""
    from pipeline.db import AsyncSessionLocal
    from pipeline.models import JobStatus, Run, Stage, StageStatus

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Run).where(Run.status == JobStatus.running))
        runs = result.scalars().all()
        for run in runs:
            run.status = JobStatus.failed
            run.ended_at = now
            run.error = "Pipeline service restarted while this run was active"
            stages = await db.execute(
                select(Stage).where(
                    Stage.run_id == run.id,
                    Stage.status == StageStatus.running,
                )
            )
            for stage in stages.scalars().all():
                stage.status = StageStatus.failed
                stage.ended_at = now
                stage.error = "Pipeline service restarted during this stage"
        await db.commit()


async def _claim_next_run() -> uuid.UUID | None:
    from pipeline.db import AsyncSessionLocal
    from pipeline.models import Event, JobStatus, Run

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Run)
            .where(
                Run.status.in_([JobStatus.pending, JobStatus.queued]),
                (Run.scheduled_for.is_(None) | (Run.scheduled_for <= now)),
            )
            .order_by(Run.priority.asc(), Run.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        run = result.scalar_one_or_none()
        if run is None:
            return None
        run.status = JobStatus.running
        run.started_at = now
        run.queued_at = run.queued_at or now
        db.add(Event(run_id=run.id, event_type="run_started", data={"source": "scheduler"}, ts=now))
        await db.commit()
        return run.id


async def scheduler_loop() -> None:
    global _wake_event
    _wake_event = asyncio.Event()
    while True:
        try:
            if active_run_count() == 0:
                run_id = await _claim_next_run()
                if run_id is not None:
                    await launch_run(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Run scheduler tick failed")
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        _wake_event.clear()


async def start_scheduler() -> None:
    global _scheduler_task
    try:
        await reconcile_interrupted_runs()
    except Exception:
        logger.exception("Could not reconcile interrupted runs at startup")
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(scheduler_loop())


async def stop_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None
