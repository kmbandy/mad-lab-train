"""Run orchestrator — manages executor lifecycle for a single pipeline run.

Each run gets one RunOrchestrator, stored in a module-level registry so that
pause/force-pause/resume API calls can reach the live executor.

Stages execute sequentially in `sequence` order. Completed stages are skipped
on resume — executors handle their own checkpoint logic internally.
"""
import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from pipeline.executors.base import BaseExecutor

# ── Registry ──────────────────────────────────────────────────────────────────

_registry: dict[uuid.UUID, "RunOrchestrator"] = {}


def get_orchestrator(run_id: uuid.UUID) -> "RunOrchestrator | None":
    return _registry.get(run_id)


def register_orchestrator(run_id: uuid.UUID, orch: "RunOrchestrator") -> None:
    _registry[run_id] = orch


def deregister_orchestrator(run_id: uuid.UUID) -> None:
    _registry.pop(run_id, None)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class RunOrchestrator:
    def __init__(self, run_id: uuid.UUID):
        self.run_id = run_id
        self._current_executor: BaseExecutor | None = None
        self._task: asyncio.Task | None = None
        self._cancel_requested = False

    def launch(self) -> None:
        """Create an asyncio background task to run all stages."""
        self._task = asyncio.create_task(self._execute())

    async def pause(self) -> None:
        if self._current_executor:
            await self._current_executor.pause()

    async def force_pause(self) -> None:
        if self._current_executor:
            await self._current_executor.force_pause()

    async def cancel(self) -> None:
        """Request termination without cancelling the asyncio task itself.

        Some executors run blocking ML work in a thread. Cancelling the wrapper
        task would orphan that work, so cancellation is propagated through the
        executor's force-pause hook and observed at the next lifecycle boundary.
        """
        self._cancel_requested = True
        if self._current_executor:
            await self._current_executor.force_pause()

    # ── Internal execution loop ───────────────────────────────────────────────

    async def _execute(self) -> None:
        from pipeline.db import AsyncSessionLocal
        from pipeline.models import ExecutionTarget, JobStatus, Run, Stage, StageConfig, StageStatus

        async with AsyncSessionLocal() as db:
            try:
                run = await db.get(Run, self.run_id)
                ec2_config: dict = run.ec2_config or {} if run else {}
                use_ec2 = run and run.execution_target == ExecutionTarget.ec2

                result = await db.execute(
                    select(Stage)
                    .where(Stage.run_id == self.run_id)
                    .options(selectinload(Stage.config))
                    .order_by(Stage.sequence.asc())
                )
                stages = result.scalars().all()
                resume_checkpoint = await _load_resume_checkpoint(db, self.run_id)
                previous_stage = None

                for stage in stages:
                    if self._cancel_requested or await _run_is_cancelled(db, self.run_id):
                        return
                    if stage.status == StageStatus.completed:
                        previous_stage = stage
                        continue

                    if previous_stage and previous_stage.output_path:
                        stage.input_path = previous_stage.output_path

                    # Mark stage running
                    stage.status = StageStatus.running
                    stage.started_at = datetime.now(timezone.utc)
                    await db.commit()

                    cfg_data = dict(stage.config.config) if stage.config else {}
                    _wire_upstream_artifact(stage, previous_stage, cfg_data)
                    if resume_checkpoint and resume_checkpoint.stage_id == stage.id:
                        cfg_data["_resume_artifact"] = resume_checkpoint.artifact_path
                        cfg_data["_resume_metadata"] = resume_checkpoint.meta

                    if use_ec2:
                        from pipeline.executors.ec2 import Ec2Executor
                        merged = {**ec2_config, **cfg_data, "stage_type": stage.stage_type.value}
                        executor = Ec2Executor(self.run_id, stage.id, merged, db)
                    else:
                        executor = _make_executor(
                            stage.stage_type.value, self.run_id, stage.id, cfg_data, db
                        )
                    self._current_executor = executor

                    output_path: str | None = None
                    stage_error: str | None = None
                    try:
                        output_path = await executor.run()
                    except Exception as exc:
                        stage_error = str(exc)

                    self._current_executor = None

                    if self._cancel_requested or await _run_is_cancelled(db, self.run_id):
                        if stage.status == StageStatus.running:
                            stage.status = StageStatus.failed
                            stage.error = "Cancelled"
                            stage.ended_at = datetime.now(timezone.utc)
                            await db.commit()
                        return

                    # Determine outcome
                    was_paused = (
                        getattr(executor, "_pause_requested", False)
                        or getattr(executor, "_force_pause", False)
                    )

                    if stage_error:
                        stage.status = StageStatus.failed
                        stage.error = stage_error
                        stage.ended_at = datetime.now(timezone.utc)
                        await db.commit()
                        await _finish_run(db, self.run_id, JobStatus.failed, stage_error)
                        return

                    if was_paused:
                        stage.status = StageStatus.paused
                        stage.ended_at = datetime.now(timezone.utc)
                        await db.commit()
                        await _finish_run(db, self.run_id, JobStatus.paused)
                        return

                    stage.status = StageStatus.completed
                    stage.output_path = output_path
                    stage.ended_at = datetime.now(timezone.utc)
                    await db.commit()
                    previous_stage = stage

                await _finish_run(db, self.run_id, JobStatus.completed)

            except Exception as exc:
                try:
                    async with AsyncSessionLocal() as err_db:
                        await _finish_run(err_db, self.run_id, JobStatus.failed, str(exc))
                except Exception:
                    pass
            finally:
                deregister_orchestrator(self.run_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _finish_run(
    db, run_id: uuid.UUID, status, error: str | None = None
) -> None:
    from pipeline.models import JobStatus, Run

    run = await db.get(Run, run_id)
    if run:
        if run.status == JobStatus.cancelled and status != JobStatus.cancelled:
            return
        run.status = status
        run.ended_at = datetime.now(timezone.utc)
        if error:
            run.error = error
        await db.commit()


async def _run_is_cancelled(db, run_id: uuid.UUID) -> bool:
    from pipeline.models import JobStatus, Run

    run = await db.get(Run, run_id, populate_existing=True)
    return bool(run and run.status == JobStatus.cancelled)


async def _load_resume_checkpoint(db, run_id: uuid.UUID):
    """Resolve the checkpoint selected by the most recent resume request."""
    from sqlalchemy import select

    from pipeline.models import Checkpoint, Event

    result = await db.execute(
        select(Event)
        .where(Event.run_id == run_id, Event.event_type == "run_resumed")
        .order_by(Event.id.desc())
        .limit(1)
    )
    event = result.scalar_one_or_none()
    checkpoint_id = (event.data or {}).get("checkpoint_id") if event else None
    if not checkpoint_id:
        return None
    try:
        return await db.get(Checkpoint, uuid.UUID(checkpoint_id))
    except (TypeError, ValueError):
        return None


def _make_executor(
    stage_type: str,
    run_id: uuid.UUID,
    stage_id: uuid.UUID,
    config: dict,
    db,
) -> BaseExecutor:
    from pipeline.executors.convert import ConvertExecutor
    from pipeline.executors.data_gen import DataGenExecutor
    from pipeline.executors.dataset_prep import DatasetPrepExecutor
    from pipeline.executors.eval import EvalExecutor
    from pipeline.executors.finetune import FinetuneExecutor
    from pipeline.executors.merge import MergeExecutor
    from pipeline.executors.moeify import MoEifyExecutor
    from pipeline.executors.pretrain import PretrainExecutor
    from pipeline.executors.prune import PruneExecutor
    from pipeline.executors.quant import QuantExecutor
    from pipeline.executors.upload import UploadExecutor

    mapping: dict[str, type[BaseExecutor]] = {
        "dataset_prep": DatasetPrepExecutor,
        "data_gen": DataGenExecutor,
        "finetune": FinetuneExecutor,
        "pretrain": PretrainExecutor,
        "quant": QuantExecutor,
        "moeify": MoEifyExecutor,
        "merge": MergeExecutor,
        "convert": ConvertExecutor,
        "upload": UploadExecutor,
        "prune": PruneExecutor,
        "eval": EvalExecutor,
    }

    cls = mapping.get(stage_type)
    if cls is None:
        raise ValueError(f"Unknown stage type: {stage_type}")
    return cls(run_id, stage_id, config, db)


def _wire_upstream_artifact(stage, previous_stage, config: dict) -> None:
    """Inject the preceding artifact using the executor's established key."""
    if previous_stage is None or not previous_stage.output_path:
        return

    from pipeline.models import StageType

    output_path = previous_stage.output_path
    key_by_type = {
        StageType.moeify: "base_model",
        StageType.quant: "model_path",
        StageType.prune: "model_path",
        StageType.eval: "model_path",
        StageType.convert: "input_path",
        StageType.upload: "source_path",
    }
    key = key_by_type.get(stage.stage_type)
    if key:
        config.setdefault(key, output_path)

    if stage.stage_type == StageType.finetune and previous_stage.stage_type in {
        StageType.pretrain,
        StageType.moeify,
        StageType.merge,
        StageType.prune,
    }:
        config.setdefault("base_model", output_path)

    if stage.stage_type == StageType.merge and config.get("mode", "adapter") == "adapter":
        config.setdefault("adapter_path", output_path)
        previous_config = previous_stage.config.config if previous_stage.config else {}
        if previous_stage.stage_type == StageType.finetune and previous_config.get("base_model"):
            config.setdefault("base_model", previous_config["base_model"])
