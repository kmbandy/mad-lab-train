import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pipeline import orchestrator, scheduler
from pipeline.models import JobStatus, StageStatus


class _Result:
    def __init__(self, values):
        self._values = values

    def scalar_one_or_none(self):
        return self._values[0] if self._values else None

    def scalars(self):
        return self

    def all(self):
        return self._values


class _Session:
    def __init__(self, results):
        self.results = iter(results)
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _query):
        return _Result(next(self.results))

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


@pytest.fixture(autouse=True)
def clear_registry():
    orchestrator._registry.clear()
    yield
    orchestrator._registry.clear()


@pytest.mark.asyncio
async def test_claim_next_run_marks_it_running(monkeypatch):
    run = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.queued,
        started_at=None,
        queued_at=None,
    )
    session = _Session([[run]])
    monkeypatch.setattr("pipeline.db.AsyncSessionLocal", lambda: session)

    claimed = await scheduler._claim_next_run()

    assert claimed == run.id
    assert run.status == JobStatus.running
    assert run.started_at is not None
    assert run.queued_at is not None
    assert session.commits == 1
    assert session.added[0].event_type == "run_started"


@pytest.mark.asyncio
async def test_reconcile_fails_interrupted_run_and_stage(monkeypatch):
    run = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.running,
        ended_at=None,
        error=None,
    )
    stage = SimpleNamespace(
        status=StageStatus.running,
        ended_at=None,
        error=None,
    )
    session = _Session([[run], [stage]])
    monkeypatch.setattr("pipeline.db.AsyncSessionLocal", lambda: session)

    await scheduler.reconcile_interrupted_runs()

    assert run.status == JobStatus.failed
    assert "restarted" in run.error
    assert stage.status == StageStatus.failed
    assert "restarted" in stage.error
    assert session.commits == 1


@pytest.mark.asyncio
async def test_finish_run_does_not_overwrite_cancelled_status():
    run = SimpleNamespace(
        status=JobStatus.cancelled,
        ended_at=datetime.now(timezone.utc),
        error=None,
    )

    class DB:
        commits = 0

        async def get(self, *_args, **_kwargs):
            return run

        async def commit(self):
            self.commits += 1

    db = DB()
    await orchestrator._finish_run(db, uuid.uuid4(), JobStatus.completed)

    assert run.status == JobStatus.cancelled
    assert db.commits == 0


@pytest.mark.asyncio
async def test_record_checkpoint_inserts_durable_row(monkeypatch):
    from pipeline.executors.base import BaseExecutor

    session = _Session([[]])
    monkeypatch.setattr("pipeline.db.AsyncSessionLocal", lambda: session)

    class Executor(BaseExecutor):
        async def run(self):
            return None

        async def pause(self):
            return None

        async def force_pause(self):
            return None

    executor = Executor(uuid.uuid4(), uuid.uuid4(), {}, session)
    await executor.record_checkpoint(12, "/tmp/checkpoint-12", {"step": 12})

    assert session.commits == 1
    assert session.added[0].sequence == 12
    assert session.added[0].artifact_path == "/tmp/checkpoint-12"


@pytest.mark.asyncio
async def test_record_checkpoint_refreshes_existing_row(monkeypatch):
    from pipeline.executors.base import BaseExecutor

    existing = SimpleNamespace(
        is_clean=False,
        artifact_path="old",
        meta={},
        created_at=datetime.now(timezone.utc),
    )
    original_created_at = existing.created_at
    session = _Session([[existing]])
    monkeypatch.setattr("pipeline.db.AsyncSessionLocal", lambda: session)

    class Executor(BaseExecutor):
        async def run(self): return None
        async def pause(self): return None
        async def force_pause(self): return None

    executor = Executor(uuid.uuid4(), uuid.uuid4(), {}, session)
    await executor.record_checkpoint(12, "new", {"step": 12})

    assert existing.is_clean is True
    assert existing.artifact_path == "new"
    assert existing.meta == {"step": 12}
    assert existing.created_at >= original_created_at
    assert not session.added
