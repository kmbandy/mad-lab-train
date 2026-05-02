import asyncio
import json
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from pipeline.db import get_db
from pipeline.models import Event, JobStatus, Run

router = APIRouter(tags=["streams"])

_POLL_INTERVAL = 0.5  # seconds between DB polls for new events


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """SSE stream of structured events for a run. Polls events table and yields new rows."""

    async def event_generator():
        last_event_id = 0

        while True:
            result = await db.execute(
                select(Run.status).where(Run.id == run_id)
            )
            status = result.scalar_one_or_none()
            if status is None:
                yield {"data": json.dumps({"error": "run not found"})}
                return

            # Yield any new events since last poll
            result = await db.execute(
                select(Event)
                .where(Event.run_id == run_id, Event.id > last_event_id)
                .order_by(Event.id.asc())
            )
            events = result.scalars().all()
            for event in events:
                last_event_id = event.id
                payload = {
                    "run_id": str(event.run_id),
                    "stage_id": str(event.stage_id) if event.stage_id else None,
                    "stage_type": event.stage_type.value if event.stage_type else None,
                    "event_type": event.event_type,
                    "ts": event.ts.isoformat(),
                    "data": event.data,
                }
                yield {"data": json.dumps(payload)}

            # Stop streaming when run reaches terminal state
            terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
            if status in terminal:
                return

            await asyncio.sleep(_POLL_INTERVAL)

    return EventSourceResponse(event_generator())
