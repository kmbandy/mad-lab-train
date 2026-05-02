import uuid
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession


@runtime_checkable
class WeightPager(Protocol):
    """Interface for NVMe→VRAM weight paging. v1: not implemented.
    Future: MoE weight router pager implements this and injects into LLM-Pruner executor."""

    async def page_in(self, layer_ids: list[str]) -> None: ...
    async def page_out(self, layer_ids: list[str]) -> None: ...


class BaseExecutor(ABC):
    """Base class for all stage executors."""

    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        self.run_id = run_id
        self.stage_id = stage_id
        self.config = config
        self.db = db

    @abstractmethod
    async def run(self) -> str | None:
        """Execute the stage. Returns output_path or None."""
        ...

    @abstractmethod
    async def pause(self) -> None:
        """Request a clean pause at the next checkpoint."""
        ...

    @abstractmethod
    async def force_pause(self) -> None:
        """Immediately stop execution."""
        ...

    async def emit_event(self, event_type: str, data: dict, stage_type: str | None = None) -> None:
        """Write a structured event to the events table."""
        from datetime import datetime, timezone

        from pipeline.models import Event, StageType

        event = Event(
            run_id=self.run_id,
            stage_id=self.stage_id,
            stage_type=StageType(stage_type) if stage_type else None,
            event_type=event_type,
            data=data,
            ts=datetime.now(timezone.utc),
        )
        self.db.add(event)
        await self.db.flush()
