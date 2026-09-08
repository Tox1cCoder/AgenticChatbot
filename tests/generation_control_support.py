"""Shared doubles for driving the *real* generation lifecycle in tests.

The point of putting the repository double here rather than faking
:class:`GenerationControlService` is that the service is where lifecycle
legality lives — which statuses a Stop may be issued from, that a replayed
command returns its own recorded result, that a continuation id is single-use.
Faking the service would assert nothing about any of it.

``FakeRepository`` is faithful about the two things the service is written
against: the ``version`` fence, and returning ``None`` (not raising) for a
transition it refuses. What it cannot model is real concurrency; that lives in
``tests/integration/test_generation_repository_postgres.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

from app.models.generation import GenerationStatus
from app.schemas.generation import CommandClaim, CreateGeneration, GenerationSnapshot
from app.services.generation_control_bus import InMemoryGenerationControlBus
from app.services.generation_control_service import GenerationControlService

CONVERSATION_ID = UUID("11111111-1111-1111-1111-111111111111")
USER_ID = UUID("22222222-2222-2222-2222-222222222222")

__all__ = [
    "CONVERSATION_ID",
    "USER_ID",
    "FakeRepository",
    "build_control_service",
]


class FakeRepository:
    """Enough of the repository to hold the service to its own contract."""

    def __init__(self) -> None:
        self.rows: dict[UUID, dict] = {}
        self.commands: dict[tuple[UUID, str], dict] = {}
        self.transitions: list[dict] = []

    async def acreate(self, command: CreateGeneration) -> GenerationSnapshot:
        generation_id = uuid4()
        self.rows[generation_id] = {
            "id": generation_id,
            "logical_turn_id": command.logical_turn_id,
            "conversation_id": command.conversation_id,
            "user_id": command.user_id,
            "checkpoint_thread_id": command.checkpoint_thread_id,
            "active_agent_id": command.active_agent_id,
            "status": GenerationStatus.STARTING,
            "version": 1,
            "execution_epoch": 0,
            "continuation_id": None,
            "continuation_available": False,
            "continuation_block_reason": None,
            "assistant_message_id": None,
            "terminal_reason": None,
            "research_accounting": None,
        }
        return self._snapshot(generation_id)

    async def aget_owned(self, generation_id, user_id, conversation_id):
        row = self.rows.get(generation_id)
        if row is None or row["user_id"] != user_id or row["conversation_id"] != conversation_id:
            return None
        return self._snapshot(generation_id)

    async def aget_by_logical_turn(self, logical_turn_id, user_id, conversation_id):
        for generation_id, row in self.rows.items():
            if (
                str(row["logical_turn_id"]) == str(logical_turn_id)
                and row["user_id"] == user_id
                and row["conversation_id"] == conversation_id
            ):
                return self._snapshot(generation_id)
        return None

    async def atransition(
        self,
        *,
        generation_id,
        user_id,
        conversation_id,
        expected_statuses,
        expected_version,
        values,
    ):
        self.transitions.append({"generation_id": generation_id, "values": dict(values)})
        row = self.rows.get(generation_id)
        if row is None or row["user_id"] != user_id or row["conversation_id"] != conversation_id:
            return None
        if row["version"] != expected_version or row["status"] not in expected_statuses:
            return None
        row.update(values)
        row["version"] += 1
        return self._snapshot(generation_id)

    async def aget_resume_context(self, generation_id, user_id, conversation_id):
        row = self.rows.get(generation_id)
        if row is None or row["user_id"] != user_id or row["conversation_id"] != conversation_id:
            return None
        return SimpleNamespace(
            checkpoint_thread_id=row["checkpoint_thread_id"],
            active_agent_id=row["active_agent_id"],
            research_accounting=row["research_accounting"],
            execution_budget=row.get("execution_budget"),
        )

    async def aclaim_command(self, *, generation_id, idempotency_key, action, fence):
        key = (generation_id, idempotency_key)
        existing = self.commands.get(key)
        if existing is not None:
            return CommandClaim(
                claimed=False,
                action=existing["action"],
                fence=existing["fence"],
                result=existing["result"],
            )
        self.commands[key] = {"action": action, "fence": fence, "result": None}
        return CommandClaim(claimed=True, action=action, fence=fence, result=None)

    async def arecord_command_result(self, *, generation_id, idempotency_key, result):
        entry = self.commands.get((generation_id, idempotency_key))
        if entry is not None and entry["result"] is None:
            entry["result"] = result

    # -- helpers -------------------------------------------------------

    def _snapshot(self, generation_id: UUID) -> GenerationSnapshot:
        row = self.rows[generation_id]

        class _Row:
            def __init__(self, values: dict) -> None:
                self.__dict__.update(values)

        return GenerationSnapshot.from_row(_Row(row))

    def status_of(self, generation_id: UUID) -> GenerationStatus:
        return self.rows[generation_id]["status"]

    def force(self, generation_id: UUID, **values) -> GenerationSnapshot:
        self.rows[generation_id].update(values)
        return self._snapshot(generation_id)


def build_control_service(*, stop_wait_seconds: float = 0.05) -> GenerationControlService:
    """A real control service over an in-memory repository and bus.

    The repository and bus are reachable as ``_test_repository`` and
    ``_test_bus`` so a test can assert on rows and published signals without
    threading them through every helper.
    """
    repository = FakeRepository()
    bus = InMemoryGenerationControlBus()
    control = GenerationControlService(
        repository=repository, bus=bus, stop_wait_seconds=stop_wait_seconds
    )
    control._test_repository = repository  # type: ignore[attr-defined]
    control._test_bus = bus  # type: ignore[attr-defined]
    return control
