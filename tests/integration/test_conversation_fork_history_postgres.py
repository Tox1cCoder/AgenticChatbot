"""A forked child's context, memory and independence against real PostgreSQL.

The fork itself is ``ConversationForkService`` over the real repository (helpers
reused from ``test_conversation_fork_postgres``). Everything the child's next
turn reads is real too: ``ConversationHistoryProvider`` over the real message
and compaction repositories, ``ConversationCompactionWorker`` with its real
lease protocol, ``MessageService`` streaming with the real generation lifecycle.
Only the models are scripted: the summary generator is a deterministic fake
that lists the user turns it was shown, and the workflow is a stub that reads
history exactly as routing does (``agent_key="router"``) and answers.

Skipped without ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update

from app.ai.conversation_compactor import ConversationCompactor
from app.ai.conversation_memory import MEMORY_KEYS
from app.ai.history import ConversationHistoryProvider
from app.ai.token_counter import TokenCounter
from app.ai.workflow.state import build_checkpoint_thread_id
from app.api.ai_sdk import AISDKChatRequest, chat_ui_message_stream
from app.models.chat_image import ChatImage
from app.models.conversation import Conversation
from app.models.conversation_memory_summary import ConversationMemorySummary
from app.models.conversation_summary_job import ConversationSummaryJob
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.enums import MessageRole
from app.models.generation import Generation, GenerationStatus
from app.models.message import Message
from app.repositories.chat_image import ChatImageRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.conversation_compaction import ConversationCompactionRepository
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.generation import GenerationRepository
from app.repositories.message import MessageRepository
from app.schemas.custom_agent import runtime_agent_id_for
from app.schemas.message import MessageCreate, MessageRead, MessageUpdate
from app.services.conversation_fork_memory import ForkMemoryPreparer
from app.services.custom_agent_capability_resolver import resolve_custom_agent_capabilities
from app.services.custom_agent_service import CustomAgentService
from app.services.event_streaming.events import make_event
from app.services.generation_control_bus import InMemoryGenerationControlBus
from app.services.generation_control_service import GenerationControlService
from app.services.generation_registry import get_generation_registry
from app.services.message_service import MessageService
from app.workers.conversation_compaction import ConversationCompactionWorker, dispatch_due_jobs
from tests.integration import test_conversation_fork_postgres as fork_support
from tests.integration.test_conversation_fork_postgres import (
    ASSISTANT,
    USER,
    Pg,
    Source,
    _commit,
    _completed_turn,
    _conversation,
    _fork,
    _generation,
    _message,
    _messages,
    _request,
    _service,
    _user,
)

# The fork suite's PostgreSQL fixtures: schema, engines and per-test cleanup.
pg_engines = fork_support.pg_engines
pg = fork_support.pg
pytestmark = pytest.mark.selector_event_loop

FACT = "Remember this: my locker code is KESTREL-7319."
LEAK = "POST-CUTOFF"
LOGGER = "app.services.conversation_fork_memory"


def _settings(**overrides: Any) -> SimpleNamespace:
    """Production budget and compaction defaults, small enough to read."""
    values = {
        "conversation_summary_enabled": True,
        "conversation_fork_memory_prepare_timeout_seconds": 10.0,
        "conversation_summary_lease_seconds": 120,
        "conversation_summary_max_attempts": 5,
        "conversation_summary_timeout_seconds": 10,
        "conversation_summary_retry_base_seconds": 5,
        "conversation_summary_retry_max_seconds": 900,
        "router_history_max_messages": 12,
        "router_history_max_tokens": 3000,
        "chat_history_max_messages": 24,
        "chat_history_max_tokens": 9000,
        "rag_history_max_messages": 12,
        "rag_history_max_tokens": 3000,
        "search_history_max_messages": 16,
        "search_history_max_tokens": 5000,
        "planning_history_max_messages": 16,
        "planning_history_max_tokens": 5000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeSummarizer:
    """Deterministic summary model: one fact per user turn it was shown."""

    def __init__(self, *, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.transcripts: list[list[dict[str, Any]]] = []

    async def __call__(self, *, prompt: str, **_kwargs: Any) -> str:
        body = prompt.split("BEGIN_UNTRUSTED_TRANSCRIPT_JSON\n", 1)[1]
        transcript = json.loads(body.split("\nEND_UNTRUSTED_TRANSCRIPT_JSON", 1)[0])
        self.transcripts.append(transcript)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        payload = {key: [] for key in MEMORY_KEYS}
        payload["facts"] = [item["content"][:200] for item in transcript if item["role"] == "user"][
            :50
        ]
        return json.dumps(payload)


class Memory:
    """The child's memory machinery, wired the way the container wires it."""

    def __init__(self, pg: Pg, *, summarizer: FakeSummarizer | None = None, **overrides):
        self.settings = _settings(**overrides)
        self.compaction = ConversationCompactionRepository(
            pg.sync, async_session_factory=pg.async_factory
        )
        self.messages = MessageRepository(
            pg.sync, compaction_repository=self.compaction, async_session_factory=pg.async_factory
        )
        self.history = ConversationHistoryProvider(self.messages, self.compaction, self.settings)
        self.summarizer = summarizer or FakeSummarizer()
        self.runs: list[bool] = []
        self.worker = ConversationCompactionWorker(
            repository=self.compaction,
            compactor=ConversationCompactor(
                token_counter=TokenCounter(),
                generator=self.summarizer,
                provider="gemini",
                model="gemini-2.5-flash",
                trigger_messages=60,
                trigger_tokens=18_000,
                keep_recent_turns=4,
                max_summary_tokens=1_500,
            ),
            settings=self.settings,
            jitter=lambda _low, _high: 0.0,
        )
        self.preparer = ForkMemoryPreparer(
            repository=self.compaction, runner=self.run, settings=self.settings
        )

    async def run(self, conversation_id: UUID, *, force: bool = False):
        self.runs.append(force)
        return await self.worker.execute(conversation_id, force=force)

    async def router_context(self, child: UUID, owner: UUID, current: UUID | None = None):
        return await self.history.build_context(
            conversation_id=child, user_id=owner, current_message_id=current, agent_key="router"
        )


def _long_source(pg: Pg, *, turns: int, after: int = 2) -> Source:
    """``turns`` completed turns with ``FACT`` first, then ``after`` post-cutoff turns."""
    owner_id = _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id, title="Long"), {})
    for index in range(turns):
        question = FACT if index == 0 else f"question {index}"
        _completed_turn(pg, source, 2 * index + 1, question, f"answer {index}")
    for index in range(after):
        sequence = 2 * (turns + index) + 1
        _completed_turn(pg, source, sequence, f"{LEAK} question {index}", f"{LEAK} answer")
    return source


async def _fork_at(pg: Pg, source: Source, sequence: int, *, publisher: Any = None):
    service = _service(pg, publisher=publisher)
    read, created = await _fork(service, source, _request(source.ids[sequence]))
    assert created is True
    return read


def _job(pg: Pg, conversation_id: UUID) -> ConversationSummaryJob:
    with pg.sync() as session:
        return session.get(ConversationSummaryJob, conversation_id)


def _texts(context) -> list[str]:
    return [message.content for message in context.messages]


# ---------------------------------------------------------------------------
# first-turn preparation
# ---------------------------------------------------------------------------


async def test_a_fact_before_the_history_boundary_reaches_the_first_child_turn(pg):
    """40 copied rows: over the router's 12, under the background trigger of 60.

    Without preparation the first prompt holds rows 29-40 and no summary; the
    worker would also skip this prefix on its own, so nothing else adds it.
    """
    source = _long_source(pg, turns=20)
    child = await _fork_at(pg, source, 40)
    memory = Memory(pg)
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    await turns.stream("What is my locker code?")

    [(request, context)] = turns.seen
    assert context.memory is not None and FACT in context.memory.content
    assert memory.runs == [True]
    rows = [message for message in context.messages if message.role.value != "memory"]
    assert len(rows) <= 12
    assert context.memory_sequence == 32  # four recent turns stay verbatim
    assert all(LEAK not in text for text in _texts(context))
    assert request.conversation_id == str(child.id)


async def test_a_job_the_background_worker_idled_is_reopened_for_the_first_turn(pg):
    source = _long_source(pg, turns=20)
    child = await _fork_at(pg, source, 40)
    memory = Memory(pg)
    background = await memory.run(child.id)  # what Celery does with the fork's job
    assert background.status == "skipped" and _job(pg, child.id).status == "idle"

    result = await memory.preparer.aprepare(
        conversation_id=child.id, owner_id=source.owner_id, forked_through_sequence=40
    )

    assert (result.status, result.reason) == ("prepared", "summary_ready")
    context = await memory.router_context(child.id, source.owner_id)
    assert FACT in context.memory.content


async def test_after_compaction_the_child_uses_memory_and_the_normal_budget(pg):
    """64 copied rows trigger the background worker; preparation then has nothing to do."""
    source = _long_source(pg, turns=32)
    child = await _fork_at(pg, source, 64)
    memory = Memory(pg)

    background = await memory.run(child.id)
    result = await memory.preparer.aprepare(
        conversation_id=child.id, owner_id=source.owner_id, forked_through_sequence=64
    )

    assert background.status == "completed"
    assert (result.status, result.reason) == ("skipped", "memory_present")
    assert memory.runs == [False]
    context = await memory.router_context(child.id, source.owner_id)
    assert FACT in context.memory.content
    assert context.memory_sequence == 56
    assert _texts(context)[1:] == [
        text for row in _messages(pg, child.id) if row.sequence > 56 for text in [row.content]
    ]


async def test_a_lost_fork_notification_is_recovered_by_the_reconcile_sweep(pg):
    def broker_down(_conversation_id):
        raise ConnectionError("broker unavailable")

    source = _long_source(pg, turns=32)
    child = await _fork_at(pg, source, 64, publisher=broker_down)
    memory = Memory(pg)
    assert _job(pg, child.id).status == "pending"

    dispatched: list[UUID] = []
    dispatch_due_jobs(memory.compaction, dispatched.append, debounce_seconds=60, limit=500)
    assert child.id in dispatched
    outcome = await memory.run(child.id)

    assert outcome.status == "completed"
    context = await memory.router_context(child.id, source.owner_id)
    assert FACT in context.memory.content


def _claim_elsewhere(memory: Memory, child: UUID) -> None:
    claim = memory.compaction.claim_job(child, lease_seconds=120)
    assert claim is not None


async def test_a_busy_lease_is_waited_on_and_its_summary_serves_the_first_turn(pg, caplog):
    """64 copied rows: the fork's own background run claims the job before the first message.

    The first turn waits for that run instead of falling back at once, and
    never takes a second lease.
    """
    source = _long_source(pg, turns=32)
    child = await _fork_at(pg, source, 64)
    memory = Memory(pg, summarizer=FakeSummarizer(delay=0.6))
    background = asyncio.create_task(memory.run(child.id))  # what Celery does after the fork
    await _until(lambda: _job(pg, child.id).status == "processing")
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = await turns.stream("What is my locker code?")

    assert [event.type for event in events][-1] == "complete"
    assert (await background).status == "completed"
    [(_request_, context)] = turns.seen
    assert context.memory is not None and FACT in context.memory.content
    assert memory.runs == [False]  # only the background run: no second lease
    assert len(memory.summarizer.transcripts) == 1
    assert [record for record in caplog.records if record.name == LOGGER] == []


async def test_a_lease_held_past_the_timeout_falls_back_within_the_timeout(pg, caplog):
    source = _long_source(pg, turns=32)
    child = await _fork_at(pg, source, 64)
    memory = Memory(pg, conversation_fork_memory_prepare_timeout_seconds=0.5)
    _claim_elsewhere(memory, child.id)  # a worker that holds the lease for 120 s
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = await turns.stream("What is my locker code?")
    elapsed = time.monotonic() - started

    assert [event.type for event in events][-1] == "complete"
    assert 0.5 <= elapsed < 2.0
    assert memory.runs == []
    assert _job(pg, child.id).status == "processing"
    [(_request_, context)] = turns.seen
    assert context.memory is None and FACT not in " ".join(_texts(context))
    [warning] = [record for record in caplog.records if record.name == LOGGER]
    assert warning.fork_memory["reason"] == "timeout"


async def test_a_short_fork_with_two_very_long_replies_skips_quietly(pg, caplog):
    """Over the token budget, but the compactor keeps both turns verbatim: no_complete_turn."""
    owner_id = _user(pg)
    source = Source(owner_id, _conversation(pg, owner_id, title="Long replies"), {})
    _completed_turn(pg, source, 1, FACT, "a" * 40_000)
    _completed_turn(pg, source, 3, "question 1", "b" * 40_000)
    child = await _fork_at(pg, source, 4)
    memory = Memory(pg)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        result = await memory.preparer.aprepare(
            conversation_id=child.id, owner_id=owner_id, forked_through_sequence=4
        )

    assert (result.status, result.reason) == ("skipped", "no_complete_turn")
    assert memory.runs == [True]
    assert memory.summarizer.transcripts == []  # no provider call was made
    warnings = [
        record
        for record in caplog.records
        if record.name == LOGGER and record.levelno >= logging.WARNING
    ]
    assert warnings == []


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("disabled", "compaction_disabled"),
        ("provider_failure", "summary_failed"),
        ("timeout", "timeout"),
    ],
)
async def test_every_preparation_failure_proceeds_with_bounded_history(pg, caplog, case, reason):
    source = _long_source(pg, turns=20)
    child = await _fork_at(pg, source, 40)
    summarizer = FakeSummarizer(
        error=RuntimeError("provider 503") if case == "provider_failure" else None,
        delay=1.0 if case == "timeout" else 0.0,
    )
    memory = Memory(
        pg,
        summarizer=summarizer,
        conversation_summary_enabled=case != "disabled",
        conversation_fork_memory_prepare_timeout_seconds=0.3 if case == "timeout" else 10.0,
    )
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = await turns.stream("What is my locker code?")

    assert [event.type for event in events][-1] == "complete"
    [(_request_, context)] = turns.seen
    assert context.memory is None
    assert len(context.messages) <= 12 and FACT not in " ".join(_texts(context))
    [warning] = [record for record in caplog.records if record.name == LOGGER]
    assert warning.fork_memory["conversation_id"] == str(child.id)
    assert warning.fork_memory["reason"] == reason
    if case == "provider_failure":
        assert len(summarizer.transcripts) == 1
        assert warning.fork_memory["detail"] == "generation_failed"
    if case == "timeout":
        # The timed-out run kept its lease and finished; the next turn has memory.
        await _until(
            lambda: memory.compaction.get_owned_valid_memory(child.id, source.owner_id) is not None
        )
        later = await memory.router_context(child.id, source.owner_id)
        assert FACT in later.memory.content


@pytest.mark.parametrize("provider", ["hangs", "fails_fast"])
async def test_a_failing_provider_costs_only_the_first_turn(pg, caplog, provider):
    """Turn 1 tries and falls back; turn 2 neither claims, calls the provider, nor waits.

    Persisting turn 1's reply re-queues the job with its attempts reset, so
    without the first-turn gate turn 2 would pay the same claim and wait.
    """
    source = _long_source(pg, turns=20)
    child = await _fork_at(pg, source, 40)
    hangs = provider == "hangs"
    summarizer = FakeSummarizer(
        delay=5.0 if hangs else 0.0, error=None if hangs else RuntimeError("provider 503")
    )
    memory = Memory(
        pg,
        summarizer=summarizer,
        conversation_fork_memory_prepare_timeout_seconds=0.3,
        conversation_summary_timeout_seconds=1,
    )
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await turns.stream("What is my locker code?")
        first_warnings = len([r for r in caplog.records if r.name == LOGGER])
        started = time.monotonic()
        events = await turns.stream("Are you sure?")
        second_turn_seconds = time.monotonic() - started

    assert [event.type for event in events][-1] == "complete"
    assert first_warnings == 1
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1
    assert memory.runs == [True]
    assert len(summarizer.transcripts) == 1
    assert second_turn_seconds < 1.0
    [_first, (_request_, second_context)] = turns.seen
    assert second_context.memory is None
    # Let turn 1's leased run end before cleanup.
    await _until(lambda: _job(pg, child.id).status != "processing")


async def test_a_later_turn_with_invalidated_memory_is_not_prepared_synchronously(pg):
    source = _long_source(pg, turns=20)
    child = await _fork_at(pg, source, 40)
    memory = Memory(pg)
    turns = ChildTurns(pg, memory, child.id, source.owner_id)
    await turns.stream("What is my locker code?")
    assert memory.runs == [True]
    covered = next(row for row in _messages(pg, child.id) if row.sequence == 2)
    assert memory.compaction.invalidate_for_mutation(covered.id, new_content="edited") is True

    await turns.stream("And now?")

    assert memory.runs == [True]
    [_first, (_request_, context)] = turns.seen
    assert context.memory is None


async def _until(condition, *, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# nothing past the cutoff
# ---------------------------------------------------------------------------


def _canvas(revision: int, content: str) -> dict[str, Any]:
    return {
        "canvas_artifact": {
            "artifact_id": "canvas:main",
            "revision": revision,
            "operation": "create" if revision == 1 else "update",
            "content": content,
            "language": "html",
            "title": "Plan",
        }
    }


def _with_metadata(pg: Pg, message_id: UUID, extra: dict[str, Any]) -> None:
    with pg.sync.begin() as session:
        row = session.get(Message, message_id)
        row.message_metadata = {**(row.message_metadata or {}), **extra}


async def test_child_prompt_summary_and_canvas_never_contain_rows_past_the_cutoff(pg):
    source = _long_source(pg, turns=20, after=2)
    _with_metadata(pg, source.ids[4], _canvas(1, "<p>before the cutoff</p>"))
    _with_metadata(pg, source.ids[44], _canvas(2, f"<p>{LEAK} canvas</p>"))
    with pg.sync.begin() as session:
        session.add(
            ConversationMemorySummary(
                conversation_id=source.conversation_id,
                summary_payload={**{key: [] for key in MEMORY_KEYS}, "facts": [f"{LEAK} memory"]},
                last_summarized_sequence=44,
                is_valid=True,
            )
        )
    child = await _fork_at(pg, source, 40)
    memory = Memory(pg)
    turns = ChildTurns(pg, memory, child.id, source.owner_id)

    await turns.stream("Continue from here")

    [(_request_, context)] = turns.seen
    assert all(LEAK not in text for text in _texts(context))
    [transcript] = memory.summarizer.transcripts
    assert max(item["sequence"] for item in transcript) <= 40
    assert all(LEAK not in item["content"] for item in transcript)
    canvas = await memory.history.get_latest_canvas_artifact(
        conversation_id=child.id, user_id=source.owner_id
    )
    assert (canvas.revision, canvas.content) == (1, "<p>before the cutoff</p>")
    source_canvas = await memory.history.get_latest_canvas_artifact(
        conversation_id=source.conversation_id, user_id=source.owner_id
    )
    assert source_canvas.revision == 2


# ---------------------------------------------------------------------------
# a fresh turn identity through both entry points
# ---------------------------------------------------------------------------


class ChildTurns:
    """A real ``MessageService`` whose workflow reads history the way routing does."""

    def __init__(self, pg: Pg, memory: Memory, conversation_id: UUID, owner_id: UUID) -> None:
        self.pg = pg
        self.memory = memory
        self.conversation_id = conversation_id
        self.owner_id = owner_id
        self.seen: list[tuple[Any, Any]] = []
        self.service = self._build()

    def _build(self) -> MessageService:
        service = MessageService.__new__(MessageService)
        for name in (
            "_turn_coordinator",
            "custom_agent_service",
            "chat_image_service",
            "web_image_service",
            "tool_approval_setting_repository",
            "tool_approval_repository",
            "hitl_interrupt_repository",
            "task_plan_service",
            "project_context_service",
            "redis_client",
        ):
            setattr(service, name, None)
        service.repository = self.memory.messages
        service.fork_memory_preparer = self.memory.preparer
        service.generation_control_service = GenerationControlService(
            repository=GenerationRepository(
                session_factory=self.pg.sync, async_session_factory=self.pg.async_factory
            ),
            bus=InMemoryGenerationControlBus(),
            stop_wait_seconds=0.5,
        )

        async def aget_by_id(conversation_id):
            with self.pg.sync() as session:
                return session.get(Conversation, conversation_id)

        async def allowed(*_args):
            return None

        service.conversation_validation_utils = SimpleNamespace(
            avalidate_conversation_access=allowed,
            conversation_repository=SimpleNamespace(aget_by_id=aget_by_id),
        )
        service.ai_service = SimpleNamespace(
            invalidate_history_cache=self.memory.history.invalidate,
            execute_request_stream=self._workflow,
        )
        service._persist_completed_workflow_response = self._persist
        service._compact_checkpoint_after_persist = _nothing
        return service

    async def _workflow(self, request):
        context = await self.memory.history.build_context(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            current_message_id=request.user_message_id,
            agent_key="router",
        )
        self.seen.append((request, context))
        yield make_event("message_delta", sequence=1, data={"text": "fresh answer"})
        yield make_event("complete", sequence=2, data={"response": None})

    async def _persist(self, **kwargs) -> MessageRead:
        row = await self.memory.messages.acreate(
            {
                "id": kwargs["message_id"],
                "conversation_id": kwargs["conversation_id"],
                "sender": ASSISTANT,
                "content": "fresh answer",
                "message_metadata": {
                    "reply_to_user_message_id": str(kwargs["reply_to_user_message_id"])
                },
            }
        )
        return MessageRead.model_validate(row)

    async def stream(self, content: str) -> list[Any]:
        stream = self.service.create_message_stream(
            MessageCreate(
                conversation_id=self.conversation_id, content=content, role=MessageRole.user
            ),
            self.owner_id,
        )
        return [event async for event in stream]

    async def ai_sdk(self, content: str) -> str:
        response = await chat_ui_message_stream(
            conversation_id=self.conversation_id,
            payload=AISDKChatRequest(
                messages=[{"role": "user", "parts": [{"type": "text", "text": content}]}]
            ),
            message_service=self.service,
            current_user_id=self.owner_id,
        )
        return "".join([chunk async for chunk in response.body_iterator])


async def _nothing(**_kwargs) -> None:
    return None


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_generation_registry()
    registry._store.clear()  # noqa: SLF001 - test isolation
    yield
    registry._store.clear()  # noqa: SLF001


def _source_with_execution_handles(pg: Pg) -> tuple[Source, set[str]]:
    """Copied turns carrying approval/continuation handles, and a paused later turn."""
    source = _long_source(pg, turns=2, after=0)
    approved_turn = _message(pg, source.conversation_id, 5, USER, "delete the draft")
    handles = {"interrupt-approved", "thread-approved", "interrupt-paused"}
    approved = _message(
        pg,
        source.conversation_id,
        6,
        ASSISTANT,
        "Deleted after approval.",
        {
            "reply_to_user_message_id": str(approved_turn),
            "interrupt": {"interrupt_id": "interrupt-approved", "thread_id": "thread-approved"},
            "thread_id": "thread-approved",
        },
    )
    generation = _generation(
        pg,
        source,
        turn=approved_turn,
        status=GenerationStatus.COMPLETED,
        assistant_message_id=approved,
    )
    source.ids.update({5: approved_turn, 6: approved})
    paused_turn = _message(pg, source.conversation_id, 7, USER, "and the backups")
    paused = _message(
        pg,
        source.conversation_id,
        8,
        ASSISTANT,
        "I need approval to continue.",
        {"paused": True, "interrupt": {"interrupt_id": "interrupt-paused"}, "thread_id": "t-p"},
    )
    paused_generation = _generation(
        pg,
        source,
        turn=paused_turn,
        status=GenerationStatus.CONTINUABLE,
        assistant_message_id=paused,
        continuation_available=True,
    )
    handles |= {
        str(generation),
        str(paused_generation),
        str(source.conversation_id),
        build_checkpoint_thread_id(str(source.conversation_id), str(approved_turn)),
        build_checkpoint_thread_id(str(source.conversation_id), str(paused_turn)),
    }
    return source, handles


def _generations(pg: Pg, conversation_id: UUID) -> list[Generation]:
    with pg.sync() as session:
        return list(
            session.execute(
                select(Generation).where(Generation.conversation_id == conversation_id)
            ).scalars()
        )


@pytest.mark.parametrize("entry", ["stream", "ai_sdk"])
async def test_a_child_turn_starts_a_fresh_generation_and_thread(pg, entry):
    source, handles = _source_with_execution_handles(pg)
    before = {(g.id, g.status) for g in _generations(pg, source.conversation_id)}
    child = await _fork_at(pg, source, 6)
    turns = ChildTurns(pg, Memory(pg), child.id, source.owner_id)

    if entry == "stream":
        events = await turns.stream("next step")
        assert [event.type for event in events][-1] == "complete"
    else:
        body = await turns.ai_sdk("next step")
        assert body.rstrip().endswith("data: [DONE]") and "fresh answer" in body

    [(request, _context)] = turns.seen
    [generation] = _generations(pg, child.id)
    turn = UUID(request.user_message_id)
    assert generation.logical_turn_id == str(turn)
    assert generation.checkpoint_thread_id == build_checkpoint_thread_id(str(child.id), str(turn))
    assert generation.status == GenerationStatus.COMPLETED
    assert {(g.id, g.status) for g in _generations(pg, source.conversation_id)} == before
    serialized = request.model_dump_json()
    assert request.thread_id is None
    assert not {handle for handle in handles if handle in serialized}
    for row in _messages(pg, child.id):
        assert not {handle for handle in handles if handle in json.dumps(row.message_metadata)}


# ---------------------------------------------------------------------------
# independence: edits, deletes, caches, summaries, source deletion
# ---------------------------------------------------------------------------


def _valid_memory(pg: Pg, conversation_id: UUID, through: int) -> None:
    with pg.sync.begin() as session:
        session.merge(
            ConversationMemorySummary(
                conversation_id=conversation_id,
                summary_payload={**{key: [] for key in MEMORY_KEYS}, "facts": ["known"]},
                last_summarized_sequence=through,
                is_valid=True,
            )
        )


def _memory_row(pg: Pg, conversation_id: UUID) -> ConversationMemorySummary:
    with pg.sync() as session:
        return session.get(ConversationMemorySummary, conversation_id)


def _mutating_service(memory: Memory, invalidated: list[str]) -> MessageService:
    service = MessageService.__new__(MessageService)
    service.repository = memory.messages
    service.message_validation_utils = SimpleNamespace(validate_message_access=lambda *_: None)

    def invalidate(conversation_id: str) -> None:
        invalidated.append(conversation_id)
        memory.history.invalidate(conversation_id)

    service.ai_service = SimpleNamespace(invalidate_history_cache=invalidate)
    return service


async def test_edits_and_deletes_stay_in_their_own_conversation(pg):
    source = _long_source(pg, turns=3, after=0)
    child = await _fork_at(pg, source, 6)
    _valid_memory(pg, source.conversation_id, 4)
    _valid_memory(pg, child.id, 4)
    _commit(
        pg,
        update(ConversationSummaryJob)
        .where(ConversationSummaryJob.conversation_id == child.id)
        .values(status="idle"),
    )
    memory = Memory(pg)
    invalidated: list[str] = []
    service = _mutating_service(memory, invalidated)
    owner = source.owner_id
    await memory.router_context(source.conversation_id, owner)
    await memory.router_context(child.id, owner)
    child_rows = {row.sequence: row for row in _messages(pg, child.id)}

    service.update_message(child_rows[2].id, owner, MessageUpdate(content="child edit"))

    assert invalidated == [str(child.id)]
    cached = {key[0] for key in memory.history._cache}  # noqa: SLF001
    assert cached == {str(source.conversation_id)}
    assert _memory_row(pg, child.id).is_valid is False
    assert _memory_row(pg, source.conversation_id).is_valid is True
    assert _job(pg, child.id).status == "pending"
    source_rows = {row.sequence: row for row in _messages(pg, source.conversation_id)}
    assert source_rows[2].content == "answer 0"

    service.delete_message(source_rows[4].id, owner)

    assert invalidated == [str(child.id), str(source.conversation_id)]
    assert _memory_row(pg, source.conversation_id).is_valid is False
    assert {row.sequence: row.deleted_at for row in _messages(pg, child.id)}[4] is None
    child_context = await memory.router_context(child.id, owner)
    assert "child edit" in _texts(child_context) and "answer 1" in _texts(child_context)


async def test_soft_deleting_the_source_leaves_the_child_text_media_and_canvas(pg):
    source = _long_source(pg, turns=2, after=0)
    with pg.sync.begin() as session:
        image = ChatImage(
            conversation_id=source.conversation_id,
            user_id=source.owner_id,
            sha256="c" * 64,
            size_bytes=1,
            storage_path="image.png",
        )
        session.add(image)
    image_reference = {"image_id": str(image.id), "url": f"/chat-images/{image.id}"}
    _with_metadata(pg, source.ids[4], {**_canvas(1, "<p>kept</p>"), "images": [image_reference]})
    child = await _fork_at(pg, source, 4)

    assert ConversationRepository(pg.sync).delete(source.conversation_id) is True

    memory = Memory(pg)
    context = await memory.router_context(child.id, source.owner_id)
    assert _texts(context) == [FACT, "answer 0", "question 1", "answer 1"]
    canvas = await memory.history.get_latest_canvas_artifact(
        conversation_id=child.id, user_id=source.owner_id
    )
    assert canvas.content == "<p>kept</p>"
    [reply] = [row for row in _messages(pg, child.id) if row.sequence == 4]
    assert reply.message_metadata["images"] == [image_reference]
    assert ChatImageRepository(pg.sync).get_for_user(image.id, source.owner_id) is not None
    with pg.sync() as session:
        assert session.get(Conversation, child.id).deleted_at is None


# ---------------------------------------------------------------------------
# custom agents: shared definitions resolved on the requesting device
# ---------------------------------------------------------------------------

SAVED_TOOL = {
    "type": "client",
    "device_id": "device-a",
    "session_id": "session-a",
    "catalog_version": "1",
    "tool_instance_id": "instance-a",
    "server_name": "desktop-commander",
    "qualified_tool_id": "desktop-commander::read_file",
    "tool_name": "read_file",
}
LIVE_ON_B = {
    **SAVED_TOOL,
    "device_id": "device-b",
    "session_id": "session-b",
    "catalog_version": "4",
    "tool_instance_id": "instance-b",
}


def _attach_agents(pg: Pg, source: Source) -> tuple[UUID, UUID]:
    ids = []
    with pg.sync.begin() as session:
        for index, name in enumerate(("Reader", "Retired")):
            agent = CustomAgent(
                owner_id=source.owner_id,
                name=name,
                slug=f"{name.lower()}-{uuid4().hex[:6]}",
                prompt="p",
                provider_type="gemini",
                model="m",
                tool_refs=[SAVED_TOOL] if index == 0 else [],
            )
            session.add(agent)
            session.flush()
            ids.append(agent.id)
            session.add(
                ConversationCustomAgent(
                    owner_id=source.owner_id,
                    conversation_id=source.conversation_id,
                    custom_agent_id=agent.id,
                    agent_order=index,
                )
            )
    return ids[0], ids[1]


async def test_copied_agent_attachments_resolve_on_the_requesting_device(pg):
    source = _long_source(pg, turns=2, after=0)
    reader, retired = _attach_agents(pg, source)
    _with_metadata(pg, source.ids[4], {"agent": {"id": runtime_agent_id_for(retired)}})
    child = await _fork_at(pg, source, 4)
    _commit(pg, update(CustomAgent).where(CustomAgent.id == retired).values(deleted_at=func.now()))
    agents = CustomAgentService.__new__(CustomAgentService)
    agents.repository = CustomAgentRepository(pg.sync, async_session_factory=pg.async_factory)

    state = await agents.abuild_runtime_state(source.owner_id, child.id)

    # The deleted definition degrades the way it does in any conversation: gone.
    assert list(state) == [runtime_agent_id_for(reader)]
    previous = await Memory(pg).history.get_previous_final_agent_id(
        conversation_id=child.id, user_id=source.owner_id
    )
    assert previous == runtime_agent_id_for(retired) and previous not in state
    saved = state[runtime_agent_id_for(reader)]["tool_refs"]
    on_b = resolve_custom_agent_capabilities(
        selected_tool_refs=saved,
        selected_skill_refs=[],
        live_tool_refs=[LIVE_ON_B],
        live_skill_refs=[],
        request_device_id="device-b",
        device_available=True,
    )
    assert on_b.status == "ready"
    assert on_b.resolved_client_tool_refs[0]["session_id"] == "session-b"
    offline = resolve_custom_agent_capabilities(
        selected_tool_refs=saved,
        selected_skill_refs=[],
        live_tool_refs=[LIVE_ON_B],
        live_skill_refs=[],
        request_device_id="device-a",
        device_available=False,
    )
    assert offline.status == "device_unavailable" and offline.resolved_client_tool_refs == []
