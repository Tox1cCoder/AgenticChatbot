"""Benchmark the conversation fork snapshot against the disposable test database.

Forks prefixes of 100, 1,000 and 5,000 messages carrying large metadata through
the real ``ConversationForkService`` and repository (one REPEATABLE READ
transaction on the async transport). No model is called and no compaction
notification is published. For each size it records:

* latency of ``fork_conversation`` (median and range over timed runs),
* peak Python allocation during one extra run (``tracemalloc``; it slows the
  run, so that run is not timed),
* rows copied and their UTF-8 bytes (content plus serialized metadata, measured
  in PostgreSQL on the child),
* SQL statements executed per fork (cursor executions, an ``executemany``
  counting once) and how many of them were batched.

Only ``chatbot_test`` is ever used; the URL is derived from settings and never
printed. Every row the benchmark creates is deleted at the end.

Usage::

    .venv/Scripts/python.exe scripts/benchmark_conversation_fork.py [--runs 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Text, cast, create_engine, delete, event, func, insert, select  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.base import Base  # noqa: E402
from app.models.chat_image import ChatImage  # noqa: E402
from app.models.conversation import Conversation  # noqa: E402
from app.models.conversation_memory_summary import ConversationMemorySummary  # noqa: E402
from app.models.conversation_summary_job import ConversationSummaryJob  # noqa: E402
from app.models.custom_agent import ConversationCustomAgent, CustomAgent  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.enums import MessageRole  # noqa: E402
from app.models.feedback import Feedback  # noqa: E402
from app.models.generation import Generation, GenerationCommand, GenerationStatus  # noqa: E402
from app.models.message import Message  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.user import User  # noqa: E402
from app.models.web_image_reference import WebImageReference  # noqa: E402
from app.repositories.conversation_fork import ConversationForkRepository  # noqa: E402
from app.schemas.conversation import ConversationForkRequest  # noqa: E402
from app.services.conversation_fork_service import ConversationForkService  # noqa: E402
from app.services.conversation_turn_coordinator import (  # noqa: E402
    ConversationTurnCoordinator,
    InProcessTurnLockBackend,
)

TEST_DATABASE = "chatbot_test"
SIZES = (100, 1_000, 5_000)
_TABLES = [
    User.__table__,
    Project.__table__,
    Conversation.__table__,
    Message.__table__,
    Feedback.__table__,
    Generation.__table__,
    GenerationCommand.__table__,
    ConversationMemorySummary.__table__,
    ConversationSummaryJob.__table__,
    CustomAgent.__table__,
    ConversationCustomAgent.__table__,
    Document.__table__,
    ChatImage.__table__,
    WebImageReference.__table__,
]


@dataclass
class Result:
    messages: int
    source_bytes: int
    runs: int
    latency_median_ms: float
    latency_min_ms: float
    latency_max_ms: float
    peak_python_mib: float
    rows_copied: int
    bytes_copied: int
    statements: int
    batched_statements: int


def _test_url() -> str:
    url = make_url(settings.database_url).set(
        database=TEST_DATABASE, drivername="postgresql+psycopg"
    )
    if url.database != TEST_DATABASE:
        raise SystemExit("refusing to benchmark anything but the disposable test database")
    # render_as_string: str(URL) masks the password. The URL is never printed.
    return url.render_as_string(hide_password=False)


def _assistant_metadata(turn: int, reply_to: UUID) -> dict:
    """About 9 KiB: reasoning, ten web sources, suggestions and a tool artifact."""
    return {
        "reply_to_user_message_id": str(reply_to),
        "model": "gemini-2.5-flash",
        "provider": "gemini",
        "reasoning_summary": f"Turn {turn} reasoning. " + ("Considered the options. " * 60),
        "web_sources": [
            {
                "url": f"https://example.com/{turn}/{index}",
                "title": f"Source {index} for turn {turn}",
                "snippet": "Relevant passage from the source. " * 14,
            }
            for index in range(10)
        ],
        "suggested_questions": [f"Follow-up {index} for turn {turn}?" for index in range(3)],
        "tool_artifacts": [{"tool": "web_search", "output": "raw tool output " * 120}],
    }


def _seed_source(sync: sessionmaker, owner_id: UUID, size: int) -> tuple[UUID, UUID]:
    """A ``size``-message source of completed turns; returns (conversation, last reply)."""
    conversation_id = uuid4()
    started = datetime(2026, 1, 1, tzinfo=UTC)
    messages: list[dict] = []
    generations: list[dict] = []
    for turn in range(size // 2):
        user_id, reply_id = uuid4(), uuid4()
        sequence = 2 * turn + 1
        messages.append(
            {
                "id": user_id,
                "conversation_id": conversation_id,
                "sender": MessageRole.user.value,
                "content": f"Question {turn}: " + ("please explain the next part. " * 16),
                "message_metadata": {},
                "sequence": sequence,
                "created_at": started + timedelta(seconds=sequence),
                "updated_at": started + timedelta(seconds=sequence),
            }
        )
        messages.append(
            {
                "id": reply_id,
                "conversation_id": conversation_id,
                "sender": MessageRole.assistant.value,
                "content": f"Answer {turn}: " + ("Here is the explanation in detail. " * 28),
                "message_metadata": _assistant_metadata(turn, user_id),
                "sequence": sequence + 1,
                "created_at": started + timedelta(seconds=sequence + 1),
                "updated_at": started + timedelta(seconds=sequence + 1),
            }
        )
        generations.append(
            {
                "id": uuid4(),
                "logical_turn_id": str(user_id),
                "checkpoint_thread_id": f"{conversation_id}:{user_id}",
                "user_id": owner_id,
                "conversation_id": conversation_id,
                "status": GenerationStatus.COMPLETED,
                "assistant_message_id": reply_id,
                "continuation_available": False,
            }
        )
    with sync.begin() as session:
        session.add(
            Conversation(
                id=conversation_id,
                owner_id=owner_id,
                title=f"Benchmark {size}",
                next_message_sequence=size + 1,
            )
        )
    with sync.begin() as session:
        for start in range(0, len(messages), 1_000):
            session.execute(insert(Message), messages[start : start + 1_000])
        session.execute(insert(Generation), generations)
    return conversation_id, messages[-1]["id"]


def _conversation_bytes(sync: sessionmaker, conversation_id: UUID) -> tuple[int, int]:
    with sync() as session:
        rows, size = session.execute(
            select(
                func.count(),
                func.coalesce(
                    func.sum(
                        func.octet_length(Message.content)
                        + func.octet_length(cast(Message.message_metadata, Text))
                    ),
                    0,
                ),
            ).where(Message.conversation_id == conversation_id)
        ).one()
    return int(rows), int(size)


class StatementCounter:
    def __init__(self, engine) -> None:
        self.statements = 0
        self.batched = 0
        event.listen(engine.sync_engine, "before_cursor_execute", self._count)

    def _count(self, _conn, _cursor, _statement, _parameters, _context, executemany) -> None:
        self.statements += 1
        if executemany:
            self.batched += 1

    def reset(self) -> None:
        self.statements = 0
        self.batched = 0


async def _fork_once(service, owner_id: UUID, source_id: UUID, reply_id: UUID):
    read, created = await service.fork_conversation(
        source_conversation_id=source_id,
        owner_id=owner_id,
        request=ConversationForkRequest(message_id=reply_id, request_id=uuid4()),
    )
    assert created, "every benchmark run must create a new child"
    return read


async def _measure(size: int, runs: int, sync, async_factory, counter) -> Result:
    owner_id = uuid4()
    with sync.begin() as session:
        session.add(
            User(
                id=owner_id,
                username=f"bench-{owner_id.hex[:20]}",
                email=f"{owner_id}@example.test",
                password_hash="benchmark",
            )
        )
    try:
        source_id, reply_id = _seed_source(sync, owner_id, size)
        _rows, source_bytes = _conversation_bytes(sync, source_id)
        service = ConversationForkService(
            repository=ConversationForkRepository(sync, async_session_factory=async_factory),
            turn_coordinator=ConversationTurnCoordinator(
                backend=InProcessTurnLockBackend(), timeout_seconds=5.0
            ),
            compaction_publisher=None,
            lock_timeout_seconds=5.0,
            max_messages=settings.conversation_fork_max_messages,
            max_bytes=settings.conversation_fork_max_bytes,
        )
        await _fork_once(service, owner_id, source_id, reply_id)  # warm-up
        latencies: list[float] = []
        child = None
        for _ in range(runs):
            counter.reset()
            started = time.perf_counter()
            child = await _fork_once(service, owner_id, source_id, reply_id)
            latencies.append((time.perf_counter() - started) * 1000)
        statements, batched = counter.statements, counter.batched
        tracemalloc.start()
        await _fork_once(service, owner_id, source_id, reply_id)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows_copied, bytes_copied = _conversation_bytes(sync, child.id)
        return Result(
            messages=size,
            source_bytes=source_bytes,
            runs=runs,
            latency_median_ms=round(statistics.median(latencies), 1),
            latency_min_ms=round(min(latencies), 1),
            latency_max_ms=round(max(latencies), 1),
            peak_python_mib=round(peak / (1024 * 1024), 1),
            rows_copied=rows_copied,
            bytes_copied=bytes_copied,
            statements=statements,
            batched_statements=batched,
        )
    finally:
        _purge(sync, owner_id)


def _purge(sync: sessionmaker, owner_id: UUID) -> None:
    with sync.begin() as session:
        conversations = select(Conversation.id).where(Conversation.owner_id == owner_id)
        for model in (ConversationSummaryJob, ConversationMemorySummary, ConversationCustomAgent):
            session.execute(delete(model).where(model.conversation_id.in_(conversations)))
        session.execute(delete(Generation).where(Generation.conversation_id.in_(conversations)))
        session.execute(delete(Message).where(Message.conversation_id.in_(conversations)))
        session.execute(delete(Conversation).where(Conversation.owner_id == owner_id))
        session.execute(delete(User).where(User.id == owner_id))


def _table(results: list[Result]) -> str:
    lines = [
        "| Messages | Source bytes | Latency median (min-max) | Peak Python alloc "
        "| Rows copied | Bytes copied | SQL statements (batched) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.messages:,} | {result.source_bytes / 1048576:.1f} MiB "
            f"| {result.latency_median_ms:,.0f} ms ({result.latency_min_ms:,.0f}-"
            f"{result.latency_max_ms:,.0f}) | {result.peak_python_mib:.1f} MiB "
            f"| {result.rows_copied:,} | {result.bytes_copied / 1048576:.1f} MiB "
            f"| {result.statements} ({result.batched_statements}) |"
        )
    return "\n".join(lines)


async def main(runs: int) -> list[Result]:
    url = _test_url()
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=_TABLES)
    async_engine = create_async_engine(url, poolclass=NullPool)
    counter = StatementCounter(async_engine)
    sync = sessionmaker(bind=engine, expire_on_commit=False)
    async_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    try:
        return [await _measure(size, runs, sync, async_factory, counter) for size in SIZES]
    finally:
        await async_engine.dispose()
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="timed forks per size")
    arguments = parser.parse_args()
    if sys.platform == "win32":
        outcome = asyncio.run(main(arguments.runs), loop_factory=asyncio.SelectorEventLoop)
    else:
        outcome = asyncio.run(main(arguments.runs))
    print(_table(outcome))
    print(json.dumps([asdict(result) for result in outcome], indent=2))
