"""The authoritative lifecycle of one assistant generation.

A turn used to be identified by the message it produced, which is why Stop was
unreliable: the message ID exists only after the answer does, so there was
nothing to cancel during the part of the turn that takes the longest. This
table owns identity instead. It is created before the first model call and
outlives every worker that touches it.

Three properties are database guarantees rather than application checks,
because the operations they protect run concurrently across workers:

* ``version`` is the compare-and-set fence. Every transition is one
  ``UPDATE ... WHERE version = :expected``, so two Continues racing on the same
  paused turn produce exactly one epoch increment.
* one active lifecycle per conversation, enforced by a partial unique index
  over :data:`ACTIVE_STATUSES`. ``continuable`` is deliberately outside that
  set: a paused turn holds no worker and must not block the next question.
* one row per ``(generation_id, idempotency_key)`` in
  :class:`GenerationCommand`, so a command that arrives twice returns its own
  recorded result. A single "last command" column cannot: Stop completes
  against epoch 0, Continue overwrites the column, and a delayed retry of the
  Stop is no longer recognisable — so it cancels epoch 1 instead.
"""

import enum
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.base import Base

__all__ = [
    "ACTIVE_STATUSES",
    "CONTINUABLE_STATUSES",
    "TERMINAL_STATUSES",
    "Generation",
    "GenerationCommand",
    "GenerationCommandAction",
    "GenerationStatus",
]


class GenerationStatus(str, enum.Enum):
    """Where one generation stands. The only authority on that question."""

    STARTING = "starting"
    RUNNING = "running"
    FINALIZING_AFTER_LIMIT = "finalizing_after_limit"
    CONTINUABLE = "continuable"
    CONTINUING = "continuing"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"


class GenerationCommandAction(str, enum.Enum):
    """The two commands a client may issue against a live generation."""

    STOP = "stop"
    CONTINUE = "continue"


#: A worker is doing work, or is being asked to stop doing it. The partial
#: unique index is built from exactly this set, so adding a status here changes
#: what "one turn at a time" means.
ACTIVE_STATUSES = frozenset(
    {
        GenerationStatus.STARTING,
        GenerationStatus.RUNNING,
        GenerationStatus.FINALIZING_AFTER_LIMIT,
        GenerationStatus.CONTINUING,
        GenerationStatus.STOP_REQUESTED,
    }
)

#: Nothing further happens on its own. Some of these can still be continued.
TERMINAL_STATUSES = frozenset(
    {
        GenerationStatus.STOPPED,
        GenerationStatus.COMPLETED,
        GenerationStatus.COMPLETED_PARTIAL,
        GenerationStatus.FAILED,
    }
)

#: A Continue may be accepted from here, subject to ``continuation_available``.
#: ``stopped`` appears in both this and :data:`TERMINAL_STATUSES` on purpose:
#: Stop ends an epoch, and whether it ended the turn is a separate decision.
CONTINUABLE_STATUSES = frozenset(
    {
        GenerationStatus.CONTINUABLE,
        GenerationStatus.STOPPED,
    }
)

_ACTIVE_STATUS_SQL = ", ".join(
    f"'{status.value}'" for status in sorted(ACTIVE_STATUSES, key=lambda item: item.value)
)


def _enum_member_values(enum_class: type[enum.Enum]) -> list[str]:
    """Persist stable enum values instead of Python member names."""
    return [str(member.value) for member in enum_class]


class Generation(Base):
    """One assistant generation, across however many epochs it takes."""

    __tablename__ = "generations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    terminal_at = Column(DateTime(timezone=True), nullable=True)

    # --- identity --------------------------------------------------------
    # ``logical_turn_id`` survives Continue; ``id`` does too. What increments
    # is ``execution_epoch``. Declared as a unique *index* rather than
    # ``unique=True``: autogenerate distinguishes a UNIQUE constraint from a
    # unique index, and the mismatch reads as schema drift forever.
    logical_turn_id = Column(String(160), nullable=False)
    checkpoint_thread_id = Column(String(320), nullable=False)

    # --- producer --------------------------------------------------------
    # ``hostname:pid:started_at`` for the worker streaming this turn, so a
    # later process can tell an abandoned turn from a live one. The partial
    # unique index below admits one active row per conversation, so a row left
    # active by a dead worker blocks that conversation until something
    # terminalizes it, and nothing else recorded identifies the producer:
    # ``build_sha`` is shared by every worker of a build, and the worker count
    # belongs to the launch command rather than to any setting.
    #
    # Nullable on purpose. Rows written before this column existed name no
    # producer, and the reaper reads those as unknown rather than as dead.
    producer_token = Column(String(255), nullable=True)

    # --- owners ----------------------------------------------------------
    # Every read is filtered by both. A generation is not addressable by id
    # alone; that would let one user probe another's conversations.
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )

    # --- lifecycle -------------------------------------------------------
    status = Column(
        SQLEnum(
            GenerationStatus,
            name="generation_status",
            create_type=True,
            values_callable=_enum_member_values,
        ),
        nullable=False,
    )
    version = Column(Integer, nullable=False, default=1, server_default=text("1"))
    execution_epoch = Column(Integer, nullable=False, default=0, server_default=text("0"))
    active_agent_id = Column(String(160), nullable=True)
    terminal_reason = Column(String(128), nullable=True)

    # --- carried accounting ----------------------------------------------
    # Both are bounded JSON snapshots, not transcripts. ``research_accounting``
    # lives here because a Continue may be served by a different worker than
    # the epoch it continues, and an in-process budget would look to that
    # worker exactly like a fresh turn with a full quota.
    execution_budget = Column(JSONB, nullable=True)
    research_accounting = Column(JSONB, nullable=True)

    # --- continuation ----------------------------------------------------
    # The continuation ID is opaque and single-use: it is what a client
    # presents to continue, and rotating it is what makes a consumed
    # continuation unusable without consulting the status alone.
    # ``SET NULL``, not the default: a generation row is bookkeeping *about* a
    # turn and must not be able to veto deletion of the message the turn
    # produced. ``CASCADE`` would take the opposite wrong turn and destroy the
    # lifecycle history. Same choice, for the same reason, as
    # ``model_usage_events.request_message_id``.
    assistant_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "messages.id",
            name="fk_generations_assistant_message",
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    continuation_id = Column(UUID(as_uuid=True), nullable=True)
    continuation_available = Column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    continuation_block_reason = Column(String(128), nullable=True)

    __table_args__ = (
        Index("uq_generations_logical_turn_id", "logical_turn_id", unique=True),
        # One turn in flight per conversation. ``continuable`` is excluded, so a
        # paused turn does not block the next question while it waits.
        Index(
            "uq_generations_active_per_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text(f"status IN ({_ACTIVE_STATUS_SQL})"),
        ),
        Index("ix_generations_owner_status", "user_id", "status"),
        Index("ix_generations_checkpoint_thread", "checkpoint_thread_id"),
        Index("ix_generations_continuation_id", "continuation_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<Generation(id={self.id}, turn='{self.logical_turn_id}', "
            f"status={self.status.value if self.status else None}, "
            f"epoch={self.execution_epoch}, version={self.version})>"
        )


class GenerationCommand(Base):
    """One Stop or Continue, recorded with the fence it was issued against.

    The ledger is the idempotency mechanism, not a log of one. A caller claims
    a command by inserting; the unique index decides the winner, and the loser
    reads the recorded result instead of acting. ``fence`` is the lifecycle
    version the client believed it was addressing, which is what lets a delayed
    replay be refused as stale rather than applied to a later epoch.
    """

    __tablename__ = "generation_commands"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    generation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idempotency_key = Column(String(160), nullable=False)
    action = Column(
        SQLEnum(
            GenerationCommandAction,
            name="generation_command_action",
            create_type=True,
            values_callable=_enum_member_values,
        ),
        nullable=False,
    )
    fence = Column(Integer, nullable=False)
    result = Column(JSONB, nullable=True)

    __table_args__ = (
        Index(
            "uq_generation_commands_key",
            "generation_id",
            "idempotency_key",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<GenerationCommand(generation={self.generation_id}, "
            f"action={self.action.value if self.action else None}, fence={self.fence})>"
        )
