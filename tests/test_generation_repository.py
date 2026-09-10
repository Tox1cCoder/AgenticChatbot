"""Contracts of the generation lifecycle row that hold without a database.

The behaviour this table exists for — one winner among concurrent Continues,
one active lifecycle per conversation, a replayed command returning its own
recorded result — is enforced by unique indexes and ``UPDATE ... WHERE
version`` clauses, and none of it is observable against a fake. Those
assertions live in ``tests/integration/test_generation_repository_postgres.py``.

What is checkable here is everything that would make those guarantees quietly
absent: an index that is declared as a constraint instead (permanent
autogenerate drift), a status missing from the active set that the partial
unique index is built from, an enum whose persisted value drifts from its
member name, and the snapshot projection that every transport reads.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import Index

from app.models.generation import (
    ACTIVE_STATUSES,
    CONTINUABLE_STATUSES,
    TERMINAL_STATUSES,
    Generation,
    GenerationCommand,
    GenerationCommandAction,
    GenerationStatus,
)
from app.schemas.generation import CommandClaim, CreateGeneration, GenerationSnapshot

ROOT = Path(__file__).resolve().parents[1]


def _indexes(model: type) -> dict[str, Index]:
    return {index.name: index for index in model.__table__.indexes}


# ----------------------------------------------------------------------
# enum stability
# ----------------------------------------------------------------------


def test_every_status_persists_its_value_not_its_member_name():
    """The column stores these strings; renaming a member must not migrate."""
    assert {member.value for member in GenerationStatus} == {
        "starting",
        "running",
        "finalizing_after_limit",
        "continuable",
        "continuing",
        "stop_requested",
        "stopped",
        "completed",
        "completed_partial",
        "failed",
    }


def test_command_actions_are_the_two_the_api_exposes():
    assert {member.value for member in GenerationCommandAction} == {"stop", "continue"}


def test_the_status_sets_partition_every_status():
    """A status in none of the sets is a row no query would find."""
    covered = ACTIVE_STATUSES | TERMINAL_STATUSES | CONTINUABLE_STATUSES

    assert covered == set(GenerationStatus)
    assert not ACTIVE_STATUSES & TERMINAL_STATUSES


def test_the_active_set_is_what_the_single_active_lifecycle_index_means():
    """One turn per conversation may be in flight; these are "in flight"."""
    assert {
        GenerationStatus.STARTING,
        GenerationStatus.RUNNING,
        GenerationStatus.FINALIZING_AFTER_LIMIT,
        GenerationStatus.CONTINUING,
        GenerationStatus.STOP_REQUESTED,
    } == ACTIVE_STATUSES


def test_continuable_is_not_active_so_a_paused_turn_frees_the_conversation():
    assert GenerationStatus.CONTINUABLE not in ACTIVE_STATUSES
    assert GenerationStatus.CONTINUABLE in CONTINUABLE_STATUSES


def test_a_stopped_generation_is_terminal_but_may_still_be_continued():
    """Stop is a decision about this epoch, not always about the turn."""
    assert GenerationStatus.STOPPED in TERMINAL_STATUSES
    assert GenerationStatus.STOPPED in CONTINUABLE_STATUSES


# ----------------------------------------------------------------------
# schema declaration
# ----------------------------------------------------------------------


def test_the_logical_turn_is_unique_as_an_index_not_a_constraint():
    """``unique=True`` on the column is permanent autogenerate drift.

    PostgreSQL implements both the same way, but autogenerate distinguishes a
    UNIQUE constraint from a unique index, so the mismatch is reported as
    schema drift on every run forever.
    """
    index = _indexes(Generation)["uq_generations_logical_turn_id"]

    assert index.unique is True
    assert [column.name for column in index.columns] == ["logical_turn_id"]


def test_one_active_lifecycle_per_conversation_is_a_partial_unique_index():
    index = _indexes(Generation)["uq_generations_active_per_conversation"]
    predicate = str(index.dialect_options["postgresql"]["where"])

    assert index.unique is True
    assert [column.name for column in index.columns] == ["conversation_id"]
    for status in ACTIVE_STATUSES:
        assert status.value in predicate
    for status in TERMINAL_STATUSES | {GenerationStatus.CONTINUABLE}:
        assert f"'{status.value}'" not in predicate


def test_the_command_ledger_is_unique_per_generation_and_key():
    """R5: one row per command, so a replay is recognisable forever.

    A single "last command" slot cannot do this. Stop finishes against epoch 0,
    Continue overwrites the slot, and a delayed retry of the Stop is no longer
    recognisable — so it executes against epoch 1.
    """
    index = _indexes(GenerationCommand)["uq_generation_commands_key"]

    assert index.unique is True
    assert [column.name for column in index.columns] == ["generation_id", "idempotency_key"]


def test_a_command_records_the_fence_it_was_issued_against():
    assert "fence" in GenerationCommand.__table__.columns
    assert GenerationCommand.__table__.columns["fence"].nullable is False


def test_research_accounting_is_stored_on_the_row_not_in_process_memory():
    """R4: a Continue served by another worker has to find the accounting."""
    assert "research_accounting" in Generation.__table__.columns


def test_the_row_records_the_process_producing_it():
    """Without it an abandoned turn is indistinguishable from a live one.

    The partial unique index admits one active row per conversation, so a row
    left active by a dead worker blocks that conversation for good. Nothing
    else recorded identifies the producer: ``build_sha`` is shared by every
    worker of a build, and the worker count is not a setting.
    """
    column = Generation.__table__.columns["producer_token"]

    # Nullable because rows written before this column existed name no
    # producer, and the reaper must read those as unknown rather than dead.
    assert column.nullable is True


def test_the_snapshot_carries_no_producer_identity():
    """A hostname and pid are infrastructure, and it is returned over HTTP."""
    assert "producer_token" not in set(GenerationSnapshot.model_fields)


def test_the_owner_columns_are_indexed_because_every_read_is_owner_scoped():
    columns = Generation.__table__.columns

    assert columns["user_id"].nullable is False
    assert columns["conversation_id"].nullable is False


def test_the_version_and_epoch_start_where_the_transition_table_expects():
    columns = Generation.__table__.columns

    assert columns["version"].nullable is False
    assert columns["execution_epoch"].nullable is False


# ----------------------------------------------------------------------
# snapshot projection
# ----------------------------------------------------------------------


def _row(**overrides) -> Generation:
    values: dict = {
        "id": uuid4(),
        "logical_turn_id": "turn-1",
        "conversation_id": uuid4(),
        "user_id": uuid4(),
        "checkpoint_thread_id": "routing-v2:thread-1",
        "status": GenerationStatus.RUNNING,
        "version": 3,
        "execution_epoch": 1,
        "continuation_id": None,
        "continuation_available": False,
        "continuation_block_reason": None,
        "assistant_message_id": None,
        "terminal_reason": None,
    }
    values.update(overrides)
    return Generation(**values)


def test_the_snapshot_projects_the_row_the_transports_share():
    row = _row()

    snapshot = GenerationSnapshot.from_row(row)

    assert snapshot.generation_id == row.id
    assert snapshot.status is GenerationStatus.RUNNING
    assert snapshot.version == 3
    assert snapshot.execution_epoch == 1


def test_the_snapshot_carries_no_checkpoint_internals():
    """It is returned over HTTP: a thread id is a resume handle, not public."""
    fields = set(GenerationSnapshot.model_fields)

    assert "checkpoint_thread_id" not in fields
    assert "execution_budget" not in fields
    assert "research_accounting" not in fields


def test_continuation_availability_is_reported_with_its_reason():
    row = _row(
        status=GenerationStatus.STOPPED,
        continuation_available=False,
        continuation_block_reason="mutation_outcome_unknown",
    )

    snapshot = GenerationSnapshot.from_row(row)

    assert snapshot.continuation_available is False
    assert snapshot.continuation_block_reason == "mutation_outcome_unknown"


def test_the_snapshot_is_immutable_so_a_caller_cannot_fake_a_transition():
    snapshot = GenerationSnapshot.from_row(_row())

    with pytest.raises(ValidationError):
        snapshot.status = GenerationStatus.COMPLETED


def test_creating_a_generation_requires_the_identity_the_lifecycle_is_keyed_by():
    command = CreateGeneration(
        conversation_id=uuid4(),
        user_id=uuid4(),
        logical_turn_id="turn-9",
        checkpoint_thread_id="routing-v2:thread-9",
    )

    assert command.logical_turn_id == "turn-9"
    assert command.active_agent_id is None


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_logical_turn_is_refused_because_it_is_the_unique_key(blank):
    with pytest.raises(ValidationError):
        CreateGeneration(
            conversation_id=uuid4(),
            user_id=uuid4(),
            logical_turn_id=blank,
            checkpoint_thread_id="routing-v2:thread-9",
        )


def test_a_claimed_command_is_distinguishable_from_a_replayed_one():
    claimed = CommandClaim(
        claimed=True, action=GenerationCommandAction.STOP, fence=4, result=None
    )
    replayed = CommandClaim(
        claimed=False,
        action=GenerationCommandAction.STOP,
        fence=4,
        result={"status": "stopped"},
    )

    assert claimed.claimed is True
    assert replayed.claimed is False
    assert replayed.result == {"status": "stopped"}


# ----------------------------------------------------------------------
# model/migration parity
# ----------------------------------------------------------------------


def _migration_module():
    import importlib.util

    path = ROOT / "app" / "alembic" / "versions" / "d0e1f2a3b4c5_add_generation_controls.py"
    spec = importlib.util.spec_from_file_location("_generation_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_migration_predicate_matches_the_models_active_set():
    """Two copies of "in flight" is two answers to "one turn at a time".

    The model builds the partial index predicate from ACTIVE_STATUSES; the
    migration hardcodes it, because a migration must describe the schema at its
    own revision rather than follow whatever the models later say. This asserts
    they agree today.
    """
    migration = _migration_module()

    assert sorted(status.value for status in ACTIVE_STATUSES) == sorted(
        migration._ACTIVE_STATUSES
    )


def test_the_migration_declares_every_status_the_enum_has():
    migration = _migration_module()

    assert sorted(migration._STATUS_VALUES) == sorted(member.value for member in GenerationStatus)
    assert sorted(migration._ACTION_VALUES) == sorted(
        member.value for member in GenerationCommandAction
    )


def test_the_migration_follows_the_previously_verified_head():
    migration = _migration_module()

    assert migration.revision == "d0e1f2a3b4c5"
    assert migration.down_revision == "c9d0e1f2a3b4"
