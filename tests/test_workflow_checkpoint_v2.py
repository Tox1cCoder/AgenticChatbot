"""Per-turn checkpoint identity, and cleanup that deletes only what it owns.

Every new turn gets its own thread, so append reducers stay turn-local and a
new turn can never load an earlier turn's state. Cleanup enumerates exact
thread IDs from durable metadata and validates their shape — deleting by an
unbounded prefix would take out threads belonging to other turns, including
ones still paused on a human decision.
"""

from __future__ import annotations

import pytest

from app.ai.workflow.state import (
    CHECKPOINT_THREAD_PREFIX,
    build_checkpoint_thread_id,
    parse_checkpoint_thread_id,
)
from app.services.checkpoint_retention_service import (
    is_v2_checkpoint_thread_id,
    owned_v2_thread_ids,
)

# ----------------------------------------------------------------------
# per-turn identity
# ----------------------------------------------------------------------


def test_each_turn_in_a_conversation_gets_its_own_thread():
    first = build_checkpoint_thread_id("conversation-1", "message-1")
    second = build_checkpoint_thread_id("conversation-1", "message-2")

    assert first == f"{CHECKPOINT_THREAD_PREFIX}:conversation-1:message-1"
    assert first != second


def test_a_v2_thread_id_round_trips_to_its_parts():
    thread_id = build_checkpoint_thread_id("conversation-1", "message-1")
    assert parse_checkpoint_thread_id(thread_id) == ("conversation-1", "message-1")


@pytest.mark.parametrize(
    "thread_id",
    [
        "conversation-1",
        "routing-v2:conversation-1",
        "routing-v2::message-1",
        "routing-v2:conversation-1:",
        "routing-v1:conversation-1:message-1",
        "routing-v2:conversation-1:message-1:extra",
        "",
    ],
)
def test_a_malformed_or_v1_thread_id_is_rejected(thread_id):
    with pytest.raises(ValueError):
        parse_checkpoint_thread_id(thread_id)
    assert is_v2_checkpoint_thread_id(thread_id) is False


def test_a_conversation_or_turn_id_containing_a_separator_is_rejected():
    """Otherwise two different turns could build the same thread id."""
    with pytest.raises(ValueError):
        build_checkpoint_thread_id("conversation:1", "message-1")
    with pytest.raises(ValueError):
        build_checkpoint_thread_id("conversation-1", "message:1")


# ----------------------------------------------------------------------
# owned-thread enumeration
# ----------------------------------------------------------------------


def test_owned_threads_are_enumerated_from_persisted_turn_ids():
    ids = owned_v2_thread_ids("conversation-1", ["message-1", "message-2"])
    assert ids == [
        "routing-v2:conversation-1:message-1",
        "routing-v2:conversation-1:message-2",
    ]


def test_owned_threads_skip_unusable_turn_ids_instead_of_guessing():
    ids = owned_v2_thread_ids("conversation-1", ["message-1", "", None, "bad:id"])
    assert ids == ["routing-v2:conversation-1:message-1"]


def test_owned_threads_are_empty_without_a_conversation():
    assert owned_v2_thread_ids("", ["message-1"]) == []
    assert owned_v2_thread_ids(None, ["message-1"]) == []


def test_owned_threads_deduplicate_repeated_turn_ids():
    ids = owned_v2_thread_ids("conversation-1", ["message-1", "message-1"])
    assert ids == ["routing-v2:conversation-1:message-1"]


def test_enumeration_never_produces_a_prefix_wildcard():
    """A prefix delete would reach turns this conversation does not own."""
    ids = owned_v2_thread_ids("conversation-1", ["message-1"])
    for thread_id in ids:
        assert "*" not in thread_id
        assert "%" not in thread_id
        assert is_v2_checkpoint_thread_id(thread_id)
