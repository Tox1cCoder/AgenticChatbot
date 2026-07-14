"""Pure lifecycle recovery policy for HITL user interfaces."""

import pytest

from app.ui.hitl_recovery import (
    extract_error_code,
    is_recoverable_resume_conflict,
    reconciliation_action,
    should_suppress_pending_interrupt,
)


def test_extracts_codes_from_stream_and_fastapi_error_envelopes():
    assert extract_error_code({"errorCode": "INTERRUPT_CONFLICT"}) == "INTERRUPT_CONFLICT"
    assert extract_error_code({"detail": {"code": "INTERRUPT_CONFLICT"}}) == "INTERRUPT_CONFLICT"


def test_classifies_recoverable_resume_conflicts():
    assert is_recoverable_resume_conflict({"error_code": "INTERRUPT_ALREADY_RESOLVED"})
    assert is_recoverable_resume_conflict({"detail": {"code": "INTERRUPT_CONFLICT"}})
    assert not is_recoverable_resume_conflict({"code": "INTERRUPT_FAILED"})


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pending", "restore_form"),
        ("resolving", "show_processing"),
        ("resolved", "refresh_history"),
        ("failed", "require_new_message"),
        ("expired", "require_new_message"),
        (None, "require_new_message"),
    ],
)
def test_classifies_every_lifecycle_state(status, expected):
    assert reconciliation_action(status) == expected


def test_suppresses_only_the_matching_pending_interrupt():
    assert should_suppress_pending_interrupt("int-1", "int-1")
    assert not should_suppress_pending_interrupt("int-2", "int-1")
    assert not should_suppress_pending_interrupt(None, "int-1")
    assert not should_suppress_pending_interrupt("int-1", None)
