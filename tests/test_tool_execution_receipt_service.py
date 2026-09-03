"""Durable receipts for mutating tool calls.

A checkpoint is written *after* a node returns. A mutation that succeeded and
then lost the process therefore leaves no record, and the replay calls the
provider again — a second charge, a second message sent, a second row created.
A receipt closes that gap by being reserved *before* the call and completed in
the same breath as the effect.

What a receipt cannot do is invent exactly-once behavior a provider does not
offer. A reserved receipt with no completion means the outcome is genuinely
unknown, and saying so is the only honest answer: retrying would risk a
duplicate, and reporting failure would risk denying an effect that happened.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.models.tool_execution_receipt import ReceiptStatus
from app.services.tool_execution_receipt_service import (
    MutationExecutionScope,
    MutationOutcomeUnknown,
    NormalizedToolResult,
    ReceiptRecord,
    ToolExecutionReceiptService,
    execution_key,
)

USER_ID = UUID("11111111-1111-1111-1111-111111111111")
CONVERSATION_ID = UUID("22222222-2222-2222-2222-222222222222")
OTHER_USER_ID = UUID("33333333-3333-3333-3333-333333333333")


def _scope(**overrides) -> MutationExecutionScope:
    payload = {
        "thread_id": "routing-v2:conversation-1:turn-1",
        "dispatch_id": "d1",
        "task_id": "w1",
        "tool_call_id": "call-1",
        "tool_id": "mcp::write",
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
        "turn_id": "turn-1",
        "provider_idempotency": False,
    }
    payload.update(overrides)
    return MutationExecutionScope(**payload)


class FakeReceiptRepository:
    """In-memory stand-in with the same compare-and-set contract as SQL."""

    def __init__(self):
        self.status: ReceiptStatus | None = None
        self.completed_result: dict | None = None
        self.provider_receipt_id: str | None = None
        self.reserved: list[str] = []
        self.completed: list[tuple[str, dict | None]] = []
        self.failed: list[tuple[str, str]] = []
        self.unknown: list[str] = []
        self.owner_user_id: UUID = USER_ID

    async def areserve(self, *, scope: MutationExecutionScope, key: str) -> ReceiptRecord:
        if self.status is None:
            self.reserved.append(key)
            self.status = ReceiptStatus.RESERVED
            return ReceiptRecord(execution_key=key, status=ReceiptStatus.RESERVED, fresh=True)
        if self.owner_user_id != scope.user_id:
            # Another user's receipt is not this caller's to observe or reuse.
            self.reserved.append(key)
            return ReceiptRecord(execution_key=key, status=ReceiptStatus.RESERVED, fresh=True)
        return ReceiptRecord(
            execution_key=key,
            status=self.status,
            fresh=False,
            result=self.completed_result,
            provider_receipt_id=self.provider_receipt_id,
        )

    async def acomplete(self, *, key: str, result: dict | None, provider_receipt_id: str | None):
        self.completed.append((key, result))
        self.status = ReceiptStatus.COMPLETED
        self.completed_result = result
        self.provider_receipt_id = provider_receipt_id

    async def afail(self, *, key: str, error_code: str) -> None:
        self.failed.append((key, error_code))
        self.status = ReceiptStatus.FAILED

    async def amark_outcome_unknown(self, *, key: str) -> None:
        self.unknown.append(key)
        self.status = ReceiptStatus.OUTCOME_UNKNOWN


def _service(**overrides) -> ToolExecutionReceiptService:
    payload = {"repository": FakeReceiptRepository()}
    payload.update(overrides)
    return ToolExecutionReceiptService(**payload)


# ----------------------------------------------------------------------
# the execution key
# ----------------------------------------------------------------------


def test_execution_key_is_stable():
    scope = _scope()
    assert execution_key(scope) == execution_key(scope)
    assert len(execution_key(scope)) == 64


def test_execution_key_is_derived_only_from_call_identity():
    """Provider capability is not identity: the same call keeps the same key."""
    base = _scope()
    assert execution_key(base) == execution_key(_scope(provider_idempotency=True))
    assert execution_key(base) != execution_key(_scope(tool_call_id="call-2"))
    assert execution_key(base) != execution_key(_scope(task_id="w2"))
    assert execution_key(base) != execution_key(_scope(dispatch_id="d2"))
    assert execution_key(base) != execution_key(_scope(thread_id="routing-v2:c:other"))


def test_execution_key_separator_cannot_be_forged_from_field_content():
    """Concatenation without a reserved separator would collide."""
    left = _scope(dispatch_id="d", task_id="1:w")
    right = _scope(dispatch_id="d:1", task_id="w")
    assert execution_key(left) != execution_key(right)


def test_a_model_cannot_supply_or_read_an_execution_key():
    from pydantic import ValidationError

    assert "execution_key" not in MutationExecutionScope.model_fields
    with pytest.raises(ValidationError):
        MutationExecutionScope.model_validate({**_scope().model_dump(), "execution_key": "forged"})


# ----------------------------------------------------------------------
# the state machine
# ----------------------------------------------------------------------


async def test_a_first_call_reserves_then_completes():
    service = _service()
    invoke = AsyncMock(return_value=NormalizedToolResult(content="created"))

    result = await service.execute_mutation(_scope(), invoke)

    assert result.content == "created"
    invoke.assert_awaited_once()
    key = execution_key(_scope())
    assert service.repository.reserved == [key]
    assert service.repository.completed == [(key, {"content": "created"})]


async def test_completed_receipt_returns_recorded_result():
    service = _service()
    service.repository.status = ReceiptStatus.COMPLETED
    service.repository.completed_result = {"content": "created", "artifact_ref": "blob:1"}
    invoke = AsyncMock()

    result = await service.execute_mutation(_scope(), invoke)

    assert result.content == "created"
    assert result.artifact_ref == "blob:1"
    invoke.assert_not_awaited()


async def test_failed_receipt_is_retried_because_nothing_happened():
    """A recorded failure means the provider never accepted the call."""
    service = _service()
    service.repository.status = ReceiptStatus.FAILED
    invoke = AsyncMock(return_value=NormalizedToolResult(content="created"))

    result = await service.execute_mutation(_scope(), invoke)

    assert result.content == "created"
    invoke.assert_awaited_once()


async def test_reserved_non_idempotent_call_becomes_unknown():
    service = _service()
    service.repository.status = ReceiptStatus.RESERVED
    invoke = AsyncMock()

    with pytest.raises(MutationOutcomeUnknown) as excinfo:
        await service.execute_mutation(_scope(provider_idempotency=False), invoke)

    invoke.assert_not_awaited()
    assert excinfo.value.execution_key == execution_key(_scope())
    assert service.repository.unknown == [execution_key(_scope())]


async def test_reserved_idempotent_call_retries_under_the_same_key():
    """Only a provider that deduplicates on our key may be called twice."""
    service = _service()
    service.repository.status = ReceiptStatus.RESERVED
    seen_keys: list[str] = []

    async def invoke(*, idempotency_key: str) -> NormalizedToolResult:
        seen_keys.append(idempotency_key)
        return NormalizedToolResult(content="created", provider_receipt_id="prov-1")

    result = await service.execute_mutation(_scope(provider_idempotency=True), invoke)

    assert seen_keys == [execution_key(_scope())]
    assert result.provider_receipt_id == "prov-1"
    assert service.repository.status is ReceiptStatus.COMPLETED


async def test_outcome_unknown_receipt_is_never_retried():
    service = _service()
    service.repository.status = ReceiptStatus.OUTCOME_UNKNOWN
    invoke = AsyncMock()

    with pytest.raises(MutationOutcomeUnknown):
        await service.execute_mutation(_scope(), invoke)

    invoke.assert_not_awaited()


async def test_a_raising_invocation_records_failure_and_propagates():
    service = _service()

    async def invoke() -> NormalizedToolResult:
        raise RuntimeError("provider refused")

    with pytest.raises(RuntimeError, match="provider refused"):
        await service.execute_mutation(_scope(), invoke)

    key = execution_key(_scope())
    assert service.repository.failed == [(key, "tool_execution_failed")]


async def test_control_flow_exceptions_leave_the_receipt_reserved():
    """An approval pause is not a failure and must not close the receipt."""
    from langgraph.errors import GraphBubbleUp

    service = _service()

    async def invoke() -> NormalizedToolResult:
        raise GraphBubbleUp("paused")

    with pytest.raises(GraphBubbleUp):
        await service.execute_mutation(_scope(), invoke)

    assert service.repository.failed == []
    assert service.repository.unknown == []
    assert service.repository.status is ReceiptStatus.RESERVED


async def test_another_users_receipt_is_not_reused():
    service = _service()
    service.repository.status = ReceiptStatus.COMPLETED
    service.repository.completed_result = {"content": "someone elses effect"}
    service.repository.owner_user_id = OTHER_USER_ID
    invoke = AsyncMock(return_value=NormalizedToolResult(content="mine"))

    result = await service.execute_mutation(_scope(), invoke)

    assert result.content == "mine"
    invoke.assert_awaited_once()


# ----------------------------------------------------------------------
# nothing leaks
# ----------------------------------------------------------------------


def test_the_normalized_result_carries_no_key():
    assert "execution_key" not in NormalizedToolResult.model_fields
    result = NormalizedToolResult(content="created", provider_receipt_id="prov-1")
    assert "execution_key" not in result.model_dump()


def test_model_visible_payload_strips_key_and_provider_receipt():
    result = NormalizedToolResult(
        content="created", provider_receipt_id="prov-1", artifact_ref="blob:1"
    )
    payload = result.model_visible_payload()

    assert payload == {"content": "created", "artifact_ref": "blob:1"}
    assert "provider_receipt_id" not in payload


def test_receipt_keys_are_absent_from_the_streaming_and_rendering_surfaces():
    """A key or provider receipt reaching a stream is a leak, not a detail."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    surfaces = [
        *sorted((repo_root / "app" / "services" / "event_streaming").rglob("*.py")),
        repo_root / "app" / "ai" / "tool_result_rendering.py",
    ]
    for path in surfaces:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        assert "execution_key" not in source, path
        assert "provider_receipt" not in source, path
