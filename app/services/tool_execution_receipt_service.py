"""Receipt-backed execution for mutating tool calls.

The gap this closes: a LangGraph checkpoint is written after a node returns,
so a mutation that reached the provider and then lost the process leaves no
record. The replay calls the provider again — a second charge, a second
message sent, a second row created.

The state machine is deliberately small:

* no row          -> reserve, invoke, complete in the same breath as the effect
* ``completed``   -> return the recorded result; do not invoke
* ``failed``      -> the provider never accepted it, so invoking again is safe
* ``reserved``    -> a previous attempt was lost mid-flight
* ``outcome_unknown`` -> already adjudicated as unknowable; never retried

The ``reserved`` case is the only interesting one, and it is where honesty
matters more than convenience. If the provider deduplicates on a key we
supply, retrying under the *same* key is safe and we do it. If it does not,
nobody can say whether the effect happened: retrying risks a duplicate and
reporting failure risks denying a real effect. So the row becomes
``outcome_unknown`` and the caller is told exactly that.

A model can neither supply nor read an execution key. Identity comes from
runtime and config metadata only, because a key the model could choose is a
key it could reuse to replay someone else's effect — or vary to force a
duplicate.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field

from app.models.tool_execution_receipt import ReceiptStatus

logger = logging.getLogger(__name__)

__all__ = [
    "MutationExecutionScope",
    "MutationOutcomeUnknown",
    "NormalizedToolResult",
    "ReceiptRecord",
    "ToolExecutionReceiptService",
    "execution_key",
]

#: Reserved separator for the key preimage. A field cannot contain it, so no
#: combination of field values can be rearranged into a different call's key.
_KEY_SEPARATOR = "\x1f"


class MutationOutcomeUnknown(RuntimeError):
    """The effect may or may not have happened, and nothing can decide it.

    Raised instead of retrying or reporting failure. The execution key travels
    with it so an operator can find the exact row to reconcile.
    """

    def __init__(self, execution_key: str) -> None:
        super().__init__("mutation_outcome_unknown")
        self.execution_key = execution_key


class MutationExecutionScope(BaseModel):
    """The identity of one mutating tool call.

    There is no ``execution_key`` field: the key is derived, never supplied.
    ``provider_idempotency`` describes a provider capability and is
    deliberately *not* part of the identity, so discovering that a provider
    deduplicates does not change which row a call maps to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str = Field(min_length=1, max_length=320)
    dispatch_id: str = Field(min_length=1, max_length=64)
    task_id: str = Field(min_length=1, max_length=160)
    tool_call_id: str = Field(min_length=1, max_length=255)
    tool_id: str = Field(min_length=1, max_length=512)
    user_id: UUID
    conversation_id: UUID
    turn_id: str = Field(min_length=1, max_length=160)
    provider_idempotency: bool = False


def execution_key(scope: MutationExecutionScope) -> str:
    """The stable 64-character identity of one mutating call.

    Derived from the checkpoint thread, the dispatch, the task, and the tool
    call — the four things that together name one call in one turn, and that a
    replay reproduces exactly.
    """
    raw = _KEY_SEPARATOR.join(
        (scope.thread_id, scope.dispatch_id, scope.task_id, scope.tool_call_id)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NormalizedToolResult(BaseModel):
    """What a mutating tool produced, in the shape a replay can return.

    ``provider_receipt_id`` is retained for reconciliation and is stripped from
    anything the model, the stream, or the trace can see: it is a provider-side
    handle, not part of the answer.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    artifact_ref: str | None = None
    provider_receipt_id: str | None = None

    def model_visible_payload(self) -> dict[str, Any]:
        """The subset a model may read. Never includes a provider receipt."""
        payload: dict[str, Any] = {"content": self.content}
        if self.artifact_ref:
            payload["artifact_ref"] = self.artifact_ref
        return payload

    def stored_payload(self) -> dict[str, Any]:
        """The bounded record a replay reads back."""
        return self.model_visible_payload()


class ReceiptRecord(BaseModel):
    """What the repository knows about one execution key right now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_key: str
    status: ReceiptStatus
    fresh: bool = False
    result: dict[str, Any] | None = None
    provider_receipt_id: str | None = None


MutationInvoke = Callable[..., Awaitable[NormalizedToolResult]]


class ToolExecutionReceiptService:
    """Runs one mutating tool call at most once per execution key."""

    def __init__(self, *, repository: Any) -> None:
        self.repository = repository

    async def execute_mutation(
        self, scope: MutationExecutionScope, invoke: MutationInvoke
    ) -> NormalizedToolResult:
        """Reserve, invoke, and record one mutation.

        Insert this after authorization and approval and before the provider
        call: a receipt for a call the user was never allowed to make would
        make the refusal unretryable.
        """
        key = execution_key(scope)
        record = await self.repository.areserve(scope=scope, key=key)

        if not record.fresh:
            decided = self._replay(record, scope, key)
            if decided is not None:
                return decided
            if record.status is ReceiptStatus.RESERVED and not scope.provider_idempotency:
                await self.repository.amark_outcome_unknown(key=key)
                logger.warning(
                    "Mutation %s on %s has an unknown outcome; provider offers no idempotency",
                    key[:12],
                    scope.tool_id,
                )
                raise MutationOutcomeUnknown(key)

        return await self._invoke_and_record(scope, invoke, key)

    def _replay(
        self, record: ReceiptRecord, scope: MutationExecutionScope, key: str
    ) -> NormalizedToolResult | None:
        """Answer from the recorded outcome, when there is one to answer from."""
        if record.status is ReceiptStatus.COMPLETED:
            stored = record.result or {}
            return NormalizedToolResult(
                content=str(stored.get("content") or ""),
                artifact_ref=stored.get("artifact_ref"),
                provider_receipt_id=record.provider_receipt_id,
            )
        if record.status is ReceiptStatus.OUTCOME_UNKNOWN:
            raise MutationOutcomeUnknown(key)
        # ``failed`` means the provider never accepted the call, so the effect
        # did not happen and re-running is the correct behavior.
        return None

    async def _invoke_and_record(
        self, scope: MutationExecutionScope, invoke: MutationInvoke, key: str
    ) -> NormalizedToolResult:
        try:
            result = await self._call(invoke, key=key, scope=scope)
        except GraphBubbleUp:
            # Control flow, not an outcome. The receipt stays reserved so the
            # resumed turn recognizes the call it already started.
            raise
        except MutationOutcomeUnknown:
            await self.repository.amark_outcome_unknown(key=key)
            raise
        except Exception:
            await self.repository.afail(key=key, error_code="tool_execution_failed")
            raise

        await self.repository.acomplete(
            key=key,
            result=result.stored_payload(),
            provider_receipt_id=result.provider_receipt_id,
        )
        return result

    @staticmethod
    async def _call(
        invoke: MutationInvoke, *, key: str, scope: MutationExecutionScope
    ) -> NormalizedToolResult:
        """Invoke the provider, passing the key only where it deduplicates.

        A provider that ignores the key would silently accept a duplicate, so
        the key is offered only to an adapter that declares it honors one.
        """
        if scope.provider_idempotency and _accepts_idempotency_key(invoke):
            return await invoke(idempotency_key=key)
        return await invoke()


def _accepts_idempotency_key(invoke: MutationInvoke) -> bool:
    try:
        signature = inspect.signature(invoke)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    parameters = signature.parameters
    if "idempotency_key" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
