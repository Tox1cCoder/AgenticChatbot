"""Full-request preflight budgeting and structure-safe emergency reduction."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from app.ai.token_counter import TokenCounter

BudgetAction = Literal[
    "proceed",
    "durable_requested",
    "emergency_compacted",
    "reduced",
    "error",
]


class ContextBudgetExceededError(RuntimeError):
    """Safe, content-free failure raised before an invalid provider request."""

    def __init__(self, code: str) -> None:
        self.code = str(code or "context_budget_exceeded")
        super().__init__(self.code)


@dataclass(frozen=True)
class RequestEnvelope:
    provider: str
    model: str
    system_messages: tuple[Any, ...]
    history_messages: tuple[Any, ...]
    current_messages: tuple[Any, ...]
    tools: tuple[Any, ...] = ()
    attachments: tuple[Any, ...] = ()

    @property
    def messages(self) -> tuple[Any, ...]:
        return self.system_messages + self.history_messages + self.current_messages

    def with_history(self, history: Sequence[Any]) -> RequestEnvelope:
        return replace(self, history_messages=tuple(history))


@dataclass(frozen=True)
class BudgetConfig:
    max_input_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    soft_ratio: float
    hard_ratio: float
    emergency_timeout_seconds: float

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens_must_be_positive")
        if self.reserved_output_tokens < 0 or self.safety_margin_tokens < 0:
            raise ValueError("request_reserves_must_be_non_negative")
        if not 0 < self.soft_ratio < self.hard_ratio < 1:
            raise ValueError("request_budget_ratios_invalid")
        if self.emergency_timeout_seconds <= 0:
            raise ValueError("emergency_timeout_must_be_positive")
        if self.available_input_tokens <= 0:
            raise ValueError("available_input_must_be_positive")

    @property
    def available_input_tokens(self) -> int:
        return self.max_input_tokens - self.reserved_output_tokens - self.safety_margin_tokens


@dataclass(frozen=True)
class BudgetResult:
    action: BudgetAction
    envelope: RequestEnvelope
    input_tokens: int
    available_input_tokens: int
    usage_ratio: float
    count_strategy: str
    durable_requested: bool = False
    emergency_compacted: bool = False
    removed_groups: int = 0
    error_code: str | None = None


class RequestBudgetService:
    """Count the actual request and reduce only removable history structures."""

    def __init__(self, token_counter: TokenCounter) -> None:
        self.token_counter = token_counter

    async def preflight(
        self,
        envelope: RequestEnvelope,
        config: BudgetConfig,
        *,
        durable_request: Callable[[], Any] | None = None,
        emergency_compact: Callable[[tuple[Any, ...]], Sequence[Any] | Awaitable[Sequence[Any]]]
        | None = None,
    ) -> BudgetResult:
        count = self._count(envelope)
        ratio = count.input_tokens / config.available_input_tokens
        if ratio < config.soft_ratio:
            return self._result("proceed", envelope, count, config)

        durable_requested = await self._request_durable(durable_request)
        if ratio < config.hard_ratio:
            return self._result(
                "durable_requested",
                envelope,
                count,
                config,
                durable_requested=durable_requested,
            )

        hard_limit = int(config.available_input_tokens * config.hard_ratio)
        fixed_envelope = envelope.with_history(())
        fixed_count = self._count(fixed_envelope)
        if fixed_count.input_tokens > hard_limit:
            return self._result(
                "error",
                envelope,
                count,
                config,
                durable_requested=durable_requested,
                error_code="context_budget_fixed_input_exceeded",
            )

        candidate = envelope
        if emergency_compact is not None and envelope.history_messages:
            try:
                compacted_history = await asyncio.wait_for(
                    self._maybe_await(emergency_compact(envelope.history_messages)),
                    timeout=config.emergency_timeout_seconds,
                )
                candidate = envelope.with_history(compacted_history)
                compacted_count = self._count(candidate)
                if compacted_count.input_tokens <= hard_limit:
                    return self._result(
                        "emergency_compacted",
                        candidate,
                        compacted_count,
                        config,
                        durable_requested=durable_requested,
                        emergency_compacted=True,
                    )
            except Exception:
                candidate = envelope

        reduced, reduced_count, removed = self._reduce_to_limit(
            candidate,
            hard_limit=hard_limit,
        )
        if reduced_count.input_tokens <= hard_limit:
            return self._result(
                "reduced",
                reduced,
                reduced_count,
                config,
                durable_requested=durable_requested,
                removed_groups=removed,
            )
        return self._result(
            "error",
            reduced,
            reduced_count,
            config,
            durable_requested=durable_requested,
            removed_groups=removed,
            error_code="context_budget_unreducible",
        )

    def _reduce_to_limit(
        self,
        envelope: RequestEnvelope,
        *,
        hard_limit: int,
    ) -> tuple[RequestEnvelope, Any, int]:
        groups = _atomic_history_groups(envelope.history_messages)
        remaining = list(envelope.history_messages)
        removed = 0
        count = self._count(envelope)
        for group in groups:
            if count.input_tokens <= hard_limit:
                break
            if not group.complete:
                break
            del remaining[: len(group.messages)]
            removed += 1
            candidate = envelope.with_history(remaining)
            count = self._count(candidate)
        return envelope.with_history(remaining), count, removed

    def _count(self, envelope: RequestEnvelope):
        return self.token_counter.estimate_request(
            provider=envelope.provider,
            model=envelope.model,
            messages=envelope.messages,
            tools=envelope.tools,
            attachments=envelope.attachments,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        )

    @staticmethod
    async def _request_durable(callback: Callable[[], Any] | None) -> bool:
        if callback is None:
            return False
        try:
            value = callback()
            if inspect.isawaitable(value):
                await value
            return True
        except Exception:
            return False

    @staticmethod
    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _result(
        action: BudgetAction,
        envelope: RequestEnvelope,
        count: Any,
        config: BudgetConfig,
        **kwargs,
    ) -> BudgetResult:
        return BudgetResult(
            action=action,
            envelope=envelope,
            input_tokens=int(count.input_tokens),
            available_input_tokens=config.available_input_tokens,
            usage_ratio=int(count.input_tokens) / config.available_input_tokens,
            count_strategy=str(getattr(count, "strategy", "unknown")),
            **kwargs,
        )


@dataclass(frozen=True)
class _HistoryGroup:
    messages: tuple[Any, ...]
    complete: bool


def _atomic_history_groups(messages: Sequence[Any]) -> list[_HistoryGroup]:
    """Group complete turns without separating tool calls from their results."""
    groups: list[_HistoryGroup] = []
    current: list[Any] = []
    waiting_for_tool_completion = False
    for message in messages:
        role = _role(message)
        if role == "memory" and not current:
            groups.append(_HistoryGroup((message,), True))
            continue
        if role == "user" and current:
            groups.append(_HistoryGroup(tuple(current), _group_is_complete(current)))
            current = []
            waiting_for_tool_completion = False
        current.append(message)
        if role == "assistant":
            if _tool_calls(message):
                waiting_for_tool_completion = True
            else:
                waiting_for_tool_completion = False
                groups.append(_HistoryGroup(tuple(current), True))
                current = []
        elif role == "tool":
            waiting_for_tool_completion = True
    if current:
        groups.append(
            _HistoryGroup(
                tuple(current),
                _group_is_complete(current) and not waiting_for_tool_completion,
            )
        )
    return groups


def _group_is_complete(messages: Sequence[Any]) -> bool:
    return bool(messages) and _role(messages[-1]) == "assistant" and not _tool_calls(messages[-1])


def _role(message: Any) -> str:
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
    role_value = getattr(role, "value", role)
    aliases = {"human": "user", "ai": "assistant"}
    normalized = str(role_value or "").lower()
    return aliases.get(normalized, normalized)


def _tool_calls(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("tool_calls")
    return getattr(message, "tool_calls", None)
