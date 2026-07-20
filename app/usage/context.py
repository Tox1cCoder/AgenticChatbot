"""Async-safe binding for the current UsageContext and UsageOperation.

Values are stored in ContextVars so each asyncio Task (and each thread) sees
its own binding — concurrent requests never observe each other's context.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.usage.types import UsageContext, UsageOperation

# UsageContext is a frozen dataclass, so sharing this single default instance
# across tasks is safe; ruff's B039 guards against defaults that can mutate.
_usage_context: ContextVar[UsageContext] = ContextVar(
    "model_usage_context",
    default=UsageContext(),  # noqa: B039
)
_usage_operation: ContextVar[UsageOperation | None] = ContextVar(
    "model_usage_operation", default=None
)


def current_usage_context() -> UsageContext:
    """Return the UsageContext bound in the current task, or the default."""
    return _usage_context.get()


def current_usage_operation() -> UsageOperation | None:
    """Return the UsageOperation bound in the current task, if any."""
    return _usage_operation.get()


@contextmanager
def bind_usage_context(context: UsageContext) -> Iterator[UsageContext]:
    """Bind ``context`` as current for the duration of the block."""
    token = _usage_context.set(context)
    try:
        yield context
    finally:
        _usage_context.reset(token)


@contextmanager
def begin_usage_operation() -> Iterator[UsageOperation]:
    """Start a new UsageOperation and bind it as current for the block."""
    operation = UsageOperation()
    token = _usage_operation.set(operation)
    try:
        yield operation
    finally:
        _usage_operation.reset(token)
