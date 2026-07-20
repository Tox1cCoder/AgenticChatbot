"""Domain types for per-user model-usage analytics.

These types are immutable and carry no I/O — they give the usage-tracking
pipeline (recorder, repository, instrumentation, built in later tasks) a
single, well-typed vocabulary to share.
"""

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Literal
from uuid import UUID, uuid4

UsageStatus = Literal["success", "error", "cancelled", "timeout"]
UsageSource = Literal[
    "provider_reported",
    "mixed_reported_estimated",
    "locally_estimated",
    "unavailable",
]

# Fields validated by NormalizedUsage.__post_init__ that are `int | None`:
# None means "unknown" (the provider didn't report it) and is accepted.
# `generated_images` is NOT in this tuple — it's declared `int = 0`, never
# Optional, so None is rejected for it (see __post_init__).
_OPTIONAL_NUMERIC_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "input_text_tokens",
    "input_image_tokens",
    "output_text_tokens",
    "output_image_tokens",
)


def _require_non_negative_int(field_name: str, value: object) -> None:
    """Raise unless ``value`` is a non-boolean int that is >= 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"NormalizedUsage.{field_name} must be a non-negative int, got {value!r}")
    if value < 0:
        raise ValueError(f"NormalizedUsage.{field_name} must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class UsageContext:
    """Identifiers that locate a model call within the app's request graph.

    Attributes:
        user_id: The user the call was made on behalf of.
        conversation_id: The conversation the call belongs to.
        request_message_id: The inbound message that triggered the call.
        document_id: The document being processed, if any.
        correlation_id: Cross-service trace identifier.
        langsmith_run_id: LangSmith run identifier, if tracing is enabled.
        operation: Human-readable name of the operation (e.g. "chat", "rag").
        agent_id: The agent handling the call, if applicable.
    """

    user_id: UUID | None = None
    conversation_id: UUID | None = None
    request_message_id: UUID | None = None
    document_id: UUID | None = None
    correlation_id: str | None = None
    langsmith_run_id: UUID | None = None
    operation: str = "unknown"
    agent_id: str | None = None

    def child(self, **changes: Any) -> "UsageContext":
        """Derive a new context that overrides only the given fields."""
        return replace(self, **changes)


@dataclass
class UsageOperation:
    """A single logical model-call operation, shared across its retry attempts.

    Attempt numbers are allocated under a lock so concurrent retries (e.g.
    from provider-fallback code paths) never receive the same attempt number.
    """

    operation_id: UUID = field(default_factory=uuid4)
    _next_attempt: int = field(default=1, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def allocate_attempt(self) -> int:
        """Return the next attempt number for this operation, thread-safely."""
        with self._lock:
            attempt = self._next_attempt
            self._next_attempt += 1
            return attempt


@dataclass(frozen=True)
class NormalizedUsage:
    """Token/image usage normalized to a common shape across providers.

    A field of ``None`` means "unknown" (the provider didn't report it) and
    must be kept distinct from ``0`` (the provider reported zero usage).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    output_text_tokens: int | None = None
    output_image_tokens: int | None = None
    generated_images: int = 0
    source: UsageSource = "unavailable"

    def __post_init__(self) -> None:
        for field_name in _OPTIONAL_NUMERIC_USAGE_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                continue
            _require_non_negative_int(field_name, value)
        # generated_images is declared `int = 0`, not Optional, so unlike the
        # fields above, None is not a valid "unknown" for it.
        _require_non_negative_int("generated_images", self.generated_images)
