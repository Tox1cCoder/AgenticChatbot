from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeFallbackConfig:
    provider: str
    model: str
    temperature: float
    api_key: str
    key_source: str


@dataclass
class ResolvedRuntimeModelConfig:
    agent_key: str
    provider: str
    model: str
    temperature: float
    api_key: str | None
    key_source: str
    source: str
    warnings: list[str] = field(default_factory=list)
    is_custom_model: bool = False
    capabilities: dict[str, bool] = field(default_factory=dict)
    provider_fallback: dict[str, Any] | None = None
    fallback_config: RuntimeFallbackConfig | None = None
    reasoning_effort: str | None = None
    context_window: dict[str, Any] | None = None


class StrictRuntimeResolutionError(RuntimeError):
    """Raised when strict runtime resolution cannot honor the request.

    Strict resolution (``allow_provider_fallback=False``) never substitutes a
    provider, model, or credential. Callers translate ``reason`` into their own
    typed error; they never parse the message text.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail
