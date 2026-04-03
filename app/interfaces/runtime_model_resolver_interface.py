from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from app.core.runtime_modeling import ResolvedRuntimeModelConfig


class IRuntimeModelResolver(ABC):
    """Resolve runtime model configuration for an agent invocation."""

    @abstractmethod
    def resolve_runtime_config(
        self,
        user_id: UUID | None,
        agent_key: str,
        request_override: Mapping[str, Any] | None = None,
    ) -> ResolvedRuntimeModelConfig:
        """Return the effective runtime model configuration for the request."""
        pass
