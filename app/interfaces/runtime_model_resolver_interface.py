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
        *,
        require_capabilities: frozenset[str] = frozenset(),
        allow_provider_fallback: bool = True,
    ) -> ResolvedRuntimeModelConfig:
        """Return the effective runtime model configuration for the request.

        ``require_capabilities`` names capabilities the resolved model must
        advertise; a missing capability raises
        :class:`~app.core.runtime_modeling.StrictRuntimeResolutionError`.

        ``allow_provider_fallback=False`` forbids substituting a different
        provider, model, or credential. The defaults preserve the historic
        behavior for every existing caller; only the router opts out.
        """
        pass
