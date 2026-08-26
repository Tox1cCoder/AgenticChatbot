"""Focused middleware for routing-v2 specialist subgraphs.

The framework owns the ReAct loop; these pieces supply only the behavior the
framework cannot know about: which model this user gets, which tools this
device is allowed to call, when a human must approve, and how usage and
artifacts are accounted for.

Each class does one thing and is tested on its own. The order they compose in
is itself a contract — see :func:`build_specialist_middleware`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import ToolMessage

from app.ai.context_overflow import is_context_overflow_error
from app.ai.hitl_config import identity_requires_approval, resolve_call_identity
from app.core.runtime_modeling import ResolvedRuntimeModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactCaptureMiddleware",
    "RequestBudgetMiddleware",
    "RuntimeModelMiddleware",
    "ToolAuthorizationDenied",
    "ToolAuthorizationMiddleware",
    "UsageRecordingMiddleware",
    "build_specialist_middleware",
]

AuthorizeCallable = Callable[..., bool]
PreflightCallable = Callable[[Any, ResolvedRuntimeModelConfig], Awaitable[Any]]


class ToolAuthorizationDenied(RuntimeError):
    """A tool call was refused before the implementation could run."""

    def __init__(self, tool_name: str, reason: str = "") -> None:
        super().__init__(f"{tool_name} is not authorized for this request: {reason}".strip())
        self.tool_name = tool_name
        self.reason = reason


def _tool_call_field(request: Any, field: str, default: Any = None) -> Any:
    tool_call = getattr(request, "tool_call", None)
    if isinstance(tool_call, dict):
        return tool_call.get(field, default)
    return getattr(tool_call, field, default)


class RuntimeModelMiddleware(AgentMiddleware):
    """Resolve the model for this invocation and own provider recovery.

    Resolution happens per invocation because provider, model, and credential
    are user-scoped. Recovery is deliberately narrow: one context-overflow
    retry with compacted messages, then at most one configured fallback
    provider. Nothing here changes the *agent*.
    """

    def __init__(
        self,
        *,
        runtime_model_resolver: Any,
        model_factory: Any,
        agent_key: str,
        user_id: str | None,
        model_request: dict[str, Any] | None,
        compact_messages: Callable[[list[Any]], list[Any]] | None = None,
    ) -> None:
        super().__init__()
        self._resolver = runtime_model_resolver
        self._model_factory = model_factory
        self._agent_key = agent_key
        self._user_id = user_id
        self._model_request = model_request
        self._compact_messages = compact_messages
        self._runtime_config: ResolvedRuntimeModelConfig | None = None

    @property
    def runtime_config(self) -> ResolvedRuntimeModelConfig | None:
        """The config the most recent attempt actually used."""
        return self._runtime_config

    def resolve(self) -> ResolvedRuntimeModelConfig:
        config = self._resolver.resolve_runtime_config(
            self._user_id, self._agent_key, self._model_request
        )
        self._runtime_config = config
        return config

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        config = self.resolve()
        model = self._model_factory.create_model_from_runtime(config)
        attempt = request.override(model=model)

        try:
            return await handler(attempt)
        except Exception as exc:
            if self._compact_messages is not None and is_context_overflow_error(exc):
                compacted = attempt.override(messages=self._compact_messages(attempt.messages))
                return await handler(compacted)

            fallback = getattr(config, "fallback_config", None)
            if fallback is None or not (getattr(fallback, "api_key", "") or "").strip():
                raise

            logger.warning(
                "Specialist %s falling back from %s to %s after a provider error",
                self._agent_key,
                config.provider,
                fallback.provider,
            )
            fallback_config = ResolvedRuntimeModelConfig(
                agent_key=config.agent_key,
                provider=fallback.provider,
                model=fallback.model,
                temperature=fallback.temperature,
                api_key=fallback.api_key,
                key_source=fallback.key_source,
                source="fallback",
                warnings=list(config.warnings),
                capabilities=dict(config.capabilities),
            )
            self._runtime_config = fallback_config
            fallback_model = self._model_factory.create_model_from_runtime(fallback_config)
            return await handler(request.override(model=fallback_model))


class UsageRecordingMiddleware(AgentMiddleware):
    """Record exactly one usage event per provider attempt.

    A failed attempt still consumed the provider's time and the user's quota,
    so it is recorded too. This sits *inside* provider recovery so a fallback
    attempt gets its own record under its own provider/model.
    """

    def __init__(
        self,
        *,
        usage_recorder: Any,
        agent_id: str,
        runtime_config_provider: Callable[[], ResolvedRuntimeModelConfig | None],
    ) -> None:
        super().__init__()
        self._usage_recorder = usage_recorder
        self._agent_id = agent_id
        self._runtime_config_provider = runtime_config_provider

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        if self._usage_recorder is None:
            return await handler(request)

        from app.usage import begin_usage_operation, bind_usage_context, current_usage_context

        config = self._runtime_config_provider()
        provider = getattr(config, "provider", "unknown")
        model = getattr(config, "model", "unknown")

        async def _call() -> Any:
            return await handler(request)

        context = current_usage_context().child(agent_id=self._agent_id)
        with bind_usage_context(context), begin_usage_operation() as operation:
            return await self._usage_recorder.record_one_async_attempt(
                call=_call, provider=provider, model=model, operation=operation
            )


class RequestBudgetMiddleware(AgentMiddleware):
    """Run the token-budget preflight before every model attempt.

    It runs per attempt rather than once per invocation because a fallback
    changes provider, and a budget counted against the provider that failed
    would be the wrong number.
    """

    def __init__(
        self,
        *,
        preflight: PreflightCallable,
        runtime_config_provider: Callable[[], ResolvedRuntimeModelConfig | None],
    ) -> None:
        super().__init__()
        self._preflight = preflight
        self._runtime_config_provider = runtime_config_provider

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        replacement = await self._preflight(request, self._runtime_config_provider())
        if replacement:
            request = request.override(messages=replacement)
        return await handler(request)


class ToolAuthorizationMiddleware(AgentMiddleware):
    """Refuse unauthorized tool calls before the implementation runs.

    A refusal is model-visible feedback, not an exception: the model needs to
    learn that this path is closed so it can choose another. This sits
    outermost in the tool stack so authorization decides before a human is
    ever asked to approve something the caller could not run anyway.
    """

    def __init__(
        self,
        *,
        authorize: AuthorizeCallable,
        user_id: str | None,
        device_id: str | None,
    ) -> None:
        super().__init__()
        self._authorize = authorize
        self._user_id = user_id
        self._device_id = device_id

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_name = str(_tool_call_field(request, "name") or "")
        args = _tool_call_field(request, "args") or {}
        try:
            allowed = self._authorize(
                tool_name, args, user_id=self._user_id, device_id=self._device_id
            )
        except Exception as exc:  # noqa: BLE001 - a failed check must deny, not crash
            logger.warning("Tool authorization check failed for %s: %s", tool_name, exc)
            allowed = False

        if not allowed:
            return ToolMessage(
                content=f"Tool '{tool_name}' is not authorized for this request.",
                tool_call_id=str(_tool_call_field(request, "id") or ""),
                name=tool_name,
                status="error",
            )
        return await handler(request)


class ArtifactCaptureMiddleware(AgentMiddleware):
    """Collect server-owned tool artifacts into private state.

    Artifacts are recorded here, by runtime code, so a later validation pass
    can check provenance. Model text never gets to declare its own artifacts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.artifacts: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = await handler(request)
        artifact = getattr(result, "artifact", None)
        if isinstance(artifact, dict):
            self.artifacts.append(artifact)
            if artifact.get("kind") == "image" or artifact.get("image_id"):
                self.images.append(artifact)
        elif isinstance(artifact, list):
            for entry in artifact:
                if isinstance(entry, dict):
                    self.artifacts.append(entry)
                    if entry.get("kind") == "image" or entry.get("image_id"):
                        self.images.append(entry)
        return result


def _hitl_interrupt_config(hitl_policy: dict[str, Any] | None) -> dict[str, Any]:
    """Translate the current approval policy into per-tool interrupt config.

    Only tools the policy names by exact tool name map directly onto the
    framework middleware. Server/origin rules stay in application code because
    they need the resolved call identity, which the framework cannot supply.
    """
    if not isinstance(hitl_policy, dict) or not hitl_policy.get("master_enabled", True):
        return {}
    tool_names = [name for name in (hitl_policy.get("global_tools") or []) if isinstance(name, str)]
    return {name: True for name in tool_names}


def build_specialist_middleware(
    *,
    runtime_model_resolver: Any,
    model_factory: Any,
    agent_key: str,
    agent_id: str,
    user_id: str | None,
    device_id: str | None,
    model_request: dict[str, Any] | None,
    usage_recorder: Any,
    authorize: AuthorizeCallable,
    hitl_policy: dict[str, Any] | None,
    max_model_calls: int,
    max_tool_calls: int,
    preflight: PreflightCallable | None = None,
    compact_messages: Callable[[list[Any]], list[Any]] | None = None,
    artifact_sink: ArtifactCaptureMiddleware | None = None,
) -> list[AgentMiddleware]:
    """Assemble one specialist's middleware stack.

    Composition order is the contract. First in the list is the outermost
    layer, so:

    * model calls: limits -> provider recovery -> budget -> usage recording,
      which puts one usage record around each real provider attempt;
    * tool calls: authorization -> approval -> artifact capture, which means a
      call the caller may not make is refused before a human is asked about it,
      and approval happens before the implementation runs.
    """
    runtime_model = RuntimeModelMiddleware(
        runtime_model_resolver=runtime_model_resolver,
        model_factory=model_factory,
        agent_key=agent_key,
        user_id=user_id,
        model_request=model_request,
        compact_messages=compact_messages,
    )

    stack: list[AgentMiddleware] = [
        ModelCallLimitMiddleware(thread_limit=max_model_calls, exit_behavior="error"),
        ToolCallLimitMiddleware(thread_limit=max_tool_calls, exit_behavior="error"),
        runtime_model,
    ]

    if preflight is not None:
        stack.append(
            RequestBudgetMiddleware(
                preflight=preflight,
                runtime_config_provider=lambda: runtime_model.runtime_config,
            )
        )

    stack.append(
        UsageRecordingMiddleware(
            usage_recorder=usage_recorder,
            agent_id=agent_id,
            runtime_config_provider=lambda: runtime_model.runtime_config,
        )
    )

    stack.append(
        ToolAuthorizationMiddleware(authorize=authorize, user_id=user_id, device_id=device_id)
    )

    interrupt_on = _hitl_interrupt_config(hitl_policy)
    if interrupt_on:
        stack.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    stack.append(artifact_sink or ArtifactCaptureMiddleware())
    return stack


def policy_requires_approval(
    tool_call: Any,
    *,
    hitl_policy: dict[str, Any] | None,
    tool_map: dict[str, Any] | None = None,
    mcp_manager: Any = None,
) -> bool:
    """Whether one resolved call identity needs approval under the policy.

    Kept in application code because it needs origin, server, and mutation
    provenance that the framework's tool-name matching cannot see.
    """
    if not isinstance(hitl_policy, dict):
        return False
    identity = resolve_call_identity(tool_call, tool_map=tool_map, mcp_manager=mcp_manager)
    return identity_requires_approval(identity, hitl_policy)


def sequence_of(middleware: Sequence[AgentMiddleware]) -> list[str]:
    """Middleware class names in composition order (diagnostics only)."""
    return [type(item).__name__ for item in middleware]
