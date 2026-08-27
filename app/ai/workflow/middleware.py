"""Focused middleware for routing-v2 specialist subgraphs.

The framework owns the ReAct loop; these pieces supply only the behavior the
framework cannot know about: which model this user gets, which scope its tools
execute in, when a human must approve, and how usage and artifacts are
accounted for.

Each class does one thing and is tested on its own. The order they compose in
is itself a contract — see :func:`build_specialist_middleware`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.ai.context_overflow import is_context_overflow_error
from app.ai.hitl_config import (
    build_tool_interrupt_payload,
    calls_requiring_approval,
    identity_requires_approval,
    resolve_call_identity,
)
from app.ai.tool_context import tool_execution_context
from app.ai.tool_execution import (
    build_rejected_tool_artifacts,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from app.ai.utils import apply_hitl_decisions, normalize_tool_call
from app.core.runtime_modeling import ResolvedRuntimeModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "RequestBudgetMiddleware",
    "RuntimeModelMiddleware",
    "SpecialistToolScope",
    "ToolApprovalMiddleware",
    "ToolExecutionMiddleware",
    "UsageRecordingMiddleware",
    "build_specialist_middleware",
]

PreflightCallable = Callable[[Any, ResolvedRuntimeModelConfig], Awaitable[Any]]
ToolFactory = Callable[[], Awaitable[list[Any]]]


class SpecialistToolScope:
    """The authenticated scope one specialist invocation executes tools in.

    Shared by the execution and approval middleware so both resolve the same
    tool map: approval decides on a tool's provenance, and provenance comes
    from the very object that will run.
    """

    def __init__(
        self,
        *,
        agent: Any,
        agent_key: str,
        conversation_id: str | None,
        user_id: str | None,
        device_id: str | None,
        rich_response_capable: bool = True,
        internal_tools: list[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.agent_key = agent_key
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.device_id = device_id
        self.rich_response_capable = rich_response_capable
        self.internal_tools = internal_tools
        self._tool_map: dict[str, Any] | None = None
        self._bound: dict[str, Any] = {}

    def offer(self, tools: Sequence[Any]) -> None:
        """Record the tool objects this invocation actually bound to the model.

        Execution and approval must judge the same objects the model was
        offered. A tool the agent map does not know about — an internal
        ``hand_off``, a skill tool — is still callable, so it belongs in the
        execution map too.
        """
        self._bound = {tool.name: tool for tool in tools if getattr(tool, "name", None)}
        if self._tool_map is not None:
            self._tool_map.update(self._bound)

    async def tool_map(self) -> dict[str, Any]:
        """The execution map for this invocation, resolved once.

        The same mutable dict is reused for every call in the turn, so a tool
        that ``tool_search`` loads part way through is executable by the calls
        that follow it.
        """
        if self._tool_map is None:
            self._tool_map = await ensure_agent_tool_map(
                self.agent,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                device_id=self.device_id,
                internal_tools=self.internal_tools,
            )
            self._tool_map.update(self._bound)
        return self._tool_map

    def execution_context(self):
        return tool_execution_context(
            self.conversation_id,
            self.user_id,
            self.agent_key,
            self.device_id,
            rich_response_capable=self.rich_response_capable,
        )


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


class ToolExecutionMiddleware(AgentMiddleware):
    """Run tool calls through the product's execution pipeline.

    The framework's tool node would call the bound function directly, which
    skips everything that wraps a tool call in this product: the execution
    context every device-scoped tool checks itself against, the artifact and
    image records the UI trace and output validation read, blob offloading,
    error normalization, and the deferred-tool refresh that makes a
    ``tool_search`` result callable in the same turn.

    LangGraph defers unknown-tool validation to the interceptor, so this also
    serves a tool that was loaded after the agent was compiled — which is why
    it re-offers the live tool set on every model call.
    """

    def __init__(self, *, scope: SpecialistToolScope, tool_factory: ToolFactory) -> None:
        super().__init__()
        self._scope = scope
        self._tool_factory = tool_factory
        self.artifacts: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Bind the tool set as it stands now, not as it stood at compile time."""
        tools = list(await self._tool_factory() or [])
        self._scope.offer(tools)
        return await handler(request.override(tools=tools))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = normalize_tool_call(request.tool_call)
        tool_map = await self._scope.tool_map()

        with self._scope.execution_context():
            outputs, artifacts, images = await execute_tool_calls(
                tool_calls=[call],
                tool_map=tool_map,
                capture_images=True,
                device_id=self._scope.device_id,
                agent=self._scope.agent,
                conversation_id=self._scope.conversation_id,
                user_id=self._scope.user_id,
            )

        self.artifacts.extend(artifacts)
        self.images.extend(images)

        output = outputs[0] if outputs else {}
        return ToolMessage(
            content=str(output.get("content") or ""),
            tool_call_id=str(output.get("tool_call_id") or call.get("id") or ""),
            name=str(output.get("name") or call.get("name") or "tool"),
            status="error" if _is_error(artifacts) else "success",
        )


def _is_error(artifacts: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(artifact, dict) and artifact.get("status") == "error" for artifact in artifacts
    )


class ToolApprovalMiddleware(AgentMiddleware):
    """Pause for human approval before a gated tool call runs.

    The framework's own HITL middleware matches tool names. This product's
    policy is scoped by call *identity* — client origin, MCP server, and a
    mutation floor — which needs the resolved tool object, and its interrupt
    payload carries tool-call ids, device provenance, and redacted arguments
    that the API contract depends on. So the gate is here, and it reuses the
    same identity resolution and decision application every other approval
    point uses.

    Rejected calls keep their place on the ``AIMessage`` and receive a paired
    ``ToolMessage``, which is both a valid history and what stops the tool node
    from running them: it dispatches only calls that have no result yet.
    """

    def __init__(self, *, scope: SpecialistToolScope, hitl_policy: dict[str, Any]) -> None:
        super().__init__()
        self._scope = scope
        self._policy = hitl_policy
        self.rejected_artifacts: list[dict[str, Any]] = []

    async def aafter_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None or not last_ai.tool_calls:
            return None

        calls = [normalize_tool_call(call) for call in last_ai.tool_calls]
        tool_map = await self._scope.tool_map()
        gated_ids = calls_requiring_approval(
            calls, policy=self._policy, tool_map=tool_map, mcp_manager=await _mcp_manager()
        )
        if not gated_ids:
            return None

        gated = [call for call in calls if str(call.get("id") or "") in gated_ids]
        decisions = interrupt(
            build_tool_interrupt_payload(gated, tool_map=tool_map, device_id=self._scope.device_id)
        )
        approved, rejected_feedback = apply_hitl_decisions(gated, decisions or [])

        # An approver may rewrite a call's arguments. Overlay them onto the
        # original tool call rather than replacing it, so the call keeps the
        # shape the tool node dispatches on.
        decided_by_id = {str(call.get("id")): call for call in approved}
        revised = [
            {**original, "name": decided["name"], "args": decided.get("args") or {}}
            if (decided := decided_by_id.get(str(call.get("id"))))
            else original
            for call, original in zip(calls, last_ai.tool_calls, strict=True)
        ]
        rejections = [
            ToolMessage(
                content=rejected_feedback[call_id],
                tool_call_id=call_id,
                name=str(call.get("name") or "tool"),
                status="error",
            )
            for call in calls
            if (call_id := str(call.get("id") or "")) in rejected_feedback
        ]
        if rejected_feedback:
            self.rejected_artifacts.extend(
                build_rejected_tool_artifacts(tool_calls=calls, rejected_feedback=rejected_feedback)
            )

        return {"messages": [last_ai.model_copy(update={"tool_calls": revised}), *rejections]}


async def _mcp_manager() -> Any:
    """The MCP manager, or nothing — a lookup failure must not skip the gate."""
    try:
        from app.ai.mcp_registry import get_global_mcp_manager

        return await get_global_mcp_manager()
    except Exception as exc:  # noqa: BLE001 - identity falls back to tool metadata
        logger.debug("MCP manager unavailable while resolving approval identity: %s", exc)
        return None


def build_specialist_middleware(
    *,
    runtime_model_resolver: Any,
    model_factory: Any,
    agent_key: str,
    agent_id: str,
    user_id: str | None,
    model_request: dict[str, Any] | None,
    usage_recorder: Any,
    hitl_policy: dict[str, Any] | None,
    max_model_calls: int,
    max_tool_calls: int,
    tool_execution: ToolExecutionMiddleware,
    approval: ToolApprovalMiddleware | None = None,
    preflight: PreflightCallable | None = None,
    compact_messages: Callable[[list[Any]], list[Any]] | None = None,
) -> list[AgentMiddleware]:
    """Assemble one specialist's middleware stack.

    Composition order is the contract. First in the list is the outermost
    layer, so model calls run limits -> provider recovery -> budget -> usage
    recording -> live tool binding, which puts one usage record around each
    real provider attempt and offers the tool set as it stands at that moment.

    Approval runs after the model and before the tool node dispatches, so a
    gated call is decided on before its implementation can run.
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
    stack.append(tool_execution)

    if approval is not None and _policy_is_active(hitl_policy):
        stack.append(approval)

    return stack


def _policy_is_active(hitl_policy: dict[str, Any] | None) -> bool:
    return isinstance(hitl_policy, dict) and bool(hitl_policy.get("master_enabled", True))


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
