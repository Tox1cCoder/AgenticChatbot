from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.ai.workflow.contracts import RoutingDecision, WorkflowRoutingException
from app.ai.workflow.inventory import build_routing_inventory
from app.ai.workflow.routing import (
    RoutingContext,
    RoutingDecisionValidator,
    RoutingService,
)
from app.core.runtime_modeling import (
    ResolvedRuntimeModelConfig,
    RuntimeFallbackConfig,
    StrictRuntimeResolutionError,
)

USER_ID = "11111111-1111-1111-1111-111111111111"
REQUEST_ID = "request-1"


@pytest.fixture
def inventory():
    return build_routing_inventory(
        base_agent_ids=["chat_agent", "search_agent", "rag_agent"],
        custom_agents={
            "custom_agent:alpha": {
                "runtime_agent_id": "custom_agent:alpha",
                "name": "Alpha",
                "description": "alpha work",
            }
        },
    )


@pytest.fixture
def context(inventory):
    return RoutingContext(
        message="what happened today",
        inventory_version=inventory.version,
        serialized_json='{"message":"what happened today"}',
    )


class FakeStructuredModel:
    def __init__(self, owner):
        self._owner = owner

    async def ainvoke(self, messages, config=None):
        self._owner.calls += 1
        self._owner.last_messages = messages
        if self._owner.side_effect is not None:
            effect = self._owner.side_effect
            if isinstance(effect, list):
                raise effect[min(self._owner.calls, len(effect)) - 1]
            raise effect
        results = self._owner.structured_results
        if results:
            parsed = results[min(self._owner.calls, len(results)) - 1]
        else:
            parsed = self._owner.structured_result
        return {"raw": SimpleNamespace(content=""), "parsed": parsed, "parsing_error": None}


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.structured_result: RoutingDecision | None = None
        self.structured_results: list[RoutingDecision] = []
        self.structured_schema = None
        self.structured_kwargs: dict = {}
        self.side_effect: BaseException | list[BaseException] | None = None
        self.last_messages = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return FakeStructuredModel(self)


class FakeResolver:
    def __init__(self, resolved=None):
        self.resolved = resolved or ResolvedRuntimeModelConfig(
            agent_key="router",
            provider="gemini",
            model="gemini-3-flash-preview",
            temperature=1.0,
            api_key="key",
            key_source="user",
            source="default",
            capabilities={"supports_structured_output": True},
        )
        self.last_allow_provider_fallback = None
        self.last_require_capabilities = None
        self.calls = 0
        self.raises: BaseException | None = None

    def resolve_runtime_config(
        self,
        user_id,
        agent_key,
        request_override=None,
        *,
        require_capabilities=frozenset(),
        allow_provider_fallback=True,
    ):
        self.calls += 1
        self.last_allow_provider_fallback = allow_provider_fallback
        self.last_require_capabilities = require_capabilities
        if self.raises is not None:
            raise self.raises
        return self.resolved


class FakeModelFactory:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def create_model_from_runtime(self, config, **kwargs):
        self.calls += 1
        if not (config.api_key or "").strip():
            raise StrictRuntimeResolutionError("missing_credentials", "router key missing")
        return self.model


def _service(*, model=None, resolver=None, attachment_checker=None, metrics=None):
    model = model or FakeModel()
    resolver = resolver or FakeResolver()
    service = RoutingService(
        runtime_model_resolver=resolver,
        model_factory=FakeModelFactory(model),
        settings=SimpleNamespace(
            routing_timeout_seconds=8.0,
            routing_max_attempts=2,
            router_model="gemini-3-flash-preview",
        ),
        validator=RoutingDecisionValidator(attachment_checker=attachment_checker),
        metrics=metrics,
    )
    service._fake_model = model  # test handle
    return service


async def test_routes_with_configured_provider_and_structured_output(context, inventory):
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="search_agent", confidence=0.91, reason="needs current sources"
    )
    service = _service(model=model)

    decision = await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )

    assert decision.agent_id == "search_agent"
    assert model.structured_schema is RoutingDecision
    assert model.structured_kwargs.get("include_raw") is True
    assert model.calls == 1


async def test_router_sends_instructions_and_data_as_separate_messages(context, inventory):
    model = FakeModel()
    model.structured_result = RoutingDecision(agent_id="chat_agent", confidence=0.5, reason="ok")
    service = _service(model=model)

    await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )

    system_message, human_message = model.last_messages
    assert system_message.type == "system"
    assert human_message.type == "human"
    assert context.message not in system_message.content


async def test_custom_agent_targets_are_accepted(context, inventory):
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="custom_agent:alpha", confidence=0.7, reason="attached specialist"
    )
    service = _service(model=model)

    decision = await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )
    assert decision.agent_id == "custom_agent:alpha"


async def test_unknown_target_is_not_reinterpreted_or_replaced(context, inventory):
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="missing", confidence=1.0, reason="invalid target"
    )
    service = _service(model=model)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )

    assert exc.value.error.code == "routing_target_unavailable"
    assert model.calls == 2


async def test_disabled_and_detached_targets_fail_closed(context):
    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent", "canvas_agent"],
        custom_agents={},
        disabled_agent_ids={"canvas_agent"},
    )
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="canvas_agent", confidence=0.9, reason="disabled target"
    )
    service = _service(model=model)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_target_unavailable"


async def test_target_removed_after_inventory_construction_is_a_race(context, inventory):
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="custom_agent:alpha", confidence=0.9, reason="attached at build time"
    )

    async def detached(agent_id, user_id):
        return False

    service = _service(model=model, attachment_checker=detached)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_target_unavailable"
    assert exc.value.error.details.get("cause") == "target_race"


async def test_two_failures_never_fall_back_to_chat(context, inventory):
    model = FakeModel()
    model.side_effect = TimeoutError()
    service = _service(model=model)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )

    assert exc.value.error.code == "routing_timeout"
    assert exc.value.error.retriable is True
    assert model.calls == 2


async def test_transient_provider_error_retries_once_then_succeeds(context, inventory):
    model = FakeModel()
    model.side_effect = None
    model.structured_results = [
        RoutingDecision(agent_id="missing", confidence=0.9, reason="bad target"),
        RoutingDecision(agent_id="chat_agent", confidence=0.9, reason="general help"),
    ]
    service = _service(model=model)

    decision = await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )
    assert decision.agent_id == "chat_agent"
    assert model.calls == 2


async def test_provider_error_maps_to_provider_unavailable(context, inventory):
    model = FakeModel()
    model.side_effect = ConnectionError("transport down")
    service = _service(model=model)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_provider_unavailable"
    assert model.calls == 2


async def test_invalid_structured_output_maps_to_invalid_output(context, inventory):
    model = FakeModel()
    model.structured_results = [None, None]
    service = _service(model=model)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_invalid_output"
    assert model.calls == 2


async def test_router_resolution_never_uses_provider_fallback(context, inventory):
    resolver = FakeResolver()
    resolver.resolved.provider_fallback = {
        "from": "openai",
        "to": "gemini",
        "reason": "provider_not_configured",
    }
    service = _service(resolver=resolver)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )

    assert exc.value.error.code == "routing_provider_unavailable"
    assert resolver.last_allow_provider_fallback is False


async def test_a_configured_standby_candidate_does_not_block_routing(context, inventory):
    """This asserted the opposite, and the opposite was a production outage.

    Two different fields were being read as the same thing.
    ``provider_fallback`` records a substitution that *already happened* — the
    router must refuse that, and the test above pins it. ``fallback_config``
    only records that a standby provider is *available* if some agent chooses
    to reach for one. The router never does: it uses ``configured.provider``
    and ``configured.model``, and ``create_model_from_runtime`` does not read
    the field at all.

    Refusing on its mere presence meant that configuring a second provider
    forbade routing entirely — so the more completely an account was set up,
    the more certainly every turn failed with
    ``routing_provider_unavailable``.
    """
    resolver = FakeResolver()
    resolver.resolved.fallback_config = RuntimeFallbackConfig(
        provider="openai", model="gpt-5-mini", temperature=1.0, api_key="k", key_source="db"
    )
    resolver.resolved.capabilities = {"supports_structured_output": True}
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="search_agent", confidence=0.9, reason="needs the web"
    )
    service = _service(model=model, resolver=resolver)

    decision = await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )

    assert decision.agent_id == "search_agent"
    assert resolver.last_allow_provider_fallback is False


async def test_the_router_is_handed_no_standby_it_could_ever_consume(context, inventory):
    """Neutralise the capability instead of refusing the request.

    The router not reaching for a standby is currently true by construction,
    which is a weak guarantee: one future edit in the strict path could start
    consuming one. Stripping the field means there is nothing to consume, so
    the guarantee holds without depending on nobody ever writing that line.
    """
    resolver = FakeResolver()
    resolver.resolved.fallback_config = RuntimeFallbackConfig(
        provider="openai", model="gpt-5-mini", temperature=1.0, api_key="k", key_source="db"
    )
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="chat_agent", confidence=0.8, reason="general talk"
    )
    factory_configs: list = []

    service = _service(model=model, resolver=resolver)
    original = service._model_factory.create_model_from_runtime

    def _capture(config, **kwargs):
        factory_configs.append(config)
        return original(config, **kwargs)

    service._model_factory.create_model_from_runtime = _capture

    await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )

    assert factory_configs, "the factory was never called"
    assert factory_configs[0].fallback_config is None
    assert factory_configs[0].provider == "gemini"


async def test_router_requires_structured_output_capability(context, inventory):
    resolver = FakeResolver()
    service = _service(resolver=resolver)
    resolver.resolved.capabilities = {"supports_structured_output": False}

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_provider_unavailable"
    assert resolver.last_require_capabilities == frozenset({"supports_structured_output"})


async def test_strict_resolution_error_is_typed_and_does_not_call_the_provider(context, inventory):
    resolver = FakeResolver()
    resolver.raises = StrictRuntimeResolutionError("missing_credentials", "no key for user")
    model = FakeModel()
    service = _service(model=model, resolver=resolver)

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_provider_unavailable"
    assert model.calls == 0


async def test_route_makes_at_most_two_attempts_against_the_same_model(context, inventory):
    model = FakeModel()
    model.side_effect = [TimeoutError(), ConnectionError("down"), ValueError("third")]
    factory_holder = {}

    service = _service(model=model)
    factory_holder["factory"] = service._model_factory

    with pytest.raises(WorkflowRoutingException):
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )

    assert model.calls == 2
    # One resolution, one model construction: the retry reuses the same object.
    assert factory_holder["factory"].calls == 1


async def test_route_respects_the_total_deadline(context, inventory):
    model = FakeModel()

    async def slow(messages, config=None):
        await asyncio.sleep(1.0)
        raise AssertionError("should have timed out")

    structured = FakeStructuredModel(model)
    structured.ainvoke = slow
    model.with_structured_output = lambda schema, **kwargs: structured

    service = _service(model=model)
    service._timeout_seconds = 0.05

    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(
            context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
        )
    assert exc.value.error.code == "routing_timeout"


async def test_metrics_exclude_model_generated_reason_text(context, inventory):
    import json

    from app.observability.routing import RoutingMetricsRecorder

    recorder = RoutingMetricsRecorder()
    model = FakeModel()
    model.structured_result = RoutingDecision(
        agent_id="chat_agent", confidence=0.9, reason="SECRET REASONING TEXT"
    )
    service = _service(model=model, metrics=recorder)

    await service.route(
        context, inventory, user_id=USER_ID, model_request=None, request_id=REQUEST_ID
    )

    exported = json.dumps(recorder.export())
    assert "SECRET REASONING TEXT" not in exported
    assert context.message not in exported


def test_routing_service_has_no_chat_fallback_or_provider_client():
    import pathlib

    source = pathlib.Path("app/ai/workflow/routing.py").read_text(encoding="utf-8")
    assert "google.genai" not in source
    assert "from google import genai" not in source
    assert 'return "chat_agent"' not in source


def test_confidence_never_drives_a_branch():
    import ast
    import pathlib

    source = pathlib.Path("app/ai/workflow/routing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            for child in ast.walk(node.test):
                if isinstance(child, ast.Attribute) and child.attr == "confidence":
                    raise AssertionError("confidence must never gate a routing branch")
