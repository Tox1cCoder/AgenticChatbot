from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.web_research.model_context import inject_latest_web_evidence
from app.ai.workflow.middleware import (
    RequestBudgetMiddleware,
    RuntimeModelMiddleware,
    WebEvidenceMiddleware,
    build_specialist_middleware,
)
from app.core.runtime_modeling import ResolvedRuntimeModelConfig, RuntimeFallbackConfig


def _source(source_id: str, *, status: str, snippet: str):
    return SimpleNamespace(
        source_id=source_id,
        title=f"Page {source_id}",
        url=f"https://example.test/{source_id.lower()}",
        snippet=snippet,
        status=status,
    )


class _Session:
    latest_image_query = "full T1 League of Legends team photo"
    latest_image_objective = "identify the complete current roster"

    def __init__(self, *, records=()) -> None:
        self.source_registry = SimpleNamespace(records=records)
        self.capability_calls: list[bool] = []

    def model_evidence_blocks(self, *, supports_vision: bool):
        self.capability_calls.append(supports_vision)
        if not supports_vision:
            return []
        return [
            {
                "type": "text",
                "text": "Image candidate I2; source S1; 1920x1080; domain www.t1.gg.",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,Ymx1ZQ==", "detail": "low"},
            },
        ]


def _request(**overrides):
    payload = {
        "model": SimpleNamespace(name="initial"),
        "messages": [HumanMessage(content="question")],
        "tools": [],
    }
    payload.update(overrides)
    return SimpleNamespace(
        **payload,
        override=lambda **changes: _request(**{**payload, **changes}),
    )


def _config(*, supports_vision: bool, fallback=None):
    return ResolvedRuntimeModelConfig(
        agent_key="chat",
        provider="primary",
        model="primary-model",
        temperature=0.2,
        api_key="key",
        key_source="user",
        source="request",
        capabilities={"supports_vision": supports_vision},
        fallback_config=fallback,
    )


def _content_text(messages) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
    return "\n".join(parts)


def _image_urls(messages) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


@pytest.mark.asyncio
async def test_candidate_pixels_are_injected_before_budget_preflight() -> None:
    session = _Session()
    config = _config(supports_vision=True)
    preflight_messages: list[list] = []
    provider_messages: list[list] = []
    evidence = WebEvidenceMiddleware(
        session=session, runtime_config_provider=lambda: config
    )

    async def preflight(request, _runtime_config):
        preflight_messages.append(request.messages)
        return None

    budget = RequestBudgetMiddleware(
        preflight=preflight, runtime_config_provider=lambda: config
    )

    async def provider(request):
        provider_messages.append(request.messages)
        return AIMessage(content="Blue [[image:I2]]")

    async def after_evidence(request):
        return await budget.awrap_model_call(request, provider)

    await evidence.awrap_model_call(_request(), after_evidence)

    assert preflight_messages == provider_messages
    assert _image_urls(provider_messages[0]) == ["data:image/png;base64,Ymx1ZQ=="]
    assert "Image candidate I2" in _content_text(provider_messages[0])


def test_evidence_replacement_is_idempotent() -> None:
    session = _Session()
    once = inject_latest_web_evidence(
        [HumanMessage(content="question")], session, supports_vision=True
    )
    twice = inject_latest_web_evidence(once, session, supports_vision=True)

    assert len(twice) == 2
    assert _image_urls(twice) == ["data:image/png;base64,Ymx1ZQ=="]


@pytest.mark.asyncio
async def test_text_fallback_receives_no_candidate_ids_or_pixels() -> None:
    fallback = RuntimeFallbackConfig(
        provider="fallback",
        model="text-model",
        temperature=0.2,
        api_key="fallback-key",
        key_source="system",
        capabilities={"supports_vision": False},
    )
    primary = _config(supports_vision=True, fallback=fallback)
    resolver = SimpleNamespace(resolve_runtime_config=lambda *_args: primary)
    factory = SimpleNamespace(
        create_model_from_runtime=lambda config: SimpleNamespace(name=config.model)
    )
    runtime = RuntimeModelMiddleware(
        runtime_model_resolver=resolver,
        model_factory=factory,
        agent_key="chat",
        user_id="user",
        model_request=None,
    )
    session = _Session()
    evidence = WebEvidenceMiddleware(
        session=session, runtime_config_provider=lambda: runtime.runtime_config
    )
    attempts: list[tuple[str, list]] = []

    async def provider(request):
        attempts.append((request.model.name, request.messages))
        if request.model.name == "primary-model":
            raise ConnectionError("primary unavailable")
        return AIMessage(content="text fallback")

    async def after_runtime(request):
        return await evidence.awrap_model_call(request, provider)

    await runtime.awrap_model_call(_request(), after_runtime)

    fallback_messages = attempts[1][1]
    assert _image_urls(fallback_messages) == []
    assert "candidate I" not in _content_text(fallback_messages)
    assert session.capability_calls == [True, False]


def test_evidence_middleware_is_between_runtime_and_preflight() -> None:
    tool_execution = SimpleNamespace()
    stack = build_specialist_middleware(
        runtime_model_resolver=SimpleNamespace(),
        model_factory=SimpleNamespace(),
        agent_key="chat",
        agent_id="chat_agent",
        user_id="user",
        model_request=None,
        usage_recorder=None,
        hitl_policy=None,
        max_model_calls=8,
        max_tool_calls=16,
        tool_execution=tool_execution,
        web_research_session=_Session(),
        preflight=lambda *_args: None,
    )
    names = [type(item).__name__ for item in stack]

    assert names.index("RuntimeModelMiddleware") < names.index("WebEvidenceMiddleware")
    assert names.index("WebEvidenceMiddleware") < names.index("RequestBudgetMiddleware")


def test_search_snippets_are_not_repeated_but_an_opened_page_survives() -> None:
    """``web_open`` exists to read pages whose snippets were insufficient.

    Capping a deliberate read at a triage bound would gut that path, so the
    mapping line stands alone for a search result and carries an excerpt for
    a page the model chose to open.
    """

    session = _Session(
        records=(
            _source("S1", status="search_result", snippet="triage " * 300),
            _source("S2", status="opened", snippet="deep read " * 300),
        )
    )

    text = _content_text(
        inject_latest_web_evidence([HumanMessage(content="q")], session, supports_vision=False)
    )

    assert "S1: Page S1 | https://example.test/s1" in text
    assert "triage triage" not in text
    assert "S2: Page S2 | https://example.test/s2" in text
    assert "deep read deep read" in text


def test_the_literal_image_target_reaches_the_model_with_the_pixels() -> None:
    session = _Session(records=(_source("S1", status="search_result", snippet="x"),))

    text = _content_text(
        inject_latest_web_evidence([HumanMessage(content="q")], session, supports_vision=True)
    )

    assert "IMAGE TARGET: full T1 League of Legends team photo" in text
    assert "Match the requested visual form literally" in text
    assert "A roster graphic or list of names is not a full team photo" in text
    assert "Select none if no candidate visibly matches" in text


def test_a_candidate_label_carries_only_server_known_facts() -> None:
    session = _Session()

    text = _content_text(
        inject_latest_web_evidence([HumanMessage(content="q")], session, supports_vision=True)
    )

    assert "Image candidate I2; source S1; 1920x1080; domain www.t1.gg." in text


def test_no_image_target_line_without_image_blocks() -> None:
    session = _Session(records=(_source("S1", status="search_result", snippet="x"),))

    text = _content_text(
        inject_latest_web_evidence([HumanMessage(content="q")], session, supports_vision=False)
    )

    assert "IMAGE TARGET" not in text
