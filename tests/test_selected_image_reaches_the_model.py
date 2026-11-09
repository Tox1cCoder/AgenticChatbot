"""Regression: the answer model must receive validated candidate pixels."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.ai.graph import MultiAgentWorkflow, _build_inline_rich_inventory_for_state
from app.ai.schemas import AgentType
from app.ai.web_research.service import WebResearchService
from app.ai.web_tools import create_web_search_tool
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
)
from app.core.response_constants import build_bot_metadata
from app.core.rich_placement import finalize_article_content
from app.services.web_image_service import FetchedWebImage


class _ProviderTool:
    def __init__(self, name: str, payload: dict) -> None:
        self.name = name
        self.payload = payload

    async def ainvoke(self, _args):
        return json.dumps(self.payload)


class _ImageService:
    async def fetch_url(self, url: str, *, provider: str, max_bytes: int | None = None):
        content = b"red-pixels" if url.endswith("red.png") else b"blue-pixels"
        assert max_bytes is None or len(content) <= max_bytes
        return FetchedWebImage(
            content=content,
            media_type="image/png",
            width=32,
            height=32,
        )

    async def register(self, **kwargs):
        return SimpleNamespace(id=uuid4(), **kwargs)

    async def mark_selected(self, _ids, **_scope):
        return None

    async def release_references(self, _ids, **_scope):
        return None


class _GroundingRepairModel(BaseChatModel):
    requests: list[list] = []
    call_count: int = 0
    bound_tools: list = []
    tool_counts: list[int] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "grounding-repair-web-evidence"

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools = list(tools)
        return self

    def _answer(self, messages):
        self.requests.append(list(messages))
        self.tool_counts.append(len(self.bound_tools))
        self.bound_tools = []
        if self.call_count == 0:
            result = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "web_search",
                        "args": {
                            "query": "release interface",
                            "objective": "identify the current interface",
                            "visual_intent": "comparison",
                            "image_query": "release interface",
                        },
                    }
                ],
            )
        elif self.call_count == 1:
            result = AIMessage(content="Here is the current interface. [[image:I2]]")
        else:
            result = AIMessage(content=("The blue version is current [[source:S1]]. [[image:I2]]"))
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=result)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._answer(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._answer(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capabilities", [{}, {"supports_vision": False}, {"supports_vision": True}]
)
async def test_invalid_image_draft_is_repaired_with_the_same_validated_pixels(
    monkeypatch, capabilities
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    text_tool = _ProviderTool(
        "tavily_search",
        {
            "results": [
                {
                    "url": "https://source.test/release",
                    "title": "Release notes",
                    "content": "The blue interface is current.",
                }
            ]
        },
    )
    image_tool = _ProviderTool(
        "brave_image_search",
        {
            "images": [
                {
                    "thumbnail_url": "https://images.test/red.png",
                    "source_url": "https://source.test/release",
                    "title": "Red interface",
                },
                {
                    "thumbnail_url": "https://images.test/blue.png",
                    "source_url": "https://source.test/release",
                    "title": "Blue interface",
                },
            ]
        },
    )
    owner = SimpleNamespace(tools=[text_tool, image_tool], agent_config_key="chat")
    model = _GroundingRepairModel()
    definition = SpecialistDefinition(
        agent_id="chat_agent",
        agent_type=AgentType.CHAT,
        model_config_key="chat",
        system_prompt_factory=lambda _request: "Use grounded web evidence.",
        tool_factory=lambda _request: [create_web_search_tool()],
        agent=owner,
        output_policy_ids=("public_content",),
    )
    runtime_config = SimpleNamespace(
        agent_key="chat",
        provider="test",
        model="custom-chat-model",
        temperature=0.0,
        api_key="key",
        key_source="test",
        source="test",
        warnings=[],
        capabilities=capabilities,
        fallback_config=None,
    )
    factory = SpecialistFactory(
        definitions={"chat_agent": definition},
        runtime_model_resolver=SimpleNamespace(
            resolve_runtime_config=lambda *_args, **_kwargs: runtime_config
        ),
        model_factory=SimpleNamespace(create_model_from_runtime=lambda _config: model),
        web_research_service=WebResearchService(image_service=_ImageService()),
        settings=SimpleNamespace(
            generation_hard_model_calls_per_epoch=4,
            generation_soft_model_calls_per_epoch=3,
            generation_hard_tool_calls_per_epoch=4,
            generation_soft_tool_calls_per_epoch=3,
        ),
    )
    request = SpecialistRequest(
        agent_id="chat_agent",
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        device_id=None,
        persona=None,
        model_request=None,
        messages=[HumanMessage(content="Which interface is current?")],
        extras={"turn_id": str(uuid4())},
    )

    outcome = await factory.invoke(request)

    assert len(model.requests) == 3
    second = model.requests[1]
    evidence = second[-1].content
    assert len(evidence) == 5  # One evidence-text block and two label/pixel pairs.
    assert evidence[1]["text"].startswith("Image candidate I1")
    assert evidence[2]["image_url"]["url"].endswith("cmVkLXBpeGVscw==")
    assert evidence[3]["text"].startswith("Image candidate I2")
    assert evidence[4]["image_url"]["url"].endswith("Ymx1ZS1waXhlbHM=")
    third = model.requests[2]
    correction = " ".join(
        str(message.content) for message in third if isinstance(message, HumanMessage)
    )
    assert "Rewrite the complete answer" in correction
    assert model.tool_counts[-1] == 0
    repaired_evidence = next(
        message.content
        for message in third
        if (message.additional_kwargs or {}).get("web_evidence_v1")
    )
    assert repaired_evidence[3]["text"].startswith("Image candidate I2")
    assert repaired_evidence[4]["image_url"]["url"].endswith("Ymx1ZS1waXhlbHM=")
    assert "https://source.test/release" in outcome.response.message.content
    assert "<!--rich:image:web:" in outcome.response.message.content
    assert "[[image:" not in outcome.response.message.content
    assert len(outcome.provenance.rich_items) == 1

    # Continue can carry a tool-only inventory built before this search. Its
    # empty image allowlist must not discard images inspected by the specialist.
    context = {
        "inline_rich_response_v1": True,
        "rich_item_candidates": [
            {
                "id": "tool:previous-epoch",
                "type": "tool_render",
                "display_policy": "inline_or_append",
                "payload": {"render": {"version": 1, "type": "json", "data": {"value": 1}}},
            }
        ],
    }
    _build_inline_rich_inventory_for_state(context)
    assert context["_presented_rich_image_ids"] == []
    context["rich_item_candidates"].append(
        {
            "id": "image:tool:uninspected",
            "type": "image",
            "display_policy": "inline_only",
            "alt_text": "Uninspected tool image",
            "payload": {"url": "https://images.test/uninspected.png", "mime_type": "image/png"},
        }
    )
    response = outcome.response
    response.message.content += "\n\n<!--rich:image:tool:uninspected-->"
    MultiAgentWorkflow.__new__(MultiAgentWorkflow)._merge_tool_artifacts(
        {"context": context}, response
    )
    content = finalize_article_content(response, response.message.content)
    metadata = build_bot_metadata(response)

    assert "<!--rich:image:web:" in content
    assert "image:tool:uninspected" not in content
    assert [item["type"] for item in metadata["rich_items"]] == ["image", "tool_render"]
    assert metadata["rich_reference_warnings"] == []
