"""Regression: the answer model must receive validated candidate pixels."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.ai.schemas import AgentType
from app.ai.web_research.service import WebResearchService
from app.ai.web_tools import create_web_search_tool
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
)
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


class _TwoRoundModel(BaseChatModel):
    requests: list[list] = []
    call_count: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "two-round-web-evidence"

    def bind_tools(self, _tools, **_kwargs):
        return self

    def _answer(self, messages):
        self.requests.append(list(messages))
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
        else:
            result = AIMessage(
                content=(
                    "The blue version is current "
                    "[Release notes](https://source.test/release). [[image:I2]]"
                )
            )
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=result)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._answer(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._answer(messages)


@pytest.mark.asyncio
async def test_second_answer_model_request_contains_labeled_validated_pixels() -> None:
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
    model = _TwoRoundModel()
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
        model="vision-model",
        temperature=0.0,
        api_key="key",
        key_source="test",
        source="test",
        warnings=[],
        capabilities={"supports_vision": True},
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

    assert len(model.requests) == 2
    second = model.requests[1]
    evidence = second[-1].content
    assert evidence[1]["text"].startswith("Image candidate I1")
    assert evidence[2]["image_url"]["url"].endswith("cmVkLXBpeGVscw==")
    assert evidence[3]["text"].startswith("Image candidate I2")
    assert evidence[4]["image_url"]["url"].endswith("Ymx1ZS1waXhlbHM=")
    assert "<!--rich:image:web:" in outcome.response.message.content
    assert "[[image:" not in outcome.response.message.content
    assert len(outcome.provenance.rich_items) == 1
