"""Image-generator harvests inline images instead of reporting 'no response'.

When the runtime model is itself image-capable, it returns generated images
inline with no usable text. The agent must keep those images and produce a
user-facing message rather than early-returning empty content.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agents.image_generator_agent import ImageGenerationOutcome, ImageGeneratorAgent
from app.usage.types import NormalizedUsage


def _bare_agent() -> ImageGeneratorAgent:
    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = "gemini-3-pro-image-preview"
    agent.default_aspect_ratio = "1:1"
    agent.max_images = 4
    return agent


def test_harvest_inline_images_wraps_records_and_caps():
    agent = _bare_agent()
    agent.max_images = 2
    response = SimpleNamespace(
        metadata={
            "response_inline_images": [
                {"data": "AAA", "mime": "image/png"},
                {"data": "BBB", "mime": "image/jpeg"},
                {"data": "CCC", "mime": "image/png"},
            ]
        }
    )
    images = agent._harvest_inline_images(response, "a cat on a roof")
    assert len(images) == 2
    assert images[0] == {
        "data": "AAA",
        "mime": "image/png",
        "prompt": "a cat on a roof",
        "model": "gemini-3-pro-image-preview",
        "aspect_ratio": "1:1",
    }


def test_harvest_returns_empty_without_inline_images():
    agent = _bare_agent()
    assert agent._harvest_inline_images(SimpleNamespace(metadata={}), "x") == []
    assert agent._harvest_inline_images(SimpleNamespace(metadata=None), "x") == []


@pytest.mark.asyncio
async def test_invoke_history_harvests_images_when_model_returns_them():
    agent = _bare_agent()
    base_response = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=None),
        metadata={"response_inline_images": [{"data": "IMGDATA", "mime": "image/png"}]},
        error=None,
    )

    with patch(
        "app.ai.agents.base_agent.BaseAgent.invoke_model_with_history",
        new=AsyncMock(return_value=base_response),
    ):
        agent._generate_user_facing_response = AsyncMock(return_value="Here is your image!")
        agent._generate_images = AsyncMock(side_effect=AssertionError("must not regenerate"))

        result = await agent.invoke_model_with_history(
            [HumanMessage(content="draw a fox")],
            conversation_history=[],
            persona=None,
        )

    assert result.metadata["images"] == [
        {
            "data": "IMGDATA",
            "mime": "image/png",
            "prompt": "draw a fox",
            "model": "gemini-3-pro-image-preview",
            "aspect_ratio": "1:1",
        }
    ]
    assert "response_inline_images" not in result.metadata
    assert result.message.content == "Here is your image!"


@pytest.mark.asyncio
async def test_invoke_history_keeps_text_when_model_returns_caption():
    agent = _bare_agent()
    base_response = SimpleNamespace(
        message=SimpleNamespace(content="A serene fox in autumn light.", tool_calls=None),
        metadata={"response_inline_images": [{"data": "IMGDATA", "mime": "image/png"}]},
        error=None,
    )

    with patch(
        "app.ai.agents.base_agent.BaseAgent.invoke_model_with_history",
        new=AsyncMock(return_value=base_response),
    ):
        agent._generate_user_facing_response = AsyncMock(return_value="fallback")

        result = await agent.invoke_model_with_history(
            [HumanMessage(content="draw a fox")],
            conversation_history=[],
            persona=None,
        )

    assert result.message.content == "A serene fox in autumn light."
    assert result.metadata["images"][0]["data"] == "IMGDATA"
    agent._generate_user_facing_response.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_history_passes_current_multimodal_image_to_edit_generation():
    agent = _bare_agent()
    base_response = SimpleNamespace(
        message=SimpleNamespace(content="Edit the source into a watercolor", tool_calls=None),
        metadata={},
        error=None,
    )

    with patch(
        "app.ai.agents.base_agent.BaseAgent.invoke_model_with_history",
        new=AsyncMock(return_value=base_response),
    ):
        agent._generate_images = AsyncMock(
            return_value=ImageGenerationOutcome(
                images=[{"data": "EDITED"}],
                narrative="Done",
                usage=NormalizedUsage(source="unavailable"),
            )
        )
        result = await agent.invoke_model_with_history(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "make this a watercolor"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,YWJj"},
                        },
                    ]
                )
            ],
            conversation_history=[],
            persona=None,
        )

    agent._generate_images.assert_awaited_once_with(
        "Edit the source into a watercolor",
        "make this a watercolor",
        source_images=[{"data": "YWJj", "mime": "image/png"}],
    )
    assert result.metadata["images"] == [{"data": "EDITED"}]
    assert result.message.content == "Done"


@pytest.mark.asyncio
async def test_invoke_history_uses_terminal_dedicated_image_usage_for_context_gauge():
    agent = _bare_agent()
    agent.model_name = "gemini-3-pro-image"
    base_response = SimpleNamespace(
        message=SimpleNamespace(content="A polished fox illustration", tool_calls=None),
        metadata={
            "context_window": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "context_window_tokens": 128_000,
                "max_input_tokens": 128_000,
                "max_output_tokens": 16_384,
                "limit_type": "shared_context",
                "source": "registry",
                "known": True,
            }
        },
        error=None,
    )
    outcome = SimpleNamespace(
        images=[{"data": "IMAGE", "mime": "image/png"}],
        narrative="Done",
        usage=NormalizedUsage(
            input_tokens=120,
            output_tokens=320,
            total_tokens=440,
            generated_images=1,
            source="provider_reported",
        ),
    )

    with patch(
        "app.ai.agents.base_agent.BaseAgent.invoke_model_with_history",
        new=AsyncMock(return_value=base_response),
    ):
        agent._generate_images = AsyncMock(return_value=outcome)
        result = await agent.invoke_model_with_history(
            [HumanMessage(content="draw a fox")],
            conversation_history=[],
            persona=None,
        )

    context = result.metadata["context_window"]
    assert context["provider"] == "gemini"
    assert context["model"] == "gemini-3-pro-image"
    assert context["limit_type"] == "separate_io"
    assert context["context_window_tokens"] is None
    assert context["max_input_tokens"] == 65_536
    assert context["max_output_tokens"] == 32_768
    assert context["input_tokens"] == 120
    assert context["output_tokens"] == 320
    assert context["total_tokens"] == 440
    assert context["usage_ratio"] == 320 / 32_768
    assert context["usage_ratio_basis"] == "most_constrained_io_limit"
