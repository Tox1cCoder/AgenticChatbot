"""Provider abstraction for streaming image generation (Gemini / OpenAI)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from app.ai.image_generation import (
    ImageFinal,
    ImageGenerationRequest,
    ImagePartial,
    ImageUsage,
    NarrativeDelta,
    resolve_image_provider,
)
from app.ai.image_generation.gemini import GeminiImageProvider
from app.ai.image_generation.openai_provider import OpenAIImageProvider
from app.usage.types import NormalizedUsage


def _request(**overrides) -> ImageGenerationRequest:
    defaults = {
        "prompt": "a fox in autumn light",
        "model": "gemini-3-pro-image-preview",
        "max_images": 2,
        "aspect_ratio": "1:1",
    }
    defaults.update(overrides)
    return ImageGenerationRequest(**defaults)


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


def _gemini_chunk(*parts) -> SimpleNamespace:
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


def _image_part(data: bytes, mime: str = "image/png") -> SimpleNamespace:
    return SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime), text=None)


def _text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(inline_data=None, text=text)


def _gemini_client(chunks: list, *, calls: dict | None = None) -> SimpleNamespace:
    async def _stream(**_kwargs):
        if calls is not None:
            calls["generate_content_stream"] = calls.get("generate_content_stream", 0) + 1
        for chunk in chunks:
            yield chunk

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=lambda **kwargs: _stream(**kwargs))
        )
    )


def _usage_metadata(**fields) -> SimpleNamespace:
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_gemini_yields_finals_and_narrative_then_terminal_usage():
    calls: dict = {}
    client = _gemini_client(
        [
            _gemini_chunk(_text_part("Here is "), _image_part(b"png-bytes")),
            _gemini_chunk(_text_part("your fox.")),
        ],
        calls=calls,
    )
    provider = GeminiImageProvider(client)

    events = [event async for event in provider.stream_generate(_request())]

    # No usage_metadata on any chunk -> exactly one terminal ImageUsage that is
    # unavailable, always emitted after the stream is exhausted.
    assert events == [
        NarrativeDelta(text="Here is "),
        ImageFinal(
            index=0,
            data_b64=base64.b64encode(b"png-bytes").decode("utf-8"),
            mime="image/png",
        ),
        NarrativeDelta(text="your fox."),
        ImageUsage(usage=NormalizedUsage(source="unavailable"), provider_request_id=None),
    ]
    assert calls["generate_content_stream"] == 1


@pytest.mark.asyncio
async def test_gemini_consumes_terminal_usage_chunk_after_max_images():
    # The usage chunk arrives AFTER the image chunk; capping emission at
    # max_images must not truncate the stream before the provider's accounting.
    client = _gemini_client(
        [
            _gemini_chunk(_image_part(b"a")),
            SimpleNamespace(
                candidates=[],
                usage_metadata=_usage_metadata(
                    prompt_token_count=120,
                    candidates_token_count=2048,
                    total_token_count=2168,
                ),
                response_id="resp-99",
            ),
        ]
    )
    provider = GeminiImageProvider(client)

    events = [event async for event in provider.stream_generate(_request(max_images=1))]

    finals = [event for event in events if isinstance(event, ImageFinal)]
    usages = [event for event in events if isinstance(event, ImageUsage)]
    assert [event.index for event in finals] == [0]
    assert len(usages) == 1
    usage_event = usages[0]
    assert usage_event.usage.input_tokens == 120
    assert usage_event.usage.output_tokens == 2048
    assert usage_event.usage.total_tokens == 2168
    assert usage_event.usage.source == "provider_reported"
    assert usage_event.provider_request_id == "resp-99"


@pytest.mark.asyncio
async def test_gemini_caps_emission_but_keeps_consuming_for_usage():
    client = _gemini_client(
        [
            _gemini_chunk(_image_part(b"a"), _image_part(b"b"), _image_part(b"c")),
            SimpleNamespace(
                candidates=[],
                usage_metadata=_usage_metadata(total_token_count=10),
            ),
        ]
    )
    provider = GeminiImageProvider(client)

    events = [event async for event in provider.stream_generate(_request(max_images=2))]

    finals = [event for event in events if isinstance(event, ImageFinal)]
    usages = [event for event in events if isinstance(event, ImageUsage)]
    assert [event.index for event in finals] == [0, 1]
    assert len(usages) == 1
    assert usages[0].usage.total_tokens == 10


@pytest.mark.asyncio
async def test_gemini_supports_awaitable_stream_open():
    async def _stream():
        yield _gemini_chunk(_image_part(b"a"))

    async def _open(**_kwargs):
        return _stream()

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_open))
    )
    provider = GeminiImageProvider(client)

    events = [event async for event in provider.stream_generate(_request(max_images=1))]
    assert isinstance(events[0], ImageFinal)


def test_gemini_skips_invalid_source_images():
    provider = GeminiImageProvider(SimpleNamespace())
    contents = provider._build_contents(
        _request(source_images=[{"data": "!!!not-base64!!!"}, {"data": "YWJj"}])
    )
    # prompt part + the one valid source image
    assert len(contents[0].parts) == 2


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _openai_event(event_type: str, b64: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, b64_json=b64, output_format="png", **extra)


def _openai_client(events: list, calls: dict) -> SimpleNamespace:
    async def _stream():
        for event in events:
            yield event

    async def _generate(**kwargs):
        calls["generate"] = kwargs
        return _stream()

    async def _edit(**kwargs):
        calls["edit"] = kwargs
        return _stream()

    return SimpleNamespace(images=SimpleNamespace(generate=_generate, edit=_edit))


@pytest.mark.asyncio
async def test_openai_partials_then_final_then_usage():
    calls: dict = {}
    client = _openai_client(
        [
            _openai_event("image_generation.partial_image", "UUFB", partial_image_index=0),
            _openai_event("image_generation.partial_image", "UUFC", partial_image_index=1),
            _openai_event(
                "image_generation.completed",
                "UUFD",
                usage={"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            ),
        ],
        calls,
    )
    provider = OpenAIImageProvider(client=client)

    events = [event async for event in provider.stream_generate(_request(model="gpt-image-1"))]

    assert events == [
        ImagePartial(index=0, data_b64="UUFB", mime="image/png", seq=1),
        ImagePartial(index=0, data_b64="UUFC", mime="image/png", seq=2),
        ImageFinal(index=0, data_b64="UUFD", mime="image/png"),
        ImageUsage(
            usage=NormalizedUsage(
                input_tokens=50,
                output_tokens=100,
                total_tokens=150,
                source="provider_reported",
            ),
            provider_request_id=None,
        ),
    ]
    assert calls["generate"]["stream"] is True
    assert calls["generate"]["size"] == "1024x1024"
    assert "edit" not in calls


@pytest.mark.asyncio
async def test_openai_completed_without_usage_still_emits_terminal_usage():
    calls: dict = {}
    client = _openai_client([_openai_event("image_generation.completed", "UUFD")], calls)
    provider = OpenAIImageProvider(client=client)

    events = [event async for event in provider.stream_generate(_request(model="gpt-image-1"))]

    assert events == [
        ImageFinal(index=0, data_b64="UUFD", mime="image/png"),
        ImageUsage(usage=NormalizedUsage(source="unavailable"), provider_request_id=None),
    ]


@pytest.mark.asyncio
async def test_openai_uses_edit_endpoint_for_source_images():
    calls: dict = {}
    client = _openai_client([_openai_event("image_edit.completed", "RUZH")], calls)
    provider = OpenAIImageProvider(client=client)

    source = base64.b64encode(b"src").decode("utf-8")
    events = [
        event
        async for event in provider.stream_generate(
            _request(model="gpt-image-1", source_images=[{"data": source, "mime": "image/png"}])
        )
    ]

    assert events == [
        ImageFinal(index=0, data_b64="RUZH", mime="image/png"),
        ImageUsage(usage=NormalizedUsage(source="unavailable"), provider_request_id=None),
    ]
    assert "generate" not in calls
    assert calls["edit"]["image"].read() == b"src"


def test_openai_size_mapping():
    assert OpenAIImageProvider._size_for_aspect("16:9") == "1536x1024"
    assert OpenAIImageProvider._size_for_aspect("9:16") == "1024x1536"
    assert OpenAIImageProvider._size_for_aspect("7:5") == "auto"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_routes_openai_models_to_openai_provider():
    provider = resolve_image_provider("gpt-image-1", openai_api_key="test-key")
    assert isinstance(provider, OpenAIImageProvider)
    provider = resolve_image_provider("dall-e-3", openai_api_key="test-key")
    assert isinstance(provider, OpenAIImageProvider)


def test_registry_defaults_to_gemini_with_client():
    client = SimpleNamespace()
    provider = resolve_image_provider("gemini-3-pro-image-preview", gemini_client=client)
    assert isinstance(provider, GeminiImageProvider)


def test_registry_returns_none_without_gemini_client():
    assert resolve_image_provider("gemini-3-pro-image-preview") is None


# ---------------------------------------------------------------------------
# Agent integration: _generate_images consumes the provider stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_generate_images_collects_finals_and_narrative():
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = "gemini-3-pro-image-preview"
    agent.default_aspect_ratio = "1:1"
    agent.max_images = 2
    agent.recorder = None
    agent.gemini_client = _gemini_client([_gemini_chunk(_image_part(b"img"), _text_part("A fox."))])

    images, narrative = await agent._generate_images("enhanced prompt", "draw a fox")

    assert narrative == "A fox."
    assert images == [
        {
            "data": base64.b64encode(b"img").decode("utf-8"),
            "mime": "image/png",
            "prompt": "draw a fox",
            "model": "gemini-3-pro-image-preview",
            "aspect_ratio": "1:1",
        }
    ]


@pytest.mark.asyncio
async def test_agent_generate_images_returns_empty_without_provider():
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = "gemini-3-pro-image-preview"
    agent.default_aspect_ratio = "1:1"
    agent.max_images = 1
    agent.recorder = None
    agent.gemini_client = None

    assert await agent._generate_images("p", "p") == ([], "")


@pytest.mark.asyncio
async def test_agent_generate_images_publishes_previews():
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent
    from app.ai.image_generation import use_image_preview_emitter

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = "gemini-3-pro-image-preview"
    agent.default_aspect_ratio = "1:1"
    agent.max_images = 1
    agent.recorder = None
    agent.gemini_client = _gemini_client([_gemini_chunk(_image_part(b"img"))])

    published: list[dict] = []
    with use_image_preview_emitter(published.append):
        await agent._generate_images("enhanced", "original")

    assert len(published) == 1
    assert published[0]["status"] == "final"
    assert published[0]["image_index"] == 0
    assert published[0]["data_b64"] == base64.b64encode(b"img").decode("utf-8")
