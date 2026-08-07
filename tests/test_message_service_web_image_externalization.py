"""Selected remote images become protected references before persistence."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.services import message_service as message_service_module
from app.services.message_service import MessageService

IMAGE_ID = "image:tool:c1:0"


def _metadata(url: str = "https://img.example/a.jpg") -> dict:
    return {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": IMAGE_ID,
                "type": "image",
                "source": "web_search",
                "display_policy": "inline_only",
                "alt_text": "Example",
                "payload": {
                    "url": url,
                    "mime_type": "image/jpeg",
                    "source_url": "https://publisher.example/story",
                    "width": 640,
                    "height": 360,
                },
                "provenance": {"provider": "tavily"},
            },
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            },
        ],
        "rich_reference_warnings": [{"code": "stale", "id": IMAGE_ID}],
    }


def _service(web_image_service) -> MessageService:
    service = MessageService.__new__(MessageService)
    service.web_image_service = web_image_service
    return service


def _group_metadata() -> dict:
    return {
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "source": "image_search",
                "display_policy": "inline_only",
                "alt_text": "Images of a red panda",
                "payload": {
                    "items": [
                        {"url": "https://img.example/a.jpg", "mime_type": "image/jpeg"},
                        {"url": "https://img.example/b.jpg", "mime_type": "image/jpeg"},
                    ]
                },
                "provenance": {"provider": "brave_image_search"},
            }
        ]
    }


@pytest.mark.asyncio
async def test_selected_remote_image_becomes_protected_reference_without_fetch():
    image_reference_id = uuid4()
    web_images = AsyncMock()
    web_images.register.return_value = SimpleNamespace(id=image_reference_id)
    service = _service(web_images)
    conversation_id = uuid4()
    user_id = uuid4()
    metadata = _metadata()
    original = deepcopy(metadata)

    content, externalized = await service._externalize_remote_rich_images(
        f"Intro\n\n<!--rich:{IMAGE_ID}-->",
        metadata,
        conversation_id,
        user_id,
    )

    image = externalized["rich_items"][0]
    assert image["payload"]["url"] == f"/web-images/{image_reference_id}"
    assert image["payload"]["source_url"] == "https://publisher.example/story"
    assert image["provenance"] == {"provider": "tavily"}
    assert content.endswith(f"<!--rich:{IMAGE_ID}-->")
    assert metadata == original
    web_images.register.assert_awaited_once_with(
        conversation_id=conversation_id,
        user_id=user_id,
        upstream_url="https://img.example/a.jpg",
        expected_mime="image/jpeg",
        provider="tavily",
        # No visual verification ran for this image, so there are no verified
        # bytes to hand over and registration stays metadata-only.
        cached=None,
    )
    web_images.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_registration_failure_drops_only_image_and_exact_marker(caplog):
    web_images = AsyncMock()
    web_images.register.side_effect = RuntimeError("database secret")
    service = _service(web_images)
    metadata = _metadata()
    body = (
        f"Before\n\n<!--rich:{IMAGE_ID}-->\n\n"
        "<!--rich:widget:w-1-->\n\nAfter"
    )

    content, externalized = await service._externalize_remote_rich_images(
        body,
        metadata,
        uuid4(),
        uuid4(),
    )

    assert f"<!--rich:{IMAGE_ID}-->" not in content
    assert "<!--rich:widget:w-1-->" in content
    assert [item["id"] for item in externalized["rich_items"]] == ["widget:w-1"]
    assert externalized["rich_reference_warnings"] == []
    assert "database secret" not in caplog.text
    assert "code=web_image_reference_failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_payload",
    (
        {"url": "/chat-images/known", "mime_type": "image/png"},
        {"url": "/web-images/known", "mime_type": "image/png"},
        {"data": "QUJD", "mime_type": "image/png"},
    ),
)
async def test_inline_and_already_protected_images_are_not_registered(image_payload):
    web_images = AsyncMock()
    service = _service(web_images)
    metadata = _metadata()
    metadata["rich_items"][0]["payload"] = image_payload

    content, externalized = await service._externalize_remote_rich_images(
        f"<!--rich:{IMAGE_ID}-->", metadata, uuid4(), uuid4()
    )

    assert content == f"<!--rich:{IMAGE_ID}-->"
    assert externalized["rich_items"][0]["payload"] == image_payload
    web_images.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_insecure_remote_image_is_omitted_without_failing_text():
    web_images = AsyncMock()
    service = _service(web_images)
    metadata = _metadata("http://img.example/a.jpg")

    content, externalized = await service._externalize_remote_rich_images(
        f"Answer\n\n<!--rich:{IMAGE_ID}-->", metadata, uuid4(), uuid4()
    )

    assert content == "Answer"
    assert [item["id"] for item in externalized["rich_items"]] == ["widget:w-1"]
    web_images.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_optional_service_keeps_backward_compatible_metadata():
    service = _service(None)
    metadata = _metadata()
    body = f"Answer\n\n<!--rich:{IMAGE_ID}-->"

    content, externalized = await service._externalize_remote_rich_images(
        body, metadata, uuid4(), uuid4()
    )

    assert content == body
    assert externalized == metadata


@pytest.mark.asyncio
async def test_group_cells_each_become_protected_references():
    web_images = AsyncMock()
    web_images.register.side_effect = [
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
    ]
    service = _service(web_images)
    conversation_id = uuid4()
    user_id = uuid4()
    metadata = _group_metadata()
    content = "Body\n\n<!--rich:imagegroup:tool:c1-->\n"

    new_content, new_metadata = await service._externalize_remote_rich_images(
        content, metadata, conversation_id, user_id
    )

    urls = [c["url"] for c in new_metadata["rich_items"][0]["payload"]["items"]]
    assert all(u.startswith("/web-images/") for u in urls)
    assert len(set(urls)) == 2
    assert "<!--rich:imagegroup:tool:c1-->" in new_content


@pytest.mark.asyncio
async def test_one_failing_cell_keeps_the_group_and_its_marker():
    reference_id = uuid4()

    async def _register(**kwargs):
        if kwargs["upstream_url"].endswith("b.jpg"):
            raise RuntimeError("upstream rejected")
        return SimpleNamespace(id=reference_id)

    web_images = AsyncMock()
    web_images.register.side_effect = _register
    service = _service(web_images)
    content, externalized = await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )

    cells = externalized["rich_items"][0]["payload"]["items"]
    assert [c["url"] for c in cells] == [f"/web-images/{reference_id}"]
    assert "<!--rich:imagegroup:tool:c1-->" in content


@pytest.mark.asyncio
async def test_group_losing_every_cell_drops_the_item_and_marker():
    web_images = AsyncMock()
    web_images.register.side_effect = RuntimeError("upstream rejected")
    service = _service(web_images)
    content, externalized = await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n\nAfter",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )

    assert externalized["rich_items"] == []
    assert "<!--rich:imagegroup:tool:c1-->" not in content
    assert "After" in content


@pytest.mark.asyncio
async def test_final_selection_counter_records_surviving_images(monkeypatch):
    from app.services import message_service as module

    recorded = []

    class _Metrics:
        def record_final_selection(self, *, provider, count):
            recorded.append((provider, count))

    monkeypatch.setattr(module, "rich_image_metrics", _Metrics(), raising=False)
    web_images = AsyncMock()
    web_images.register.side_effect = [
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
    ]
    service = _service(web_images)
    await service._externalize_remote_rich_images(
        "Body\n\n<!--rich:imagegroup:tool:c1-->\n",
        _group_metadata(),
        uuid4(),
        uuid4(),
    )
    assert recorded == [("brave_image_search", 1)]


def _patch_response_builders(monkeypatch, body, metadata):
    monkeypatch.setattr(
        message_service_module,
        "extract_response_content",
        lambda *_args, **_kwargs: body,
    )
    monkeypatch.setattr(
        message_service_module,
        "finalize_article_content",
        lambda _response, content: content,
    )
    monkeypatch.setattr(
        message_service_module,
        "fix_markdown_code_blocks",
        lambda content: content,
    )
    monkeypatch.setattr(
        message_service_module,
        "build_bot_metadata",
        lambda *_args, **_kwargs: deepcopy(metadata),
    )


@pytest.mark.asyncio
async def test_completed_workflow_externalizes_before_message_create(monkeypatch):
    reference_id = uuid4()
    web_images = AsyncMock()
    web_images.register.return_value = SimpleNamespace(id=reference_id)
    service = _service(web_images)
    service.chat_image_service = None
    service.task_plan_service = None
    service._sync_response_plan_state = Mock(return_value=False)
    service._acreate_bot_response_message = AsyncMock(return_value="persisted")
    body = f"Answer\n\n<!--rich:{IMAGE_ID}-->"
    _patch_response_builders(monkeypatch, body, _metadata())
    response = SimpleNamespace(metadata={})

    result = await service._persist_completed_workflow_response(
        conversation_id=uuid4(),
        user_id=uuid4(),
        bot_response=response,
        sanitized_persona=None,
        workflow_request=None,
    )

    assert result == "persisted"
    persisted = service._acreate_bot_response_message.await_args.kwargs
    assert persisted["content"] == body
    assert persisted["metadata"]["rich_items"][0]["payload"]["url"] == (
        f"/web-images/{reference_id}"
    )


@pytest.mark.asyncio
async def test_resume_workflow_externalizes_before_message_create(monkeypatch):
    reference_id = uuid4()
    web_images = AsyncMock()
    web_images.register.return_value = SimpleNamespace(id=reference_id)
    service = _service(web_images)
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_args: None
    )
    response = SimpleNamespace(metadata={})
    service.ai_service = SimpleNamespace(resume_workflow=AsyncMock(return_value=response))
    service._sync_response_plan_state = Mock(return_value=False)
    service._acreate_bot_response_message = AsyncMock(return_value="persisted")
    service._compact_checkpoint_after_persist = AsyncMock()
    body = f"Answer\n\n<!--rich:{IMAGE_ID}-->"
    _patch_response_builders(monkeypatch, body, _metadata())

    result = await service.resume_workflow(uuid4(), uuid4())

    assert result == "persisted"
    persisted = service._acreate_bot_response_message.await_args.kwargs
    assert persisted["metadata"]["rich_items"][0]["payload"]["url"] == (
        f"/web-images/{reference_id}"
    )


@pytest.mark.asyncio
async def test_a_marker_for_an_unknown_item_is_removed_from_the_content():
    """A hallucinated marker must not reach the reader as an error caption.

    The renderer shows "rich item `<id>` is unavailable" for any marker whose
    id resolves to nothing. That is right for an item that existed and failed,
    but a model-invented id is pure noise the user should never see.
    """
    metadata = {
        "rich_items_version": 1,
        "rich_items": [],
        "rich_reference_warnings": [],
    }
    content = "T1's 2026 roster is settled.\n\n<!--rich:widget:t1_roster_2026-->\n\nMore text."

    updated_content, updated = await _service(AsyncMock())._externalize_remote_rich_images(
        content, metadata, uuid4(), uuid4()
    )

    assert "<!--rich:widget:t1_roster_2026-->" not in updated_content
    assert "T1's 2026 roster is settled." in updated_content
    assert "More text." in updated_content
    assert updated["rich_reference_warnings"] == []


@pytest.mark.asyncio
async def test_a_marker_for_a_real_item_survives():
    """Only unresolvable markers are stripped; a real one must still render."""
    web_images = AsyncMock()
    web_images.register.return_value = SimpleNamespace(id=uuid4())
    content = f"Body\n\n<!--rich:{IMAGE_ID}-->\n"

    updated_content, updated = await _service(web_images)._externalize_remote_rich_images(
        content, _metadata(), uuid4(), uuid4()
    )

    assert f"<!--rich:{IMAGE_ID}-->" in updated_content
    assert updated["rich_reference_warnings"] == []
