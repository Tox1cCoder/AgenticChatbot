from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agents.base_agent import BaseAgent
from app.ai.agents.canvas_agent import CanvasAgent, _extract_artifact, _strip_code_block
from app.ai.canvas_state import CanvasArtifactSnapshot
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_truncated_code_block_still_extracts_artifact():
    """A generation cut off before the closing fence (length cap, provider
    stop) must still yield an artifact — otherwise the raw code dump persists
    as chat content and no canvas renders (regression: 2026-07-02)."""

    text = (
        "Here is your calculator page.\n\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head><title>Calculator</title></head>\n"
        "  <body>\n"
        "    <button onclick=\"append('8')\">8</button>\n"
        '    <button onclick="append'
    )

    artifact = _extract_artifact(text)

    assert artifact is not None
    assert artifact["language"] == "html"
    assert artifact["title"] == "Calculator"
    assert artifact["truncated"] is True
    assert artifact["content"].startswith("<!DOCTYPE html>")
    assert artifact["content"].endswith('<button onclick="append')


def test_truncated_code_block_strip_keeps_description_only():
    text = "Here is your calculator page.\n\n```html\n<!DOCTYPE html>\n<body>partial"

    assert _strip_code_block(text) == "Here is your calculator page."


def test_unclosed_fence_with_no_code_is_not_an_artifact():
    assert _extract_artifact("Some text.\n\n```html\n") is None


def test_complete_code_block_is_not_marked_truncated():
    artifact = _extract_artifact("Intro.\n\n```html\n<!doctype html><body>ok</body>\n```")

    assert artifact is not None
    assert "truncated" not in artifact


def test_canvas_artifact_response_shape_omits_editable_hint():
    artifact = _extract_artifact(
        """
Built a small page.

```html
<!doctype html>
<html>
  <head><title>Demo Canvas</title></head>
  <body>Hello</body>
</html>
```
"""
    )

    assert artifact == {
        "content": (
            "<!doctype html>\n"
            "<html>\n"
            "  <head><title>Demo Canvas</title></head>\n"
            "  <body>Hello</body>\n"
            "</html>"
        ),
        "language": "html",
        "title": "Demo Canvas",
    }


def _snapshot(content: str = "<!doctype html><html><body>ORIGINAL</body></html>"):
    return CanvasArtifactSnapshot(
        artifact_id="canvas:main",
        revision=4,
        content=content,
        language="html",
        title="Working Canvas",
        message_id="message-4",
        sequence=8,
        is_latest_assistant=True,
    )


def _model_response(content: str) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CANVAS,
        agent_id="canvas_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


@pytest.mark.asyncio
async def test_canvas_edit_injects_source_and_emits_next_revision(monkeypatch):
    captured = {}

    async def fake_base_invoke(self, messages, **kwargs):
        captured["messages"] = messages
        return _model_response(
            "Updated the accent colour.\n\n"
            '```html\n<!doctype html><html><body class="blue">UPDATED</body></html>\n```'
        )

    monkeypatch.setattr(BaseAgent, "invoke_model_with_history", fake_base_invoke)
    agent = CanvasAgent.__new__(CanvasAgent)

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="make the accent blue")],
        conversation_history=[],
        persona=None,
        previous_artifact=_snapshot(),
    )

    assert "ORIGINAL" in captured["messages"][-2].content
    assert captured["messages"][-1].content == "make the accent blue"
    assert response.metadata["canvas_artifact"] == {
        "artifact_id": "canvas:main",
        "revision": 5,
        "operation": "update",
        "content": '<!doctype html><html><body class="blue">UPDATED</body></html>',
        "language": "html",
        "title": "Canvas",
    }
    assert response.metadata["canvas_update"] == {
        "status": "updated",
        "artifact_id": "canvas:main",
        "base_revision": 4,
        "revision": 5,
    }


@pytest.mark.asyncio
async def test_canvas_edit_does_not_replace_valid_source_with_truncated_output(monkeypatch):
    async def fake_base_invoke(self, messages, **kwargs):
        return _model_response("Trying the edit.\n\n```html\n<html><body>PARTIAL")

    monkeypatch.setattr(BaseAgent, "invoke_model_with_history", fake_base_invoke)
    agent = CanvasAgent.__new__(CanvasAgent)

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="change it")],
        conversation_history=[],
        persona=None,
        previous_artifact=_snapshot(),
    )

    assert "canvas_artifact" not in response.metadata
    assert response.metadata["canvas_update"]["status"] == "failed"
    assert response.metadata["canvas_update"]["reason"] == "truncated_output"
    assert response.metadata["canvas_update"]["revision"] == 4


@pytest.mark.asyncio
async def test_canvas_edit_identical_output_is_unchanged(monkeypatch):
    previous = _snapshot()

    async def fake_base_invoke(self, messages, **kwargs):
        return _model_response(f"No changes needed.\n\n```html\n{previous.content}\n```")

    monkeypatch.setattr(BaseAgent, "invoke_model_with_history", fake_base_invoke)
    agent = CanvasAgent.__new__(CanvasAgent)

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="keep it as-is")],
        conversation_history=[],
        persona=None,
        previous_artifact=previous,
    )

    assert "canvas_artifact" not in response.metadata
    assert response.metadata["canvas_update"] == {
        "status": "unchanged",
        "artifact_id": "canvas:main",
        "base_revision": 4,
        "revision": 4,
    }


@pytest.mark.asyncio
async def test_first_canvas_generation_emits_create_revision(monkeypatch):
    async def fake_base_invoke(self, messages, **kwargs):
        return _model_response("Built it.\n\n```html\n<html><body>NEW</body></html>\n```")

    monkeypatch.setattr(BaseAgent, "invoke_model_with_history", fake_base_invoke)
    agent = CanvasAgent.__new__(CanvasAgent)

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="build a page")],
        conversation_history=[],
        persona=None,
    )

    assert response.metadata["canvas_artifact"]["artifact_id"] == "canvas:main"
    assert response.metadata["canvas_artifact"]["revision"] == 1
    assert response.metadata["canvas_artifact"]["operation"] == "create"
    assert response.metadata["canvas_update"]["status"] == "updated"
