"""Canvas output must survive the real specialist loop and frontend projections."""

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agents.canvas_agent import CanvasAgent, build_canvas_specialist_definition
from app.ai.canvas_state import CanvasArtifactSnapshot
from app.ai.graph import MultiAgentWorkflow
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage
from app.services.event_streaming.ai_sdk_projection import scrub_legacy_metadata
from tests.test_specialist_subgraph_execution import (
    _factory,
    _request,
    disable_langsmith_tracing,  # noqa: F401
    scripted_model,
)

pytestmark = pytest.mark.usefixtures("disable_langsmith_tracing")


async def _run_canvas(text, previous=None):
    agent = CanvasAgent.__new__(CanvasAgent)
    definition = replace(
        build_canvas_specialist_definition(agent),
        system_prompt_factory=lambda request: agent._get_base_system_prompt(),
        tool_factory=lambda request: [],
        agent=None,
    )
    model = scripted_model([AIMessage(content=text)])
    request = _request(
        agent_id="canvas_agent",
        messages=[HumanMessage(content="Build or update my page")],
        extras={"previous_artifact": previous},
    )
    outcome = await _factory(model, definitions={"canvas_agent": definition}).invoke(request)
    return outcome.response, model


async def test_canvas_subgraph_produces_renderable_metadata_for_both_frontends(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    html = "<!doctype html><html><title>Demo</title><body>Hello</body></html>"
    response, _ = await _run_canvas(f"Built your page.\n\n```html\n{html}\n```")

    assert response.message.content == "Built your page."
    artifact = response.metadata["canvas_artifact"]
    assert artifact["content"] == html
    assert artifact["revision"] == 1
    MultiAgentWorkflow.__new__(MultiAgentWorkflow)._merge_tool_artifacts(
        {"context": {"inline_rich_response_v1": True}}, response
    )
    metadata = build_bot_metadata(WorkflowResponse(
        agent_type="canvas", agent_id="canvas_agent",
        message=WorkflowResponseMessage(role="assistant", content=response.message.content),
        metadata=response.metadata,
    ))
    assert metadata["canvas_artifact"]["content"] == html  # Streamlit
    wire = scrub_legacy_metadata(metadata)
    canvas = next(item for item in wire["rich_items"] if item["type"] == "canvas_artifact")
    assert canvas["payload"]["content"] == html  # AI SDK


@pytest.mark.parametrize("output,status", [
    ("Changed.\n```html\n<html>UPDATED</html>\n```", "updated"),
    ("Same.\n```html\n<html>ORIGINAL</html>\n```", "unchanged"),
    ("Partial.\n```html\n<html>PARTIAL", "failed"),
    ("No code this time.", "failed"),
])
async def test_canvas_subgraph_edits_receive_source_and_preserve_revision_contract(output, status):
    previous = CanvasArtifactSnapshot(
        artifact_id="canvas:main", revision=4, content="<html>ORIGINAL</html>",
        language="html", title="Original", message_id="m4", sequence=8,
        is_latest_assistant=True,
    )
    response, model = await _run_canvas(output, previous)

    source_messages = [m for m in model.model_inputs[0] if "<current_canvas" in str(m.content)]
    assert len(source_messages) == 1
    assert isinstance(source_messages[0], HumanMessage)
    assert "<html>ORIGINAL</html>" in source_messages[0].content
    assert model.model_inputs[0][-1].content == "Build or update my page"
    assert response.metadata["canvas_update"]["status"] == status
    assert response.metadata["canvas_update"]["revision"] == (5 if status == "updated" else 4)
    if status == "updated":
        assert response.metadata["canvas_artifact"]["content"] == "<html>UPDATED</html>"
    else:
        assert "canvas_artifact" not in response.metadata
