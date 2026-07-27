"""T005 resume parity at the GRAPH level (closes the deferred verification gap).

Every other automated "resume" check in this repo re-points a fake
``ai_service.resume_interrupted_execution_stream`` and therefore never touches
``MultiAgentWorkflow.resume_with_decisions_stream`` — the code that actually
installs the request-scoped media sink after a HITL interrupt. The T005 review
confirmed that wiring only by inspection and deferred runtime proof to a manual
acceptance step.

These tests drive the REAL resume generator with:

* a checkpointed ``subagent_event_sink_token`` whose original sink has died
  (the exact post-resume condition the rebind exists for),
* the REAL ``_image_generator_node`` (its own emitter/media bindings, not a
  test reimplementation of them),
* the REAL ``ImageGeneratorAgent._consume_image_stream`` over a deterministic
  fake provider,
* the REAL ``MediaDeliveryService`` / ``ImagePreviewPublisher`` / storage seam,
* the REAL ``stream_with_subagent_events`` merge and public projector.

Production changes that must make these fail:
``resume_with_decisions_stream`` dropping ``rebind_subagent_event_sink`` (or its
no-token ``Command(update=...)`` fallback), dropping the
``stream_with_subagent_events`` merge, or ``_image_generator_node`` dropping
``use_media_delivery_service``.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.ai import graph as graph_module
from app.ai.agents.image_generator_agent import ImageGeneratorAgent
from app.ai.image_generation.models import (
    ImageFinal,
    ImageGenerationRequest,
    ImagePartial,
    NarrativeDelta,
)
from app.ai.schemas import AgentMessage, AgentResponse, InterruptDecision
from app.core.config import settings
from app.services.event_streaming.subagents import (
    register_subagent_event_sink,
    resolve_subagent_event_sink,
)

_NARRATIVE = "Here is the cat you asked for."
_PARTIAL_B64 = "UEFSVElBTA"  # "PARTIAL"
_FINAL_B64 = "RklOQUxCWVRFUw"  # "FINALBYTES"
_IMAGE_ID = "11111111-2222-3333-4444-555555555555"


class _FakeImageProvider:
    """Deterministic provider: one partial, one final, then narrative."""

    async def stream_generate(self, request: ImageGenerationRequest):
        yield ImagePartial(index=0, data_b64=_PARTIAL_B64, mime="image/png", seq=1)
        yield ImageFinal(index=0, data_b64=_FINAL_B64, mime="image/png")
        yield NarrativeDelta(text=_NARRATIVE)


class _RecordingStore:
    """Minimal ChatImageStorageService seam: records writes, returns a ref."""

    def __init__(self) -> None:
        self.writes: list[dict] = []

    def store(self, *, conversation_id, user_id, mime, data_b64, name):
        self.writes.append(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "mime": mime,
                "name": name,
                "size": len(data_b64),
            }
        )
        return {
            "image_id": _IMAGE_ID,
            "url": f"/chat-images/{_IMAGE_ID}",
            "mime": mime,
            "name": name,
        }


class _FakeImageAgent:
    """Stands in for the wired agent, but runs the REAL stream consumer."""

    agent_id = "image_generator_agent"

    def __init__(self) -> None:
        self.model_name = "gemini-3-pro-image-preview"
        self.default_aspect_ratio = "1:1"

    async def invoke_model_with_history(self, *_args, **_kwargs) -> AgentResponse:
        outcome = await ImageGeneratorAgent._consume_image_stream(
            self,
            _FakeImageProvider(),
            ImageGenerationRequest(prompt="a cat", model=self.model_name),
            "a cat",
            handle=None,
        )
        return AgentResponse(
            agent_type="image_generator",
            agent_id=self.agent_id,
            message=AgentMessage(role="assistant", content=outcome.narrative),
            metadata={"images": outcome.images},
        )


def _dead_sink_token() -> str:
    """Register a sink, drop it, and confirm the token resolves to None.

    Reproduces the post-resume state: the checkpoint still carries the original
    run's token but the sink it pointed at was garbage-collected with that
    stream (the registry holds only a weak reference).
    """
    sink = graph_module.SubagentEventSink()
    token = register_subagent_event_sink(sink)
    del sink
    gc.collect()
    return token


def _build_workflow(*, store: _RecordingStore, checkpoint_values: dict):
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    workflow.chat_image_service = store
    workflow.image_generator_agent = _FakeImageAgent()
    workflow.agents = {}

    async def _aget_state(_config):
        return SimpleNamespace(next=("approval",), values=checkpoint_values)

    async def _astream(state, config=None, stream_mode=None):
        """Run the resumed image node, then emit the narrative deltas.

        ``state`` is the ``Command`` the resume generator built; the node input
        is the checkpointed state merged with any ``Command.update``, which is
        exactly how LangGraph feeds a resumed node.
        """
        node_state = dict(checkpoint_values)
        update = getattr(state, "update", None)
        if isinstance(update, dict):
            node_state.update(update)

        result = await workflow._image_generator_node(node_state)
        response = result["response"]

        chunk = SimpleNamespace(content=response.message.content, content_blocks=None)
        yield ("messages", (chunk, {"langgraph_node": "image_generator_agent"}))
        yield ("updates", {"image_generator_agent": {"messages": []}})

    workflow.graph = SimpleNamespace(aget_state=_aget_state, astream=_astream)
    return workflow


async def _collect(workflow) -> list:
    events = []
    async for event in workflow.resume_with_decisions_stream(
        "thread-1",
        [InterruptDecision(type="approve", tool_call_id="call-1")],
    ):
        events.append(event)
    return events


def _checkpoint_values(token: str | None) -> dict:
    context: dict = {}
    if token is not None:
        context["subagent_event_sink_token"] = token
    return {
        "messages": [HumanMessage(content="draw a cat")],
        "selected_agent": "image_generator_agent",
        "conversation_id": str(uuid4()),
        "user_id": str(uuid4()),
        "context": context,
    }


@pytest.mark.asyncio
async def test_resumed_graph_run_emits_early_image_reference(monkeypatch):
    """FR-IMG-008: a HITL-resumed run delivers the final image by protected
    reference BEFORE the narrative, exactly like a fresh run."""
    monkeypatch.setattr(settings, "enable_image_streaming", True)
    monkeypatch.setattr(settings, "auto_continue_enabled", False)

    token = _dead_sink_token()
    assert resolve_subagent_event_sink(token) is None, (
        "precondition: the checkpointed token must be dead before resume"
    )
    store = _RecordingStore()
    workflow = _build_workflow(store=store, checkpoint_values=_checkpoint_values(token))

    events = await _collect(workflow)
    types = [event.type for event in events]

    previews = [event for event in events if event.type == "image_preview"]
    assert previews, (
        "resume parity gap: no image_preview reached the public stream from a "
        "REAL graph resume. The resumed image node could not resolve a live "
        f"sink. events={types}"
    )

    finals = [
        event
        for event in previews
        if (event.data or {}).get("delivery", {}).get("kind") == "reference"
    ]
    assert finals, (
        "the final image was not delivered by protected reference on resume; "
        f"preview payloads={[event.data for event in previews]}"
    )
    assert finals[0].data["delivery"]["url"] == f"/chat-images/{_IMAGE_ID}"
    assert finals[0].data["status"] == "final"
    assert finals[0].data["schema_version"] == 2
    assert "data_b64" not in finals[0].data["delivery"], (
        "final delivery must be by reference, never inline base64"
    )

    partials = [
        event
        for event in previews
        if (event.data or {}).get("status") == "partial"
    ]
    assert partials, f"no transient partial preview on resume; order={types}"
    assert events.index(partials[0]) < events.index(finals[0]), (
        f"partial must precede the final reference; order={types}"
    )

    # ``image_generator_agent`` suppresses its narrative tokens by design (they
    # are the internal enhanced prompt), so the terminal event is what the
    # image must beat — FR-IMG-002's "before narrative completion".
    assert "complete" in types, f"resumed run never terminated; order={types}"
    assert events.index(finals[0]) < types.index("complete"), (
        "early delivery violated: the final image reference arrived at or "
        f"after stream completion. order={types}"
    )
    assert store.writes, "the resumed run never persisted the final image bytes"


@pytest.mark.asyncio
async def test_resumed_graph_run_without_persisted_token_still_streams_image(
    monkeypatch,
):
    """A checkpoint written before sink tokens were persisted must still get a
    sink — the resume path injects a fresh token through ``Command(update=)``."""
    monkeypatch.setattr(settings, "enable_image_streaming", True)
    monkeypatch.setattr(settings, "auto_continue_enabled", False)

    store = _RecordingStore()
    workflow = _build_workflow(store=store, checkpoint_values=_checkpoint_values(None))

    events = await _collect(workflow)
    previews = [event for event in events if event.type == "image_preview"]
    assert previews, (
        "a resumed run whose checkpoint carries no sink token got no image "
        f"preview; events={[event.type for event in events]}"
    )
    assert any(
        (event.data or {}).get("delivery", {}).get("kind") == "reference"
        for event in previews
    )


@pytest.mark.asyncio
async def test_resumed_image_is_persisted_even_with_image_streaming_disabled(
    monkeypatch,
):
    """Durable persistence is independent of the transient preview flag: with
    streaming off the resumed run emits no preview but still stores the bytes."""
    monkeypatch.setattr(settings, "enable_image_streaming", False)
    monkeypatch.setattr(settings, "auto_continue_enabled", False)

    store = _RecordingStore()
    token = _dead_sink_token()
    workflow = _build_workflow(store=store, checkpoint_values=_checkpoint_values(token))

    events = await _collect(workflow)

    assert not [event for event in events if event.type == "image_preview"]
    assert store.writes, "final image bytes must persist even with previews disabled"
