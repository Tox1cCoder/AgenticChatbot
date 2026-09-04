"""Resume parity for streamed images, at the GRAPH level.

Every other automated "resume" check in this repo re-points a fake
``ai_service.resume_interrupted_execution_stream`` and therefore never touches
``MultiAgentWorkflow.resume_with_decisions_stream`` -- the code that actually
runs after a HITL interrupt.

These tests drive the REAL resume generator with:

* the REAL image specialist path (its own emitter/media bindings, not a test
  reimplementation of them),
* the REAL ``ImageGeneratorAgent._consume_image_stream`` over a deterministic
  fake provider,
* the REAL ``MediaDeliveryService`` / ``ImagePreviewPublisher`` / storage seam,
* the REAL custom-channel projection and public projector.

Previews used to travel on a side queue reached through a weak token in
checkpoint state; a resumed run resolved that token to a dead sink and the
resume path had to rebind a live one under it. They now ride the graph's own
custom channel, which a resumed run has for the same reason the original did --
so there is nothing left to install, and the tests below assert the previews
arrive rather than that the plumbing was installed.

Production changes that must make these fail: the image node dropping
``use_media_delivery_service`` or its preview emitter, the resume path dropping
the graph stream, or the custom channel no longer being projected.
"""

from __future__ import annotations

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


class _FakeImageSpecialistFactory:
    """Stands in for the compiled subgraph, but runs the REAL stream consumer.

    The point of the test is the emitter/media-delivery binding around the
    invocation, so the consumer must be genuine while the model is not.
    """

    agent_id = "image_generator_agent"

    def __init__(self) -> None:
        self.model_name = "gemini-3-pro-image-preview"
        self.default_aspect_ratio = "1:1"

    def register(self, definition) -> None:  # pragma: no cover - interface parity
        pass

    async def invoke(self, request):
        from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome

        outcome = await ImageGeneratorAgent._consume_image_stream(
            self,
            _FakeImageProvider(),
            ImageGenerationRequest(prompt="a cat", model=self.model_name),
            "a cat",
            handle=None,
        )
        return ResponseOutcome(
            agent_id=self.agent_id,
            response=AgentResponse(
                agent_type="image_generator",
                agent_id=self.agent_id,
                message=AgentMessage(role="assistant", content=outcome.narrative),
                metadata={"images": outcome.images},
            ),
            provenance=OutcomeProvenance(),
        )


async def _empty_history(*_args, **_kwargs):
    return []


def _build_workflow(*, store: _RecordingStore, checkpoint_values: dict):
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    workflow.chat_image_service = store
    workflow.image_generator_agent = _FakeImageSpecialistFactory()
    workflow._specialist_factory = workflow.image_generator_agent
    workflow.chat_agent = SimpleNamespace(_convert_history_to_langchain_messages=lambda history: [])
    workflow._get_conversation_history = _empty_history
    workflow.agents = {"image_generator_agent": object()}

    state_reads = {"count": 0}

    async def _aget_state(_config):
        """The paused checkpoint, then the resolved one.

        A real snapshot keeps reporting the interrupt after it is answered and
        marks the task with a ``result``; the second read here models that, so
        the post-resume check sees a turn that is no longer waiting rather than
        re-emitting the approval it just consumed.
        """
        state_reads["count"] += 1
        answered = state_reads["count"] > 1
        item = SimpleNamespace(
            id="int-1",
            value={"action_requests": [{"name": "generate_image", "tool_call_id": "call-1"}]},
        )
        return SimpleNamespace(
            next=() if answered else ("image_generator_agent",),
            tasks=(
                SimpleNamespace(
                    name="image_generator_agent",
                    id="task-1",
                    interrupts=(item,),
                    result={"response": "done"} if answered else None,
                ),
            ),
            interrupts=(item,),
            values=checkpoint_values,
        )

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

        # Stand in for LangGraph's custom channel: install a writer around the
        # node exactly as a real run does, then surface what it wrote as
        # ``("custom", payload)`` tuples.
        written: list[dict] = []
        import app.ai.graph as _graph_module

        monkeypatch_writer = _graph_module._graph_stream_writer
        _graph_module._graph_stream_writer = lambda: written.append
        try:
            outcome = await workflow.invoke_specialist_subgraph("image_generator_agent", node_state)
        finally:
            _graph_module._graph_stream_writer = monkeypatch_writer
        response = outcome.response

        for payload in written:
            yield ("custom", payload)

        chunk = SimpleNamespace(content=response.message.content, content_blocks=None)
        yield ("messages", (chunk, {"langgraph_node": "image_generator_agent"}))
        yield ("updates", {"image_generator_agent": {"messages": []}})
        # Only ``finalize`` publishes a response, so the resumed run must reach
        # it before the stream can complete.
        yield (
            "updates",
            {
                "finalize": {
                    "messages": [],
                    "response": response,
                    "final_agent_id": "image_generator_agent",
                    "execution_phase": "completed",
                }
            },
        )

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


def _checkpoint_values() -> dict:
    context: dict = {}
    return {
        "messages": [HumanMessage(content="draw a cat")],
        "active_agent_id": "image_generator_agent",
        "conversation_id": str(uuid4()),
        "user_id": str(uuid4()),
        "context": context,
    }


@pytest.mark.asyncio
async def test_resumed_graph_run_emits_early_image_reference(monkeypatch):
    """FR-IMG-008: a HITL-resumed run delivers the final image by protected
    reference BEFORE the narrative, exactly like a fresh run."""
    monkeypatch.setattr(settings, "enable_image_streaming", True)

    store = _RecordingStore()
    workflow = _build_workflow(store=store, checkpoint_values=_checkpoint_values())

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

    partials = [event for event in previews if (event.data or {}).get("status") == "partial"]
    assert partials, f"no transient partial preview on resume; order={types}"
    assert events.index(partials[0]) < events.index(finals[0]), (
        f"partial must precede the final reference; order={types}"
    )

    # ``image_generator_agent`` suppresses its narrative tokens by design (they
    # are the internal enhanced prompt), so the terminal event is what the
    # image must beat — FR-IMG-002's "before narrative completion".
    errors = [e for e in events if e.type == "error"]
    assert not errors, f"resumed run errored: {[e.data for e in errors]}"
    assert "complete" in types, f"resumed run never terminated; order={types}"
    assert events.index(finals[0]) < types.index("complete"), (
        "early delivery violated: the final image reference arrived at or "
        f"after stream completion. order={types}"
    )
    assert store.writes, "the resumed run never persisted the final image bytes"


@pytest.mark.asyncio
async def test_resumed_image_is_persisted_even_with_image_streaming_disabled(
    monkeypatch,
):
    """Durable persistence is independent of the transient preview flag: with
    streaming off the resumed run emits no preview but still stores the bytes."""
    monkeypatch.setattr(settings, "enable_image_streaming", False)

    store = _RecordingStore()
    workflow = _build_workflow(store=store, checkpoint_values=_checkpoint_values())

    events = await _collect(workflow)

    assert not [event for event in events if event.type == "image_preview"]
    assert store.writes, "final image bytes must persist even with previews disabled"
