from app.services.event_streaming.events import make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
    flush_answer_text,
)


def _projector() -> GraphPublicStreamProjector:
    return GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **kwargs: iter(()),
        suppress_internal_stream_chunks=True,
    )


def test_web_tool_switches_answer_stream_to_terminal_buffering() -> None:
    projector = _projector()
    context = StreamProjectionContext()

    tool_events = list(
        projector.map_event(
            make_event(
                "tool_call_available",
                sequence=1,
                tool_call_id="call-1",
                tool_name="web_search",
                data={"args": {"query": "current release"}},
            ),
            context,
        )
    )
    answer_events = list(
        projector.map_event(
            make_event("message_delta", sequence=2, data={"text": "uncited draft"}),
            context,
        )
    )

    assert tool_events[0].type == "tool_call_available"
    assert context.requires_web is True
    assert answer_events == []
    assert list(flush_answer_text(context)) == []


def test_non_web_answer_keeps_normal_streaming() -> None:
    events = list(
        _projector().map_event(
            make_event("message_delta", sequence=1, data={"text": "ordinary answer"}),
            StreamProjectionContext(),
        )
    )

    assert events[0].data["text"] == "ordinary answer"
