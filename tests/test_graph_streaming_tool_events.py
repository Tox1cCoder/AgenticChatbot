from langchain_core.messages import AIMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow


def test_tool_end_events_include_every_tool_message_from_one_node_update():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    emitted: set[str] = set()
    node_state = {
        "context": {
            "tool_render_results": {
                "chunk-call": {"type": "text", "text": "chunk evidence"},
                "doc-call": {"type": "text", "text": "full document text"},
            }
        },
        "messages": [
            ToolMessage(
                content="SEARCH RESULTS:\n\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
            ToolMessage(
                content="DOCUMENT CONTENT:\n\nfull document text",
                tool_call_id="doc-call",
                name="search_documents",
            ),
        ],
    }

    events = list(
        workflow._tool_end_events_from_node_state(
            node_state=node_state,
            last_state_values=node_state,
            emitted_tool_result_ids=emitted,
        )
    )

    assert [event["tool_call_id"] for event in events] == ["chunk-call", "doc-call"]
    assert events[0]["render"]["text"] == "chunk evidence"
    assert events[1]["render"]["text"] == "full document text"


def test_tool_end_events_are_deduplicated_by_tool_call_id():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    emitted = {"chunk-call"}
    node_state = {
        "messages": [
            ToolMessage(
                content="SEARCH RESULTS:\n\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
        ],
    }

    events = list(
        workflow._tool_end_events_from_node_state(
            node_state=node_state,
            last_state_values=node_state,
            emitted_tool_result_ids=emitted,
        )
    )

    assert events == []


def test_tool_end_events_ignore_prior_turn_tool_messages():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    emitted: set[str] = set()
    node_state = {
        "messages": [
            ToolMessage(
                content="old result from a previous turn",
                tool_call_id="old-call",
                name="old_tool",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "new-call",
                        "name": "search_documents",
                        "args": {},
                    }
                ],
            ),
            ToolMessage(
                content="current result",
                tool_call_id="new-call",
                name="search_documents",
            ),
        ],
    }

    events = list(
        workflow._tool_end_events_from_node_state(
            node_state=node_state,
            last_state_values=node_state,
            emitted_tool_result_ids=emitted,
        )
    )

    assert [event["tool_call_id"] for event in events] == ["new-call"]
