from app.ai.utils import apply_hitl_decisions


def test_hitl_respond_skips_tool_and_returns_human_response_as_tool_message():
    tool_calls = [
        {
            "id": "call-1",
            "name": "ask_user",
            "args": {"question": "Which file should I use?"},
        }
    ]
    decisions = [
        {
            "type": "respond",
            "tool_call_id": "call-1",
            "action": "ask_user",
            "args": {"response": "Use the quarterly report."},
        }
    ]

    approved, feedback = apply_hitl_decisions(tool_calls, decisions)

    assert approved == []
    assert feedback == {"call-1": "Use the quarterly report."}
