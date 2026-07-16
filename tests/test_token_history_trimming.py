from app.ai.schemas import AgentMessage, MessageRole
from app.ai.token_instrumentation import estimate_agent_message_tokens, trim_history_to_budget


def _message(role: MessageRole, content: str) -> AgentMessage:
    return AgentMessage(role=role, content=content)


def test_message_limit_keeps_only_complete_turns() -> None:
    history = [
        _message(MessageRole.USER, "old question"),
        _message(MessageRole.ASSISTANT, "old answer"),
        _message(MessageRole.USER, "recent question"),
        _message(MessageRole.ASSISTANT, "recent answer"),
    ]

    trimmed = trim_history_to_budget(history, max_messages=3)

    assert [(message.role, message.content) for message in trimmed] == [
        (MessageRole.USER, "recent question"),
        (MessageRole.ASSISTANT, "recent answer"),
    ]


def test_token_limit_never_keeps_only_part_of_a_turn() -> None:
    history = [
        _message(MessageRole.USER, "old question"),
        _message(MessageRole.ASSISTANT, "old answer"),
        _message(MessageRole.USER, "recent question"),
        _message(MessageRole.ASSISTANT, "recent answer"),
    ]
    recent_turn_budget = sum(estimate_agent_message_tokens(message) for message in history[-2:])

    trimmed = trim_history_to_budget(history, max_tokens=recent_turn_budget)

    assert [message.content for message in trimmed] == ["recent question", "recent answer"]


def test_message_limit_discards_an_incomplete_leading_fragment() -> None:
    history = [
        _message(MessageRole.ASSISTANT, "orphaned answer"),
        _message(MessageRole.USER, "recent question"),
        _message(MessageRole.ASSISTANT, "recent answer"),
    ]

    trimmed = trim_history_to_budget(history, max_messages=3)

    assert [message.content for message in trimmed] == ["recent question", "recent answer"]
