"""A model reply carrying no answer text, and what the turn does about it.

Reported as "Error: No response generated": `_final_text` returns `""`,
`PublicContentPolicy` raises `empty_public_content`, and the node reported it
`retriable=False` -- giving a transient provider outcome the same disposition
as "the model invented an artifact id", which is a contract violation that
really is final.

**Why the model returns no text is not established.** No recovery is attempted
here on purpose: a retry would make the symptom disappear while destroying the
evidence, and the leading hypothesis is that it shares a cause with the search
budget storm fixed alongside it. What the turn does instead is fail retriably
and log what came back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.schemas import AgentType
from app.ai.workflow.finalization import OutputValidationError, PublicContentPolicy
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
    _final_text,
)
from tests.test_specialist_subgraph_execution import scripted_model

pytestmark = pytest.mark.usefixtures("disable_langsmith_tracing")

REASONING_ONLY = AIMessage(content=[{"type": "reasoning", "reasoning": "thinking hard"}])


@pytest.fixture
def disable_langsmith_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def _factory(model, tools=None):
    definition = SpecialistDefinition(
        agent_id="chat_agent",
        agent_type=AgentType.CHAT,
        model_config_key="chat",
        system_prompt_factory=lambda request: "You are a helpful assistant.",
        tool_factory=lambda request: list(tools or []),
        output_policy_ids=("public_content",),
    )
    return SpecialistFactory(
        definitions={"chat_agent": definition},
        runtime_model_resolver=SimpleNamespace(
            resolve_runtime_config=lambda *a, **k: SimpleNamespace(
                agent_key="chat",
                provider="gemini",
                model="gemini-3-flash-preview",
                temperature=1.0,
                api_key="key",
                key_source="user",
                source="default",
                warnings=[],
                capabilities={},
                fallback_config=None,
            )
        ),
        model_factory=SimpleNamespace(create_model_from_runtime=lambda config, **kw: model),
        usage_recorder=None,
        settings=SimpleNamespace(
            generation_hard_model_calls_per_epoch=6,
            generation_soft_model_calls_per_epoch=5,
            generation_hard_tool_calls_per_epoch=6,
            generation_soft_tool_calls_per_epoch=5,
        ),
    )


def _request(**overrides):
    payload = {
        "agent_id": "chat_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "persona": None,
        "model_request": None,
        "messages": [HumanMessage(content="what is 2+2?")],
        "history": [],
        "state": {},
    }
    payload.update(overrides)
    return SpecialistRequest(**payload)


def test_a_thought_part_is_never_published_as_the_answer():
    """Why ``_final_text`` defers to ``BaseMessage.text``.

    The hand-rolled block walk it replaced collected any block carrying a
    ``text`` key, so a Gemini thought part shaped
    ``{"type": "thinking", "text": ...}`` reached the user as the assistant's
    answer. langchain-core knows which blocks are reasoning.
    """
    thought = AIMessage(content=[{"type": "thinking", "text": "internal reasoning"}])

    assert _final_text([thought]) == ""
    assert _final_text([REASONING_ONLY]) == ""
    assert _final_text([AIMessage(content=[{"type": "text", "text": "answer"}])]) == "answer"
    assert _final_text([AIMessage(content="plain")]) == "plain"


def test_the_answer_is_the_last_message_that_said_something():
    """Intermediate tool-calling turns carry no answer and are skipped."""
    produced = [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "noop", "args": {}}]),
        AIMessage(content="It is 4."),
    ]

    assert _final_text(produced) == "It is 4."


async def test_a_textless_reply_is_not_retried_and_logs_what_came_back(caplog):
    """No recovery: one model call, and the evidence kept."""
    model = scripted_model([REASONING_ONLY])

    with caplog.at_level("WARNING", logger="app.ai.workflow.specialists"):
        outcome = await _factory(model).invoke(_request())

    assert model.call_count == 1, "a retry would hide the cause"
    assert outcome.response.message.content == ""
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "no answer text" in logged
    assert "reasoning" in logged, "the log must carry the content, not just the symptom"


async def test_an_ordinary_answer_is_untouched():
    model = scripted_model([AIMessage(content="It is 4.")])

    outcome = await _factory(model).invoke(_request())

    assert outcome.response.message.content == "It is 4."
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_empty_public_content_is_retriable_but_a_contract_breach_is_not():
    """The two failures the policy registry raises are not the same kind.

    "The model said nothing" is a transient provider outcome; reporting it
    final dead-ended the turn at "Error: No response generated". "The model
    invented an artifact id" reproduces on every attempt and stays final.
    """
    from app.ai.schemas import AgentMessage, AgentResponse, MessageRole
    from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome

    outcome = ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
        ),
        provenance=OutcomeProvenance(),
    )

    with pytest.raises(OutputValidationError) as excinfo:
        await PublicContentPolicy().validate(outcome, {})

    assert excinfo.value.reason == "empty_public_content"
    assert excinfo.value.retriable is True
    assert OutputValidationError("unrecorded_artifact", "x").retriable is False


def test_produced_messages_excludes_the_carried_epoch():
    """The slice omitted ``carried_messages``, so input came back as output.

    ``_invocation_messages`` sends history + carried + messages; the slice
    counted only history + messages, so on a Continue the tail of the carried
    evidence was recorded as this epoch's own production -- and
    ``carry_messages`` then carried it forward again.
    """
    request = _request(
        history=[HumanMessage(content="h1")],
        carried_messages=[HumanMessage(content="carried-1"), HumanMessage(content="carried-2")],
    )
    produced = [AIMessage(content="the answer")]
    result = {
        "messages": [*request.history, *request.carried_messages, *request.messages, *produced]
    }

    assert SpecialistFactory._produced_messages(request, result) == produced
