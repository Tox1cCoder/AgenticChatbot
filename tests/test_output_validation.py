"""Output policies are selected from server-owned provenance.

Which contracts a response must satisfy is decided by what actually produced
it — the evidence, artifacts, and images runtime code recorded — not by the
final agent's name. An agent name can change through a handoff; what produced
the answer cannot.
"""

from __future__ import annotations

import pytest

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome
from app.ai.workflow.finalization import (
    POLICY_REGISTRY,
    OutputValidationError,
    OutputValidator,
    select_policies,
)


def _outcome(agent_id="chat_agent", content="an answer", **provenance) -> ResponseOutcome:
    return ResponseOutcome(
        agent_id=agent_id,
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id=agent_id,
            message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        ),
        provenance=OutcomeProvenance(**provenance),
    )


# ----------------------------------------------------------------------
# policy selection
# ----------------------------------------------------------------------


def test_evidence_presence_always_activates_grounding():
    policies = select_policies(_outcome(evidence=({"evidence_id": "E1"},)))
    assert "rag_grounding" in policies


def test_a_declared_policy_activates_even_with_no_evidence():
    """An empty-evidence RAG result still declares — and must run — grounding."""
    policies = select_policies(_outcome(agent_id="rag_agent", output_policy_ids=("rag_grounding",)))
    assert "rag_grounding" in policies


def test_artifacts_and_images_activate_their_provenance_validators():
    policies = select_policies(
        _outcome(artifacts=({"artifact_id": "a-1"},), images=({"image_id": "i-1"},))
    )
    assert "artifact_provenance" in policies
    assert "image_delivery" in policies


def test_public_content_is_always_selected():
    assert "public_content" in select_policies(_outcome())


def test_policies_are_not_chosen_by_agent_name_alone():
    """A canvas-named agent with no canvas provenance gets no canvas contract."""
    plain = select_policies(_outcome(agent_id="canvas_agent"))
    declared = select_policies(
        _outcome(agent_id="chat_agent", output_policy_ids=("canvas_contract",))
    )
    assert "canvas_contract" not in plain
    assert "canvas_contract" in declared


def test_every_selected_policy_exists_in_the_registry():
    policies = select_policies(
        _outcome(
            output_policy_ids=("rag_grounding", "canvas_contract", "tool_message_pairing"),
            artifacts=({"artifact_id": "a"},),
            images=({"image_id": "i"},),
        )
    )
    for policy_id in policies:
        assert policy_id in POLICY_REGISTRY


def test_unknown_declared_policy_is_rejected_rather_than_ignored():
    with pytest.raises(OutputValidationError):
        select_policies(_outcome(output_policy_ids=("made_up_policy",)))


# ----------------------------------------------------------------------
# the policies themselves
# ----------------------------------------------------------------------


async def test_public_content_rejects_an_empty_response():
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(_outcome(content="   "), {})
    assert exc.value.reason == "empty_public_content"


async def test_public_content_accepts_an_artifact_only_response():
    outcome = _outcome(content="", artifacts=({"artifact_id": "a-1"},))
    assert await OutputValidator().validate(outcome, {}) is not None


async def test_artifact_provenance_rejects_an_unrecorded_artifact_id():
    """Model text cannot introduce an artifact runtime code never captured."""
    outcome = ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="see chart"),
            tool_artifacts=[{"artifact_id": "forged-1"}],
        ),
        provenance=OutcomeProvenance(artifacts=({"artifact_id": "real-1"},)),
    )
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(outcome, {})
    assert exc.value.reason == "unrecorded_artifact"


async def test_grounding_policy_rejects_a_citation_to_unknown_evidence():
    outcome = _outcome(
        content="Revenue rose [E9].",
        evidence=({"evidence_id": "E1"},),
        output_policy_ids=("rag_grounding",),
    )
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(outcome, {})
    assert exc.value.reason == "unknown_evidence_id"


async def test_grounding_policy_accepts_citations_it_can_resolve():
    outcome = _outcome(
        content="Revenue rose [E1].",
        evidence=({"evidence_id": "E1"},),
        output_policy_ids=("rag_grounding",),
    )
    assert await OutputValidator().validate(outcome, {}) is not None


async def test_tool_message_pairing_rejects_an_orphan_tool_result():
    from langchain_core.messages import AIMessage, ToolMessage

    outcome = _outcome(
        output_policy_ids=("tool_message_pairing",),
        private_messages=(
            AIMessage(content="", id="a-1"),
            ToolMessage(content="result", tool_call_id="never-requested", id="t-1"),
        ),
    )
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(outcome, {})
    assert exc.value.reason == "unpaired_tool_message"


async def test_tool_message_pairing_accepts_a_matched_pair():
    from langchain_core.messages import AIMessage, ToolMessage

    outcome = _outcome(
        output_policy_ids=("tool_message_pairing",),
        private_messages=(
            AIMessage(content="", id="a-1", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
            ToolMessage(content="result", tool_call_id="c1", id="t-1"),
        ),
    )
    assert await OutputValidator().validate(outcome, {}) is not None


async def test_validation_records_which_policies_ran():
    outcome = _outcome(evidence=({"evidence_id": "E1"},), content="grounded [E1]")
    validated = await OutputValidator().validate(outcome, {})
    assert "rag_grounding" in validated.provenance.output_policy_ids
    assert "public_content" in validated.provenance.output_policy_ids


async def test_a_non_server_owned_outcome_is_refused():
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate({"content": "not an outcome"}, {})
    assert exc.value.reason == "outcome_not_server_owned"


# ----------------------------------------------------------------------
# planning synthesis
# ----------------------------------------------------------------------


async def test_planning_synthesis_with_rag_evidence_is_revalidated():
    """A synthesis can distort an otherwise grounded worker result."""
    outcome = ResponseOutcome(
        agent_id="planning_agent",
        response=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="Unsupported [E9]"),
        ),
        provenance=OutcomeProvenance(
            output_policy_ids=("rag_grounding",),
            evidence=({"evidence_id": "E1"},),
        ),
    )
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(outcome, {})
    assert exc.value.reason == "unknown_evidence_id"


# ----------------------------------------------------------------------
# a claim selects its own policy
# ----------------------------------------------------------------------
#
# Selecting only on what the runtime recorded leaves the worst case unchecked:
# a response that publishes an artifact, an image, or a citation when the
# runtime recorded *nothing* skips the very policy that exists to catch it.
# Empty provenance is not "nothing to verify" — against a response that claims
# something, it is the strongest evidence there is that the claim is invented.


def _claiming_outcome(*, content="an answer", artifacts=None, images=None):
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={"images": images} if images else {},
    )
    if artifacts is not None:
        response.tool_artifacts = artifacts
    return ResponseOutcome(
        agent_id="chat_agent", response=response, provenance=OutcomeProvenance()
    )


def test_a_citation_selects_grounding_even_with_no_recorded_evidence():
    assert "rag_grounding" in select_policies(
        _claiming_outcome(content="Revenue rose [E1].")
    )


def test_a_published_artifact_selects_provenance_with_nothing_recorded():
    assert "artifact_provenance" in select_policies(
        _claiming_outcome(artifacts=[{"artifact_id": "a1"}])
    )


def test_a_published_image_selects_delivery_with_nothing_recorded():
    assert "image_delivery" in select_policies(
        _claiming_outcome(images=[{"image_id": "i1"}])
    )


def test_a_plain_answer_still_selects_only_the_content_policy():
    """Selection widens for claims, not for everything."""
    assert select_policies(_claiming_outcome()) == ("public_content",)


async def test_a_citation_with_nothing_retrieved_is_rejected():
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(
            _claiming_outcome(content="Revenue rose [E9]."), {}
        )
    assert exc.value.reason == "unknown_evidence_id"


async def test_an_artifact_the_runtime_never_recorded_is_rejected():
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(
            _claiming_outcome(artifacts=[{"artifact_id": "forged"}]), {}
        )
    assert exc.value.reason == "unrecorded_artifact"


async def test_an_image_the_runtime_never_recorded_is_rejected():
    with pytest.raises(OutputValidationError) as exc:
        await OutputValidator().validate(
            _claiming_outcome(images=[{"image_id": "forged"}]), {}
        )
    assert exc.value.reason == "unrecorded_image"
