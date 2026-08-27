"""Grounding reports; it does not rewrite. That is what lets RAG stream.

The evidence set is fixed before the first answer token is generated, so
"does this citation refer to something we retrieved?" is answerable the moment
a marker completes. That check is streamable, and it is the one that protects
the reader.

Citation *density* — how many claims carry a citation — is not answerable until
the answer is finished. Enforcing it meant a whole-answer decision: regenerate
the text, or replace it with a canned abstention. A draft that may be replaced
wholesale cannot be streamed, so density enforcement, not citation checking,
was what forced RAG answers to be withheld.

So density becomes a recorded finding rather than a rewrite. The model's prose
reaches the reader as written; grounding removes citations the server cannot
vouch for and appends the source list it owns.
"""

from __future__ import annotations

from types import MappingProxyType
from uuid import UUID

from app.services.rag_evidence import EvidencePack, EvidenceRecord
from app.services.rag_grounding import (
    GroundedAnswer,
    GroundedAnswerGate,
    GroundedClaim,
    render_grounded_answer,
)


def _evidence(*evidence_ids: str) -> EvidencePack:
    return EvidencePack(
        records=tuple(
            EvidenceRecord(
                evidence_id=evidence_id,
                document_id=UUID(int=1),
                chunk_id=UUID(int=index + 1),
                image_id=None,
                filename="manual.pdf",
                page_start=1,
                page_end=1,
                section_path=("Warranty",),
                modality="text",
                content=f"content for {evidence_id}",
                trace_metadata=MappingProxyType({}),
            )
            for index, evidence_id in enumerate(evidence_ids)
        ),
        token_count=0,
        omitted_count=0,
    )


def _answer(*evidence_ids: str, text: str = "The warranty lasts 24 months.") -> GroundedAnswer:
    return GroundedAnswer(
        claims=[GroundedClaim(text=text, evidence_ids=list(evidence_ids))],
        raw_text=text,
    )


def _gate(min_coverage: float = 0.5) -> GroundedAnswerGate:
    return GroundedAnswerGate(min_coverage=min_coverage)


# ----------------------------------------------------------------------
# validation reports, never rewrites
# ----------------------------------------------------------------------


async def test_a_low_coverage_answer_keeps_the_model_prose():
    """Density is a finding, not grounds for replacing what the model wrote."""
    evidence = _evidence("E1")
    answer = GroundedAnswer(
        claims=[
            GroundedClaim(text="The warranty lasts 24 months.", evidence_ids=["E1"]),
            GroundedClaim(text="It covers accidental damage.", evidence_ids=[]),
            GroundedClaim(text="Shipping is free.", evidence_ids=[]),
        ],
        raw_text="The warranty lasts 24 months. It covers accidental damage. Shipping is free.",
    )

    finalization = await _gate().finalize_answer(evidence=evidence, answer=answer)

    assert finalization.answer.abstained is False
    assert finalization.answer.raw_text == answer.raw_text
    assert "citation_coverage_below_minimum" in finalization.validation.reason_codes


async def test_an_unstructured_answer_is_reported_not_replaced():
    evidence = _evidence("E1")
    answer = GroundedAnswer(claims=[], raw_text="Some prose the parser found no claims in.")

    finalization = await _gate().finalize_answer(evidence=evidence, answer=answer)

    assert finalization.answer.abstained is False
    assert finalization.answer.raw_text == answer.raw_text


async def test_a_valid_answer_is_still_accepted_unchanged():
    evidence = _evidence("E1")
    answer = _answer("E1")

    finalization = await _gate().finalize_answer(evidence=evidence, answer=answer)

    assert finalization.validation.valid is True
    assert finalization.answer.raw_text == answer.raw_text


async def test_validation_findings_survive_into_metadata():
    """Dropping enforcement must not drop observability."""
    evidence = _evidence("E1")
    answer = _answer("E7")

    finalization = await _gate().finalize_answer(evidence=evidence, answer=answer)
    metadata = finalization.to_metadata()

    assert "unknown_evidence_id" in metadata["reason_codes"]
    assert metadata["outcome"] == "accepted_with_findings"


async def test_a_clean_answer_reports_a_plain_accepted_outcome():
    finalization = await _gate().finalize_answer(evidence=_evidence("E1"), answer=_answer("E1"))
    assert finalization.to_metadata()["outcome"] == "accepted"


# ----------------------------------------------------------------------
# the gate no longer has a rewrite path at all
# ----------------------------------------------------------------------


async def test_finalize_answer_takes_no_regenerator():
    """A regenerate hook would be a second model call the reader never sees."""
    import inspect

    parameters = inspect.signature(GroundedAnswerGate.finalize_answer).parameters
    assert "regenerate" not in parameters
    assert "mode" not in parameters


async def test_no_enforcement_path_produces_an_abstention():
    """Every failure mode above must leave the model's answer in place."""
    evidence = _evidence("E1")
    cases = [
        _answer("E9"),  # unknown evidence id
        GroundedAnswer(claims=[], raw_text="unparsed prose"),  # unstructured
        GroundedAnswer(  # coverage below minimum
            claims=[
                GroundedClaim(text="a", evidence_ids=["E1"]),
                GroundedClaim(text="b", evidence_ids=[]),
                GroundedClaim(text="c", evidence_ids=[]),
            ],
            raw_text="a b c",
        ),
    ]

    for answer in cases:
        finalization = await _gate().finalize_answer(evidence=evidence, answer=answer)
        assert finalization.answer.abstained is False, f"{answer} was replaced by an abstention"


# ----------------------------------------------------------------------
# what grounding still removes: citations the server cannot vouch for
# ----------------------------------------------------------------------


def test_an_unknown_marker_is_neutralized_in_the_rendered_answer():
    """A citation to nothing must not render as a citation to something."""
    evidence = _evidence("E1")
    answer = GroundedAnswer(
        claims=[GroundedClaim(text="The warranty lasts 24 months.", evidence_ids=["E1", "E9"])],
        raw_text="The warranty lasts 24 months [E1][E9].",
    )

    rendered = render_grounded_answer(answer, evidence, text=answer.raw_text)

    assert "[E1]" in rendered, "a resolvable citation must survive"
    assert "[E9]" not in rendered, "an unresolvable citation must not render as one"


def test_a_known_marker_survives_rendering_untouched():
    evidence = _evidence("E1", "E2")
    answer = GroundedAnswer(
        claims=[GroundedClaim(text="Both apply.", evidence_ids=["E1", "E2"])],
        raw_text="Both apply [E1][E2].",
    )

    rendered = render_grounded_answer(answer, evidence, text=answer.raw_text)

    assert "[E1]" in rendered
    assert "[E2]" in rendered


def test_a_grouped_marker_keeps_only_its_resolvable_ids():
    evidence = _evidence("E1")
    answer = GroundedAnswer(
        claims=[GroundedClaim(text="Mixed.", evidence_ids=["E1", "E9"])],
        raw_text="Mixed [E1, E9].",
    )

    rendered = render_grounded_answer(answer, evidence, text=answer.raw_text)

    assert "E9" not in rendered
    assert "E1" in rendered


def test_the_server_source_list_is_still_appended():
    """Neutralizing markers must not cost the reader the source list."""
    evidence = _evidence("E1")
    answer = _answer("E1", text="The warranty lasts 24 months [E1].")

    rendered = render_grounded_answer(answer, evidence, text=answer.raw_text)

    assert "manual.pdf" in rendered


# ----------------------------------------------------------------------
# what this left behind
# ----------------------------------------------------------------------
#
# Removing enforcement stranded the machinery that carried it out. It is
# recorded here rather than deleted in the same change, so the cleanup is
# visible work rather than an implied leftover. Delete an entry when its code
# goes — never by loosening the assertion.

STRANDED_BY_STREAMABLE_GROUNDING = {
    "GroundedAnswerGate.finalize": "decided abstention; nothing calls it now",
    "GroundedAnswerGate.abstain": "authored the abstention text; nothing calls it now",
    "RAGAgent.regenerate_grounded_answer": "the second model call; nothing calls it now",
    "GROUNDED_ANSWER_REGENERATION_PROMPT": "instructed that call; nothing reads it now",
}


def test_stranded_enforcement_machinery_has_no_production_caller():
    """Fails when one of these grows a caller again, or when it is deleted.

    Either outcome means this record is stale. A new caller would mean
    enforcement came back without the streaming consequence being reconsidered.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    # Everything outside the module that defines them. ``finalize`` still calls
    # ``abstain`` internally — both are dead as entry points, not as a pair.
    gate_module = repo_root / "app" / "services" / "rag_grounding.py"
    callers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo_root / "app").rglob("*.py"))
        if path != gate_module
    )

    # Its definition in rag_agent.py is the one occurrence that is allowed to
    # remain; a second is a call site.
    all_runtime = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((repo_root / "app").rglob("*.py"))
    )
    assert all_runtime.count("regenerate_grounded_answer") == 1, (
        "regenerate_grounded_answer has a caller again — enforcement is back, "
        "and withholding the answer stream would be back with it"
    )
    # Narrow on purpose: ``.finalize(`` alone matches unrelated finalizers all
    # over the workflow. These are the ways a caller reaches *this* gate.
    for call in ("gate.finalize(", "gate.abstain(", "_gate.finalize(", "_gate.abstain("):
        assert call not in callers, f"{call} is reachable from production again"


def test_the_answer_model_still_carries_its_abstention_fields():
    """``abstained`` stays on the contract: a model may still decline.

    What was removed is the *server* replacing an answer with an abstention,
    not the model's ability to say it cannot answer.
    """
    assert "abstained" in GroundedAnswer.model_fields
    assert GroundedAnswer().abstained is False
