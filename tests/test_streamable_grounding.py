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


# ----------------------------------------------------------------------
# the same rule, applied token by token
# ----------------------------------------------------------------------
#
# The stream filter and the renderer must agree, or the reader watches one
# answer arrive and a different one gets stored. Both call the same rule.


def _filter(*known: str):
    from app.services.rag_grounding import CitationStreamFilter

    stream_filter = CitationStreamFilter()
    stream_filter.learn(known)
    return stream_filter


def _stream(stream_filter, *chunks: str) -> str:
    return "".join(stream_filter.feed(chunk) for chunk in chunks) + stream_filter.flush()


def test_a_resolvable_marker_streams_through_intact():
    assert _stream(_filter("E1"), "Revenue rose ", "[E1]", ".") == "Revenue rose [E1]."


def test_an_unresolvable_marker_never_reaches_the_reader():
    assert _stream(_filter("E1"), "Revenue rose ", "[E9]", ".") == "Revenue rose."


def test_a_marker_split_across_chunks_is_still_judged_whole():
    """Token boundaries are arbitrary; a marker must not slip through halved."""
    assert _stream(_filter("E1"), "Revenue rose [", "E", "9", "]", " sharply.") == (
        "Revenue rose sharply."
    )


def test_a_resolvable_marker_split_across_chunks_survives():
    assert _stream(_filter("E1"), "Revenue rose [", "E", "1", "]", ".") == "Revenue rose [E1]."


def test_a_grouped_marker_keeps_only_resolvable_ids_mid_stream():
    assert _stream(_filter("E1"), "Both [E1, E9] apply.") == "Both [E1] apply."


def test_ordinary_brackets_are_not_held_back():
    """Only evidence markers are candidates; prose in brackets must flow."""
    assert _stream(_filter("E1"), "See [note] and [1] here.") == "See [note] and [1] here."


def test_an_unclosed_bracket_is_flushed_rather_than_swallowed():
    assert _stream(_filter("E1"), "Revenue rose [E1") == "Revenue rose [E1"


def test_nothing_is_emitted_twice():
    stream_filter = _filter("E1")
    first = stream_filter.feed("Revenue rose [E1")
    second = stream_filter.feed("] sharply.")
    assert first + second + stream_filter.flush() == "Revenue rose [E1] sharply."


def test_with_no_evidence_learned_every_marker_is_dropped():
    assert _stream(_filter(), "Revenue rose [E1].") == "Revenue rose."


def test_the_filter_and_the_renderer_agree():
    """One rule, two call sites — drift between them is the bug this prevents."""
    evidence = _evidence("E1")
    text = "Revenue rose [E1] and fell [E9]."

    streamed = _stream(_filter("E1"), *text)
    rendered = render_grounded_answer(
        GroundedAnswer(
            claims=[GroundedClaim(text="Revenue rose.", evidence_ids=["E1", "E9"])],
            raw_text=text,
        ),
        evidence,
        text=text,
    )

    assert streamed == rendered.split("\n\nSources")[0]


# ----------------------------------------------------------------------
# wired into the real projector
# ----------------------------------------------------------------------
#
# The filter working in isolation says nothing about whether the stream uses
# it. These drive the projector the graph actually runs.


def _projector_and_context():
    from app.services.event_streaming.graph_public_projection import (
        GraphPublicStreamProjector,
        StreamProjectionContext,
    )

    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **_kwargs: iter(()),
        suppress_internal_stream_chunks=False,
    )
    return projector, StreamProjectionContext()


def _retrieval_update(*evidence_ids: str):
    from app.services.event_streaming.events import make_event

    return make_event(
        "state_snapshot",
        sequence=1,
        node="rag_tools",
        data={
            "kind": "updates_tuple",
            "node_state": {
                "context": {
                    "tool_artifacts": [
                        {
                            "tool_call_id": "call-1",
                            "rag_evidence": {
                                "records": [
                                    {"evidence_id": evidence_id} for evidence_id in evidence_ids
                                ]
                            },
                        }
                    ]
                }
            },
        },
    )


def _answer_chunks(projector, ctx, *chunks: str):
    from app.services.event_streaming.events import make_event
    from app.services.event_streaming.graph_public_projection import flush_answer_text

    emitted = []
    running = ""
    for index, chunk in enumerate(chunks):
        running += chunk
        for event in projector.map_event(
            make_event("message_delta", sequence=index + 2, data={"text": running}), ctx
        ):
            if event.type == "message_delta":
                emitted.append(event.data["text"])
    emitted.extend(event.data["text"] for event in flush_answer_text(ctx))
    return "".join(emitted)


def test_the_projector_learns_this_turns_evidence_from_its_retrieval():
    projector, ctx = _projector_and_context()
    list(projector.map_event(_retrieval_update("E1", "E2"), ctx))

    assert ctx.citation_filter._known == {"E1", "E2"}


def test_a_streamed_answer_keeps_citations_the_turn_retrieved():
    projector, ctx = _projector_and_context()
    list(projector.map_event(_retrieval_update("E1"), ctx))

    published = _answer_chunks(projector, ctx, "Revenue rose ", "[E1]", ".")

    assert published == "Revenue rose [E1]."


def test_a_streamed_answer_drops_a_citation_the_turn_never_retrieved():
    """The hole this closes: an invented citation reaching the reader live."""
    projector, ctx = _projector_and_context()
    list(projector.map_event(_retrieval_update("E1"), ctx))

    published = _answer_chunks(projector, ctx, "Revenue rose ", "[E9]", ".")

    assert published == "Revenue rose."


def test_a_turn_with_no_retrieval_publishes_no_citations():
    projector, ctx = _projector_and_context()

    published = _answer_chunks(projector, ctx, "Revenue rose [E1].")

    assert published == "Revenue rose."


def test_held_text_is_released_rather_than_lost():
    """Judging a marker must never cost the reader the end of the answer."""
    projector, ctx = _projector_and_context()
    list(projector.map_event(_retrieval_update("E1"), ctx))

    published = _answer_chunks(projector, ctx, "Revenue rose [E1")

    assert published == "Revenue rose [E1"
