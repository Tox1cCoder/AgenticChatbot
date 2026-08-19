"""Contract tests for Task 14 of the RAG production-hardening plan.

Two things are pinned here:

1. The four settings this plan spent thirteen tasks building must still ship
   default-off. If any of them ever flips to ``True`` by accident (a bad
   merge, a stray ``.env`` default, a "just for local testing" commit that
   leaks), this test turns red immediately. There is no evaluation evidence
   behind enabling any of them yet -- see docs/rag-rollout-runbook.md.
2. docs/rag-rollout-runbook.md exists and actually says the things an
   operator needs before touching any of these flags: every flag's current
   state, the three ordered grounded-answer-gate blockers, and the fact that
   no release gate is binding. These assertions read the file's text rather
   than re-deriving its content, so they catch the runbook silently losing a
   required section during a future edit -- they do not re-verify the
   runbook's claims against the source code (that verification happened once,
   by hand, when the runbook was written).
"""

import re
from pathlib import Path

import pytest

from app.core.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNBOOK_PATH = REPO_ROOT / "docs" / "rag-rollout-runbook.md"

_RISKY_FLAGS = (
    "rag_hybrid_retrieval_enabled",
    "rag_grounded_answer_gate_enabled",
    "rag_multimodal_image_embeddings_enabled",
    "rag_exact_cache_enabled",
    "rag_semantic_chunking_enabled",
)


@pytest.fixture
def settings():
    return get_settings()


def test_risky_rag_features_default_off(settings):
    """Brief's Step 1 contract test, verbatim. Already green on arrival: all
    four settings have defaulted to False since the tasks that introduced
    them (10-13). Kept as a regression guard, not a RED-to-GREEN exercise.
    """
    assert settings.rag_hybrid_retrieval_enabled is False
    assert settings.rag_grounded_answer_gate_enabled is False
    assert settings.rag_multimodal_image_embeddings_enabled is False
    assert settings.rag_exact_cache_enabled is False


def test_semantic_chunking_default_off(settings):
    """Fifth default-off flag in the rollout order; not in the brief's
    literal Step 1 list, but the same regression guard applies.
    """
    assert settings.rag_semantic_chunking_enabled is False


def test_citation_verification_ships_enabled_by_default(settings):
    """Plan-defect reconciliation.

    The plan's Task 10 Step 4 assumed ``enable_citation_verification``
    started ``False`` and listed enabling it as a future rollout step. It has
    shipped ``default=True`` since Task 10 -- it was never off. This pins the
    actual state so a future change is a deliberate decision, not an
    accidental "fix" of a bug that doesn't exist.
    """
    assert settings.enable_citation_verification is True


def test_min_citation_coverage_is_pinned_pending_reselection(settings):
    """0.5 is an unqualified placeholder, not a value chosen from evaluation
    results, and it currently gates a different quantity (fraction of an
    answer's claims carrying a citation) than the one it was originally
    picked for (fraction of retrieved documents referenced) -- see
    docs/rag-rollout-runbook.md. Pinning the number so it cannot drift
    silently; changing it should be a deliberate, evidenced decision.
    """
    assert settings.min_citation_coverage == 0.5


def test_image_slot_reservation_cap_is_inactive_at_shipped_defaults(settings):
    """``RagAgent._cap_candidates_with_image_reservation`` only does anything
    once ``rag_evidence_candidate_limit`` exceeds ``rag_top_k`` (or a caller
    passes a smaller ``top_k``) -- otherwise the reranker has already
    truncated to the evidence limit and the cap early-returns. Shipped
    defaults (top_k=15, evidence_candidate_limit=10) keep the cap dormant.
    Pinning this relationship before the multimodal image-embeddings flag is
    ever enabled, per the Task 14 dispatch context.
    """
    assert settings.rag_evidence_candidate_limit <= settings.rag_top_k


def test_runbook_exists():
    assert RUNBOOK_PATH.exists(), (
        "docs/rag-rollout-runbook.md must exist and document rollout preconditions "
        "for every default-off RAG flag"
    )


def _runbook_text() -> str:
    return RUNBOOK_PATH.read_text(encoding="utf-8")


def test_runbook_documents_every_flag_by_name():
    text = _runbook_text()
    for flag in _RISKY_FLAGS + ("enable_citation_verification", "min_citation_coverage"):
        assert flag in text, f"runbook must name {flag} explicitly"


def test_runbook_orders_the_three_grounded_gate_blockers():
    """Early sections legitimately forward-reference a later blocker (e.g.
    "see Blocker 2 below"), so this checks the order of the actual blocker
    section headers, not the first mention of each substring anywhere.
    """
    text = _runbook_text()
    first = text.index("**Blocker 1")
    second = text.index("**Blocker 2")
    third = text.index("**Blocker 3")
    assert first < second < third, (
        "the three grounded-answer-gate blocker sections must appear in order -- "
        "blocker 3 cannot clear until blocker 1 does, and presenting them as "
        "an unordered checklist hides that dependency"
    )


def test_runbook_states_release_gates_are_non_binding():
    """Anchored to the dedicated section and requiring several specific
    facts to co-occur there (fourteen gates, the unmeasured status literal,
    the word "non-binding", and "nothing" being binding) -- not just two
    isolated substrings that could each survive an unrelated, or even
    false, rewrite of the surrounding prose.
    """
    text = _runbook_text()
    heading = "## No automated quality gate exists behind any of these decisions"
    assert heading in text, "runbook must have a dedicated non-binding-gates section"
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    assert "fourteen gates" in section
    assert '"status": "unmeasured"' in section
    assert "non-binding" in section
    assert "nothing" in section.lower()


def test_runbook_names_a_rollback_setting_for_every_risky_flag():
    """Each risky flag's own rollout-order section must itself contain a
    Rollback: line naming that same setting.

    A bare document-wide count of "Rollback:" occurrences (the prior version
    of this test) stays green even if one specific flag's rollback line is
    deleted -- including item 7's, the grounded-answer gate, the single flag
    this entire task exists to keep off -- as long as enough *other* entries
    still have one. This locates each flag's own section by its Setting:
    line and requires that same section to name itself as the rollback.
    """
    text = _runbook_text()
    sections = re.split(r"(?=^### \d+\. )", text, flags=re.MULTILINE)
    for flag in _RISKY_FLAGS:
        section = next(
            (s for s in sections if re.search(rf"\*\*Setting:\*\*\s*`{re.escape(flag)}`", s)),
            None,
        )
        assert section is not None, f"no rollout-order section found whose Setting: is {flag}"
        assert re.search(rf"\*\*Rollback:\*\*.*{re.escape(flag)}", section), (
            f"{flag}'s rollout-order section must have a Rollback: line naming {flag} itself"
        )


def test_runbook_documents_env_example_block_manually():
    text = _runbook_text()
    assert ".env.example" in text
    assert "manual" in text.lower()


def test_readme_points_operators_at_the_runbook():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/rag-rollout-runbook.md" in readme
