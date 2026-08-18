"""Round-1 fix (finding 2): EvidenceAssembler's metrics kwarg was inert.

``execute_search_documents_action`` is the sole repo-wide construction site
for ``EvidenceAssembler``. Before this fix it passed no ``metrics``, so
``rag_stage_duration_seconds{stage="evidence_assembly"}`` and
``rag_evidence_pack_tokens`` never received a sample in production even
though the class itself already recorded them when given a metrics object
(pinned separately in tests/test_rag_evidence.py).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.ai import rag_tool_actions


class _FakeRagAgent:
    """Minimal stand-in exposing only what this one action path calls."""

    def __init__(self) -> None:
        self.retriever = None

    async def _search(self, query, *, conversation_id, user_id, include_evidence_metadata):
        del query, conversation_id, user_id, include_evidence_metadata
        return [{"document_id": str(uuid4()), "content": "hello", "chunk_id": str(uuid4())}]

    async def _fetch_images_for_chunks(self, *args, **kwargs):
        del args, kwargs
        return []


@pytest.mark.asyncio
async def test_execute_search_documents_action_wires_rag_metrics_into_evidence_assembler(
    monkeypatch,
):
    from app.observability.rag import rag_metrics

    captured_kwargs: dict = {}
    original_init = rag_tool_actions.EvidenceAssembler.__init__

    def _capturing_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rag_tool_actions.EvidenceAssembler, "__init__", _capturing_init)

    await rag_tool_actions.execute_search_documents_action(
        rag_agent=_FakeRagAgent(),
        conversation_id=str(uuid4()),
        tool_args={"action": "search_chunks", "query": "revenue"},
        context={},
        max_agentic_images=0,
        user_id=str(uuid4()),
    )

    assert captured_kwargs.get("metrics") is rag_metrics
