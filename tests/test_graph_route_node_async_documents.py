"""Routing must not block the loop on its document-count query.

``_route_node`` runs before the first token, so a synchronous COUNT there delays
every concurrent stream's time-to-first-token, not just this one's.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.ai.graph import MultiAgentWorkflow


class _AsyncDocumentRepository:
    def __init__(self, count: int):
        self._count = count
        self.sync_calls = 0
        self.async_calls = 0

    def count_by_conversation(self, conversation_id):
        self.sync_calls += 1
        return self._count

    async def acount_by_conversation(self, conversation_id):
        self.async_calls += 1
        return self._count


def _workflow(document_repository):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.document_repository = document_repository
    return workflow


async def test_uses_the_async_twin_and_not_the_sync_method():
    repository = _AsyncDocumentRepository(count=3)
    workflow = _workflow(repository)

    assert await workflow._aconversation_has_documents(str(uuid4())) is True
    assert repository.async_calls == 1
    assert repository.sync_calls == 0, "routing still used the blocking sync method"


async def test_returns_false_when_the_conversation_has_no_documents():
    repository = _AsyncDocumentRepository(count=0)
    assert await _workflow(repository)._aconversation_has_documents(str(uuid4())) is False


async def test_returns_false_without_a_repository():
    assert await _workflow(None)._aconversation_has_documents(str(uuid4())) is False


async def test_returns_false_without_a_conversation_id():
    repository = _AsyncDocumentRepository(count=5)
    assert await _workflow(repository)._aconversation_has_documents(None) is False
    assert repository.async_calls == 0


async def test_returns_false_for_a_malformed_conversation_id():
    """UUID() raises ValueError; routing must degrade, not fail the turn."""
    assert (
        await _workflow(_AsyncDocumentRepository(count=1))._aconversation_has_documents(
            "not-a-uuid"
        )
        is False
    )


async def test_returns_false_when_the_query_raises():
    class _Broken:
        async def acount_by_conversation(self, conversation_id):
            raise RuntimeError("database unavailable")

    assert await _workflow(_Broken())._aconversation_has_documents(str(uuid4())) is False


@pytest.mark.parametrize("count, expected", [(0, False), (1, True), (42, True)])
async def test_matches_the_sync_helper_for_the_same_counts(count, expected):
    repository = _AsyncDocumentRepository(count=count)
    workflow = _workflow(repository)
    conversation_id = str(uuid4())

    assert workflow._conversation_has_documents(conversation_id) is expected
    assert await workflow._aconversation_has_documents(conversation_id) is expected
