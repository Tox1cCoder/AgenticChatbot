"""Task 10: document-worker embedding attribution.

The index worker derives ownership from the *verified* conversation owner and
passes it explicitly through ``DocumentIndexService`` into the embedding
service (ContextVars do not cross the ThreadPoolExecutor boundary). A
maintenance call with no owner records ``user_id = NULL``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from prometheus_client import CollectorRegistry

from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.usage.recorder import ModelUsageRecorder
from app.usage.types import UsageContext


class _FakeRepo:
    def __init__(self) -> None:
        self.commands: list[RecordEventCommand] = []

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def _response(*vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=list(v)) for v in vectors])


def _recording_embedding_service(monkeypatch, repo):
    from app.services import rag_embedding_service as mod

    fake_client = MagicMock()
    fake_client.models = MagicMock()
    monkeypatch.setattr(mod.genai, "Client", lambda **_kwargs: fake_client)

    service = mod.GeminiRAGEmbeddingService(
        api_key="test-key",
        model_name="gemini-embedding-2",
        dimension=2,
        embedding_batch_size=32,
        recorder=ModelUsageRecorder(
            repository=repo,
            enqueue_failed_write=lambda payload: None,
            metrics=ModelUsageMetrics(registry=CollectorRegistry()),
        ),
    )
    return service, fake_client


def test_document_embedding_attributed_to_conversation_owner(monkeypatch):
    repo = _FakeRepo()
    service, client = _recording_embedding_service(monkeypatch, repo)
    client.models.embed_content.side_effect = [_response([0.1, 0.2])]
    owner_id, conversation_id, document_id = uuid4(), uuid4(), uuid4()

    service.embed_documents(
        ["chunk text"],
        titles=["report.pdf"],
        usage_context=UsageContext(
            user_id=owner_id,
            conversation_id=conversation_id,
            document_id=document_id,
            operation="document_index",
        ),
    )

    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.context.user_id == owner_id
    assert command.context.conversation_id == conversation_id
    assert command.context.document_id == document_id
    assert command.context.operation == "embedding"


def test_maintenance_embedding_has_null_user(monkeypatch):
    repo = _FakeRepo()
    service, client = _recording_embedding_service(monkeypatch, repo)
    client.models.embed_content.side_effect = [_response([0.3, 0.4])]

    # No usage_context and nothing bound -> unattributed maintenance call.
    service.embed_documents(["orphan chunk"], titles=["m.pdf"])

    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.context.user_id is None
    assert command.context.operation == "embedding"
