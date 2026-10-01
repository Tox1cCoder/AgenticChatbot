"""Phase 7 guards: ``DocumentService.delete_document`` no longer builds RAGAgent.

Document cleanup belongs to ``DocumentIndexService``, not an agent. Pinning
this here protects against the regression where CRUD cleanup accidentally
pulled in the model-runtime agent again.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from uuid import uuid4

from app.services.document_service import DocumentService


def test_delete_document_does_not_instantiate_rag_agent():
    source = inspect.getsource(DocumentService.delete_document) + inspect.getsource(
        DocumentService._delete_document
    )
    assert "RAGAgent" not in source, (
        "DocumentService.delete_document must delegate to DocumentIndexService — "
        "instantiating RAGAgent for cleanup couples CRUD to the model runtime."
    )


def test_document_service_module_does_not_import_rag_agent():
    import app.services.document_service as mod

    source = inspect.getsource(mod)
    # The file may still reference RAGAgent in a comment or docstring, so
    # only check the import statements.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("from app.ai.agents.rag_agent"), (
            "document_service.py must not import RAGAgent — Phase 7 cleanup"
        )
        assert "import app.ai.agents.rag_agent" not in stripped, (
            "document_service.py must not import RAGAgent — Phase 7 cleanup"
        )


def test_delete_document_calls_document_index_service():
    """Deletion routes chunk/vector cleanup through the index service, then the row."""
    calls: list[tuple[str, object]] = []
    document_id = uuid4()
    repository = SimpleNamespace(
        get_by_id=lambda doc_id: SimpleNamespace(id=doc_id),
        delete=lambda doc_id: calls.append(("repository.delete", doc_id)) or True,
    )
    index_service = SimpleNamespace(
        delete_document_index=lambda doc_id: calls.append(("index.delete", doc_id))
    )
    service = DocumentService(
        document_repository=repository,
        document_processing_service=SimpleNamespace(),
        document_validation_utils=SimpleNamespace(),
        document_index_service=index_service,
    )

    assert asyncio.run(service.delete_document(document_id)) is True
    assert calls == [("index.delete", document_id), ("repository.delete", document_id)]
