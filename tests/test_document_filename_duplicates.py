"""Duplicate-filename detection tests for repository and service layers.

Covers:
* normalization (casefold + path stripping + NFC)
* ``DocumentService.validate_and_create_document`` rejecting duplicates
* repository create translating IntegrityError into the domain exception
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError


def test_normalize_strips_path_and_casefolds():
    from app.services.document_service import normalize_document_filename

    assert normalize_document_filename("Report.PDF") == "report.pdf"
    assert normalize_document_filename("dir\\sub\\REPORT.pdf") == "report.pdf"
    assert normalize_document_filename(" /tmp/Slides.PPTX ") == "slides.pptx"


def test_normalize_rejects_empty_inputs():
    from app.core.exceptions.validation import FileValidationError
    from app.services.document_service import normalize_document_filename

    with pytest.raises((FileValidationError, ValueError)):
        normalize_document_filename("")
    with pytest.raises((FileValidationError, ValueError)):
        normalize_document_filename("   ")


def test_service_rejects_duplicate_filename_before_create():
    from types import SimpleNamespace

    from app.core.exceptions.validation import DuplicateDocumentFilenameError
    from app.schemas.document import DocumentStatus
    from app.services.document_service import DocumentService

    service = object.__new__(DocumentService)

    async def _validate_upload(filename, file_size):
        return {"valid": True}

    processing_service = MagicMock()
    processing_service.validate_upload_file = MagicMock(side_effect=_validate_upload)
    service.processing_service = processing_service

    document_repository = MagicMock()
    document_repository.get_by_conversation_and_filename_key.return_value = SimpleNamespace(
        id=uuid4(), status=DocumentStatus.READY.value
    )
    document_repository.create.side_effect = AssertionError(
        "Repository create must not be called when a duplicate exists"
    )
    service.repository = document_repository

    service.document_validation_utils = MagicMock()
    service.index_service = MagicMock()

    with pytest.raises(DuplicateDocumentFilenameError):
        asyncio.run(
            service.validate_and_create_document(
                filename="duplicate.pdf",
                file_size=10,
                content_type="application/pdf",
                conversation_id=uuid4(),
            )
        )

    document_repository.create.assert_not_called()


def test_replacing_a_failed_upload_also_removes_its_index():
    from types import SimpleNamespace

    from app.schemas.document import DocumentStatus
    from app.services.document_service import DocumentService

    failed = SimpleNamespace(
        id=uuid4(),
        conversation_id=uuid4(),
        filename="report.pdf",
        status=DocumentStatus.FAILED.value,
    )
    service = object.__new__(DocumentService)

    async def _validate_upload(filename, file_size):
        return {"valid": True}

    service.processing_service = MagicMock(validate_upload_file=_validate_upload)
    service.repository = MagicMock()
    service.repository.get_by_conversation_and_filename_key.return_value = failed
    service.repository.get_by_id.return_value = failed
    service.repository.delete.return_value = True
    service.repository.create.side_effect = RuntimeError("stop after the replace")
    service.index_service = MagicMock()

    with pytest.raises(RuntimeError, match="stop after the replace"):
        asyncio.run(
            service.validate_and_create_document(
                filename="report.pdf",
                file_size=10,
                content_type="application/pdf",
                conversation_id=failed.conversation_id,
            )
        )

    service.index_service.delete_document_index.assert_called_once_with(failed.id)
    service.repository.delete.assert_called_once_with(failed.id)


def test_repository_create_maps_integrity_error_to_domain_exception():
    from app.core.exceptions.validation import DuplicateDocumentFilenameError
    from app.repositories.document import DocumentRepository
    from app.schemas.document import DocumentCreate, DocumentStatus

    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock(
        side_effect=IntegrityError("INSERT", {}, Exception("unique constraint failed"))
    )
    session.rollback = MagicMock()
    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = session
    session_factory.return_value.__exit__ = MagicMock(return_value=False)

    repo = DocumentRepository(session_factory=session_factory)
    payload = DocumentCreate(
        conversation_id=uuid4(),
        filename="dup.pdf",
        filename_key="dup.pdf",
        file_type="application/pdf",
        status=DocumentStatus.PROCESSING.value,
    )

    with pytest.raises(DuplicateDocumentFilenameError):
        repo.create(payload)
