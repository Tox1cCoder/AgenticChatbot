"""Duplicate-filename detection tests for repository and service layers.

Covers:
* normalization (casefold + path stripping + NFC)
* repository ``filename_exists_in_conversation`` lookup
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


def test_repository_filename_exists_in_conversation_returns_true_when_present():
    from app.repositories.document import DocumentRepository

    session = MagicMock()
    query = MagicMock()
    query.filter.return_value = query
    session.query.return_value = query
    session.query.return_value.scalar.return_value = True

    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = session

    repo = DocumentRepository(session_factory=session_factory)
    assert repo.filename_exists_in_conversation(uuid4(), "report.pdf") is True


def test_service_rejects_duplicate_filename_before_create():
    from app.core.exceptions.validation import DuplicateDocumentFilenameError
    from app.services.document_service import DocumentService

    service = object.__new__(DocumentService)

    async def _validate_upload(filename, file_size):
        return {"valid": True}

    processing_service = MagicMock()
    processing_service.validate_upload_file = MagicMock(side_effect=_validate_upload)
    service.processing_service = processing_service

    document_repository = MagicMock()
    document_repository.filename_exists_in_conversation.return_value = True
    document_repository.create.side_effect = AssertionError(
        "Repository create must not be called when a duplicate exists"
    )
    service.repository = document_repository

    service.document_validation_utils = MagicMock()
    service.index_service = MagicMock()
    service._event_bus = MagicMock()

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
