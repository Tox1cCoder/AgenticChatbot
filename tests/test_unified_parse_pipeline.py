"""Phase 4 guards: one server-side parse pipeline.

Tests pin:
  * The server owns a single ``SUPPORTED_UPLOAD_EXTENSIONS`` set.
  * MinerU helpers are format-neutral (no PDF-only naming).
  * Rich formats (``.pdf``, ``.docx``, ``.pptx``, ``.xlsx``, ``.html``, ``.md``) are accepted.
  * Excel workbooks use the server-side openpyxl path, not MinerU markdown output.
  * Unknown extensions are rejected.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from app.services.document_parse_service import DocumentParseService
from app.services.document_processing_service import DocumentProcessingService


def test_supported_upload_extensions_is_exported():
    import app.api.documents as documents_api

    assert hasattr(documents_api, "SUPPORTED_UPLOAD_EXTENSIONS")
    exts = set(documents_api.SUPPORTED_UPLOAD_EXTENSIONS)
    # The product-supported set must include plain text plus the MinerU-targeted formats.
    expected = {".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}
    missing = expected - exts
    assert not missing, f"Missing extensions: {sorted(missing)}"


def test_processing_service_accepts_supported_extensions():
    accepted = (".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md")
    for ext in accepted:
        result = DocumentProcessingService._validate_file_extension(f"file{ext}")
        assert result == ext, f"Expected .{ext} to be accepted"


def test_processing_service_rejects_unsupported_extensions():
    for bad in (".exe", ".zip", ".tar", ".csv.gz", ""):
        with pytest.raises(ValueError):
            DocumentProcessingService._validate_file_extension(f"evil{bad}")


def test_mineru_helper_names_are_format_neutral():
    """PDF-only helper names are renamed to format-neutral equivalents."""
    members = dict(inspect.getmembers(DocumentProcessingService))
    assert "_process_with_mineru" in members, (
        "Expected format-neutral _process_with_mineru; PDF-only name is retired"
    )
    # Allow the old name to exist temporarily only if it delegates to the new name,
    # but the plan says delete PDF-only naming — enforce that.
    if "_process_pdf_with_mineru" in members:
        src = inspect.getsource(members["_process_pdf_with_mineru"])
        assert "_process_with_mineru" in src, (
            "If _process_pdf_with_mineru is kept, it must delegate to _process_with_mineru"
        )


def test_process_document_routes_supported_rich_formats_to_server_parsers():
    """The dispatch switch keeps Excel off the MinerU markdown path."""
    src = inspect.getsource(DocumentProcessingService.process_document)
    assert ".xlsx" in DocumentProcessingService.EXCEL_EXTENSIONS
    assert ".xlsx" not in DocumentProcessingService.MINERU_EXTENSIONS
    assert "_process_excel_workbook" in src
    for ext in (".pdf", ".docx", ".pptx", ".html", ".md"):
        assert ext in DocumentProcessingService.MINERU_EXTENSIONS
        assert ext in src, f"Expected process_document to dispatch {ext} through MinerU routing"


def test_plain_text_parse_normalizes_once_without_character_chunking(tmp_path):
    text_path = tmp_path / "long.txt"
    text = "First paragraph. " * 200
    text_path.write_text(text, encoding="utf-8")
    settings = type(
        "Settings",
        (),
        {
            "rag_chunk_target_tokens": 5,
            "rag_chunk_overlap_tokens": 1,
            "rag_chunk_max_tokens": 10,
        },
    )()
    service = DocumentParseService(settings=settings)

    result = asyncio.run(service.parse_document(str(text_path), "long.txt", "doc-1"))

    assert len(result.blocks) == 1
    assert result.blocks[0].kind == "paragraph"
    assert result.blocks[0].text == text


def test_parse_document_records_the_parse_stage_metric(tmp_path, monkeypatch):
    """Round-1 fix (finding 3): the "parse" stage enum had no producer."""
    from app.observability.rag import rag_metrics

    text_path = tmp_path / "short.txt"
    text_path.write_text("hello world", encoding="utf-8")
    settings = type(
        "Settings",
        (),
        {
            "rag_chunk_target_tokens": 5,
            "rag_chunk_overlap_tokens": 1,
            "rag_chunk_max_tokens": 10,
        },
    )()
    service = DocumentParseService(settings=settings)

    recorded: list[tuple[str, float]] = []
    original_stage = rag_metrics.stage

    def _capture(stage, *, elapsed_seconds, labels=None):
        recorded.append((stage, elapsed_seconds))
        return original_stage(stage, elapsed_seconds=elapsed_seconds, labels=labels)

    monkeypatch.setattr(rag_metrics, "stage", _capture)

    asyncio.run(service.parse_document(str(text_path), "short.txt", "doc-2"))

    assert any(stage == "parse" and elapsed >= 0.0 for stage, elapsed in recorded)
