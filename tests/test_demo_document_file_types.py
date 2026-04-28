"""Phase 11 guard: demo upload UI accepts every server-supported extension.

The server validates uploads against
``app.api.documents.SUPPORTED_UPLOAD_EXTENSIONS`` (.txt, .pdf, .docx,
.pptx, .xlsx, .html, .md). The legacy demo and sidebar uploaders must
not silently reject formats the server already accepts.
"""

from __future__ import annotations

import re
from pathlib import Path

EXPECTED = {"txt", "pdf", "docx", "pptx", "xlsx", "html", "md"}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _extract_document_type_list(source: str) -> set[str] | None:
    """Find any ``type=[...]`` literal that names document formats and return the set.

    Demo files can have multiple ``type=[...]`` literals (e.g. an image
    uploader and a document uploader). We return the first one that
    contains ``"pdf"`` since the document uploader is the only place that
    accepts PDFs.
    """
    pattern = re.compile(r"type=\[([^\]]*)\]")
    for match in pattern.finditer(source):
        raw = match.group(1)
        items = {piece.strip().strip('"').strip("'") for piece in raw.split(",")}
        items.discard("")
        if "pdf" in items:
            return items
    return None


def test_demo_py_upload_extensions_match_server_validation():
    demo_path = _REPO_ROOT / "demo.py"
    if not demo_path.exists():
        # Demo file optional in some environments — skip rather than fail.
        import pytest

        pytest.skip("demo.py not present")

    source = demo_path.read_text(encoding="utf-8")
    extensions = _extract_document_type_list(source)
    assert extensions is not None, "Could not find a document type=[...] literal in demo.py"
    assert EXPECTED.issubset(extensions), (
        f"demo.py upload list missing extensions: {sorted(EXPECTED - extensions)}"
    )


def test_upload_support_py_extensions_match_server_validation():
    upload_path = _REPO_ROOT / "upload_support.py"
    if not upload_path.exists():
        import pytest

        pytest.skip("upload_support.py not present")

    source = upload_path.read_text(encoding="utf-8")
    extensions = _extract_document_type_list(source)
    assert extensions is not None, (
        "Could not find a document type=[...] literal in upload_support.py"
    )
    assert EXPECTED.issubset(extensions), (
        f"upload_support.py upload list missing extensions: {sorted(EXPECTED - extensions)}"
    )
