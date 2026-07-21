"""Small UTF-8 text loader without the sunset ``langchain-community`` package."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_utf8_text_document(file_path: str | Path) -> Document:
    """Load one UTF-8 file while preserving the former loader's public shape."""
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Error loading {file_path}") from exc
    return Document(page_content=text, metadata={"source": str(file_path)})
