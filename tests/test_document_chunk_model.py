"""Phase 2 guards: the normalized ``document_chunks`` ORM model.

These tests pin the shape of the new DocumentChunk model:
  * Required columns exist with the right semantic meaning.
  * Relationships wire Document <-> DocumentChunk <-> DocumentImage cleanly.
  * parse_artifact <-> chunks relationship exists.
"""

from __future__ import annotations

REQUIRED_FIELDS = {
    "id",
    "document_id",
    "parse_artifact_id",
    "qdrant_point_id",
    "chunk_index",
    "content",
    "content_sha256",
    "char_count",
    "token_count",
    "page_start",
    "page_end",
    "section_path",
    "block_provenance",
    "chunk_metadata",
    "index_status",
    "index_error",
    "indexed_at",
    "embedding_model",
    "embedding_dimension",
    "qdrant_collection_name",
    "created_at",
    "updated_at",
}


def test_document_chunk_is_exported_from_models_package():
    import app.models as models

    assert hasattr(models, "DocumentChunk"), "app.models must export DocumentChunk"
    assert "DocumentChunk" in models.__all__


def test_document_chunk_has_all_required_columns():
    from app.models import DocumentChunk

    columns = {c.name for c in DocumentChunk.__table__.columns}
    missing = REQUIRED_FIELDS - columns
    assert not missing, f"DocumentChunk missing required columns: {sorted(missing)}"


def test_document_chunk_table_has_unique_generation_index_tuple():
    from app.models import DocumentChunk

    uniques: list[tuple[str, ...]] = []
    for constraint in DocumentChunk.__table__.constraints:
        if constraint.__class__.__name__ == "UniqueConstraint":
            uniques.append(tuple(col.name for col in constraint.columns))

    assert ("document_id", "index_generation_id", "chunk_index") in uniques, (
        "Expected UNIQUE(document_id, index_generation_id, chunk_index). "
        f"Found: {uniques}"
    )


def test_document_chunk_required_columns_are_non_nullable():
    from app.models import DocumentChunk

    non_nullable_expected = {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "content_sha256",
        "char_count",
        "token_count",
        "index_status",
        "created_at",
        "updated_at",
    }
    columns = DocumentChunk.__table__.columns
    actually_nullable = {c.name for c in columns if c.nullable}
    leaked = non_nullable_expected & actually_nullable
    assert not leaked, f"Columns expected NOT NULL but nullable in ORM: {sorted(leaked)}"


def test_document_chunk_foreign_keys_have_correct_ondelete():
    from app.models import DocumentChunk

    fk_by_column = {}
    for fk in DocumentChunk.__table__.foreign_keys:
        fk_by_column[fk.column.table.name] = {
            "column": fk.parent.name,
            "ondelete": fk.ondelete,
        }

    assert fk_by_column.get("documents", {}).get("ondelete") == "CASCADE"
    assert fk_by_column.get("document_parse_artifacts", {}).get("ondelete") == "SET NULL"


def test_document_relationships_back_populate_chunks():
    from app.models import Document, DocumentChunk

    assert hasattr(Document, "chunks"), "Document must expose .chunks"
    doc_rel = Document.__mapper__.relationships.get("chunks")
    assert doc_rel is not None
    assert (
        doc_rel.argument == "DocumentChunk"
        or getattr(doc_rel.mapper.class_, "__name__", "") == "DocumentChunk"
    )
    assert doc_rel.back_populates == "document"

    chunk_rel = DocumentChunk.__mapper__.relationships.get("document")
    assert chunk_rel is not None
    assert chunk_rel.back_populates == "chunks"


def test_parse_artifact_relationships_back_populate_chunks():
    from app.models import DocumentChunk, DocumentParseArtifact

    assert hasattr(DocumentParseArtifact, "chunks")
    artifact_rel = DocumentParseArtifact.__mapper__.relationships.get("chunks")
    assert artifact_rel is not None
    assert artifact_rel.back_populates == "parse_artifact"

    chunk_rel = DocumentChunk.__mapper__.relationships.get("parse_artifact")
    assert chunk_rel is not None
    assert chunk_rel.back_populates == "chunks"


def test_document_image_chunk_relationship_wires_to_document_chunk():
    from app.models import DocumentChunk
    from app.models.document_image import DocumentImage

    # DocumentImage.chunk_id must be a real FK to document_chunks.id.
    chunk_id_col = DocumentImage.__table__.c["chunk_id"]
    fk_targets = {fk.column.table.name for fk in chunk_id_col.foreign_keys}
    assert "document_chunks" in fk_targets, (
        f"document_images.chunk_id must reference document_chunks. Found: {fk_targets}"
    )
    for fk in chunk_id_col.foreign_keys:
        if fk.column.table.name == "document_chunks":
            assert fk.ondelete == "SET NULL"

    # Relationship wiring.
    assert hasattr(DocumentImage, "chunk")
    image_rel = DocumentImage.__mapper__.relationships.get("chunk")
    assert image_rel is not None
    assert image_rel.back_populates == "images"

    chunk_rel = DocumentChunk.__mapper__.relationships.get("images")
    assert chunk_rel is not None
    assert chunk_rel.back_populates == "chunk"
