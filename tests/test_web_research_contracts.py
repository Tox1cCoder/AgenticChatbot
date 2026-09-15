from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai.web_research.contracts import (
    ImageCandidateRecord,
    SourceRecord,
    WebEvidenceBundle,
)


def _source(source_id: str = "S1") -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url="https://example.test/article",
        title="Example",
        snippet="Evidence",
        status="search_result",
        provider="test",
        query_index=1,
    )


def _image(index: int, source_id: str = "S1") -> ImageCandidateRecord:
    return ImageCandidateRecord(
        candidate_id=f"I{index}",
        source_id=source_id,
        delivery_url=f"/web-images/{uuid4()}",
        mime_type="image/jpeg",
        width=640,
        height=360,
        byte_size=1024,
        digest=f"{index:064x}",
        provider="test",
    )


def test_gallery_limit_is_derived_from_bundle_visual_intent() -> None:
    images = tuple(_image(index) for index in range(1, 7))

    gallery = WebEvidenceBundle(
        status="success",
        mode="agentic",
        visual_intent="gallery",
        operation_index=1,
        sources=(_source(),),
        images=images,
    )

    assert len(gallery.images) == 6
    with pytest.raises(ValidationError, match="image limit"):
        WebEvidenceBundle(
            status="success",
            mode="agentic",
            visual_intent="figure",
            operation_index=1,
            sources=(_source(),),
            images=images[:5],
        )


def test_bundle_rejects_an_image_without_an_admitted_source() -> None:
    with pytest.raises(ValidationError, match="unknown source"):
        WebEvidenceBundle(
            status="success",
            mode="quick",
            visual_intent="figure",
            operation_index=1,
            sources=(_source(),),
            images=(_image(1, source_id="S9"),),
        )


def test_delivery_url_must_be_an_opaque_protected_reference() -> None:
    payload = _image(1).model_dump()
    payload["delivery_url"] = "https://private.example/image.jpg"
    with pytest.raises(ValidationError):
        ImageCandidateRecord.model_validate(payload)
