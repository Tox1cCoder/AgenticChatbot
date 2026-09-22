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


def test_a_bundle_may_carry_a_registry_larger_than_one_cohort() -> None:
    """The registry is the stable citation store, not the published list.

    Session capacity is a text quota plus a candidate-catalog quota, so a
    bundle legitimately carries far more sources than one visual intent's
    worth. The contract bounds the *active* images instead.
    """

    sources = tuple(
        _source(f"S{index}").model_copy(update={"url": f"https://example.test/{index}"})
        for index in range(1, 21)
    )

    bundle = WebEvidenceBundle(
        status="success",
        mode="quick",
        visual_intent="figure",
        operation_index=1,
        sources=sources,
        images=(_image(1),),
    )

    assert len(bundle.sources) == 20

    with pytest.raises(ValidationError, match="image limit"):
        WebEvidenceBundle(
            status="success",
            mode="quick",
            visual_intent="figure",
            operation_index=1,
            sources=sources,
            images=tuple(_image(index) for index in range(1, 6)),
        )

    with pytest.raises(ValidationError, match="unknown source"):
        WebEvidenceBundle(
            status="success",
            mode="quick",
            visual_intent="figure",
            operation_index=1,
            sources=sources,
            images=(_image(1, source_id="S99"),),
        )


def test_operation_source_ids_must_be_unique_and_known() -> None:
    """The delta names sources the caller can already resolve in the bundle."""

    bundle = WebEvidenceBundle(
        status="success",
        mode="quick",
        visual_intent="none",
        operation_index=1,
        sources=(_source("S1"),),
        operation_source_ids=("S1",),
    )

    assert bundle.operation_source_ids == ("S1",)

    with pytest.raises(ValidationError, match="duplicate operation source ID"):
        WebEvidenceBundle(
            status="success",
            mode="quick",
            visual_intent="none",
            operation_index=1,
            sources=(_source("S1"),),
            operation_source_ids=("S1", "S1"),
        )

    with pytest.raises(ValidationError, match="operation source ID is not in sources"):
        WebEvidenceBundle(
            status="success",
            mode="quick",
            visual_intent="none",
            operation_index=1,
            sources=(_source("S1"),),
            operation_source_ids=("S2",),
        )
