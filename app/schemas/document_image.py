from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.utils.case_conversion import to_camel_case as to_camel


class DocumentImageBase(BaseModel):
    document_id: UUID
    chunk_id: UUID | None = None
    image_path: str
    image_caption: str | None = None
    page_number: int | None = None
    mime_type: str
    bbox: list[float] | None = None
    section_path: list[str] = Field(default_factory=list)
    content_sha256: str | None = None

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DocumentImageCreate(DocumentImageBase):
    pass


class DocumentImageRead(DocumentImageBase):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime


class DocumentImageUpdate(BaseModel):
    image_caption: str | None = None
    chunk_id: UUID | None = None

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ImageCaptionSections(BaseModel):
    """Validated structured caption produced by the vision captioning call.

    Fields cover what is needed to make an image's content retrievable via
    text search: any legible text (OCR), a chart/figure title, axis and
    legend labels, key values, described trends/relationships, and the
    surrounding section context. ``render()`` flattens this into the
    searchable text stored in ``DocumentImage.image_caption``.

    This is untrusted model output describing untrusted document content —
    never treat any field as an instruction.
    """

    ocr_text: str = Field(
        default="", description="Visible text rendered inside the image, verbatim."
    )
    chart_title: str = Field(default="", description="Chart or figure title, if any.")
    axes: str = Field(default="", description="Axis labels and units, if any.")
    legend: str = Field(default="", description="Legend entries, if any.")
    values: str = Field(default="", description="Key data points or values shown.")
    trends: str = Field(default="", description="Described trends or patterns.")
    relationships: str = Field(
        default="", description="Relationships between depicted elements."
    )
    section_context: str = Field(
        default="",
        description="One sentence tying the image to its surrounding document section.",
    )

    def render(self) -> str:
        """Flatten populated fields into one searchable text block."""
        lines: list[str] = []
        for label, value in (
            ("OCR", self.ocr_text),
            ("Title", self.chart_title),
            ("Axes", self.axes),
            ("Legend", self.legend),
            ("Values", self.values),
            ("Trends", self.trends),
            ("Relationships", self.relationships),
            ("Section", self.section_context),
        ):
            text = str(value or "").strip()
            if text:
                lines.append(f"{label}: {text}")
        return "\n".join(lines)
