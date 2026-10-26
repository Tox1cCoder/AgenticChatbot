"""Bounded static answer visuals: steps, bars, and comparison tables.

The model supplies content and semantic status through
``create_answer_visual``; this module validates it, binds its evidence to the
turn's admitted sources, and derives the public payload and its Markdown
fallback from the same validated content. Nothing here accepts HTML, CSS,
script, component names, or model-authored URLs.

Public payload models live here rather than in ``rich_response`` because they
are specific to this family; ``rich_response`` imports them into the rich-item
union. This module must therefore never import ``rich_response``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

ANSWER_VISUAL_TYPE = "answer_visual"
ANSWER_VISUAL_SOURCE = "answer_visual_tool"
#: Server-assigned id namespace, so a marker can be recognized as a visual
#: before its descriptor is known.
ANSWER_VISUAL_ID_PREFIX = "visual:"

#: Serialized public JSON per item, fallback and evidence included.
ANSWER_VISUAL_MAX_ITEM_BYTES = 16 * 1024
#: Referenced static visuals admitted into one answer.
ANSWER_VISUAL_MAX_PER_ANSWER = 6
#: Serialized public JSON of every admitted static visual in one answer.
ANSWER_VISUAL_MAX_TOTAL_BYTES = 64 * 1024

_MAX_ENTRIES = 12
_MAX_COLUMNS = 6
_MAX_REFS = 8

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
Prose = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Unit = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
Basis = Literal["measured", "calculated", "illustrative"]
AssessmentStatus = Literal["positive", "neutral", "caution", "negative"]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class AnswerVisualError(ValueError):
    """A visual the tool must refuse, with a message the model can act on."""


class ResolvedEvidence(BaseModel):
    """A server-resolved citation target for one evidence reference."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    url: str


# ---------------------------------------------------------------------------
# Model-facing input
# ---------------------------------------------------------------------------


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRef(_Spec):
    kind: Literal["web"] = Field(description="Evidence family. Only 'web' is supported.")
    id: str = Field(pattern=r"^S[1-9][0-9]*$", description="A source ID returned this turn.")


Refs = Annotated[list[EvidenceRef], Field(default_factory=list, max_length=_MAX_REFS)]


class StepSpec(_Spec):
    title: Label
    note: Prose | None = None
    evidence: Refs


class BarSpec(_Spec):
    label: Label
    value: NonNegative
    evidence: Refs


class CellSpec(_Spec):
    text: Label
    evidence: Refs


class AssessmentSpec(_Spec):
    status: AssessmentStatus
    label: Label
    reason: Prose
    evidence: Refs


class RowSpec(_Spec):
    label: Label
    cells: list[CellSpec] = Field(min_length=1, max_length=_MAX_COLUMNS)
    assessment: AssessmentSpec | None = None


_VARIANT_FIELDS: dict[str, frozenset[str]] = {
    "steps": frozenset({"steps"}),
    "bars": frozenset({"bars", "unit", "basis", "assumptions", "max_value"}),
    "comparison_table": frozenset({"columns", "rows"}),
}
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "steps": ("steps",),
    "bars": ("bars", "unit", "basis"),
    "comparison_table": ("columns", "rows"),
}


class AnswerVisualSpec(_Spec):
    """One static visual. Fields outside the chosen variant must be omitted."""

    variant: Literal["steps", "bars", "comparison_table"]
    title: Label
    caption: Prose | None = None
    caption_evidence: Refs
    steps: list[StepSpec] | None = Field(default=None, min_length=1, max_length=_MAX_ENTRIES)
    unit: Unit | None = None
    basis: Basis | None = None
    assumptions: Prose | None = None
    max_value: NonNegative | None = None
    bars: list[BarSpec] | None = Field(default=None, min_length=1, max_length=_MAX_ENTRIES)
    columns: list[Label] | None = Field(default=None, min_length=1, max_length=_MAX_COLUMNS)
    rows: list[RowSpec] | None = Field(default=None, min_length=1, max_length=_MAX_ENTRIES)

    @model_validator(mode="after")
    def _variant_shape(self) -> AnswerVisualSpec:
        allowed = _VARIANT_FIELDS[self.variant]
        foreign = sorted(
            name
            for fields in _VARIANT_FIELDS.values()
            for name in fields - allowed
            if getattr(self, name) is not None
        )
        if foreign:
            raise ValueError(f"fields {foreign} do not belong to variant {self.variant!r}")
        missing = [name for name in _REQUIRED_FIELDS[self.variant] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"variant {self.variant!r} requires {missing}")
        if self.variant == "bars":
            _check_bars(self)
        if self.variant == "comparison_table":
            for index, row in enumerate(self.rows or ()):
                if len(row.cells) != len(self.columns or ()):
                    raise ValueError(
                        f"rows[{index}] has {len(row.cells)} cells for "
                        f"{len(self.columns or ())} columns"
                    )
        return self


def _check_bars(spec: AnswerVisualSpec) -> None:
    values = [bar.value for bar in spec.bars or ()]
    if spec.max_value is not None and (spec.max_value <= 0 or spec.max_value < max(values)):
        raise ValueError("max_value must be greater than zero and at least every bar value")
    if spec.basis != "measured" and spec.assumptions is None:
        raise ValueError(f"basis {spec.basis!r} requires assumptions stating how values were made")
    if spec.basis == "measured" and any(not bar.evidence for bar in spec.bars or ()):
        raise ValueError(
            "measured bars need evidence on every bar; cite a source, or use basis "
            "'calculated' or 'illustrative' with assumptions"
        )


# ---------------------------------------------------------------------------
# Public payload
# ---------------------------------------------------------------------------


class _Public(BaseModel):
    model_config = ConfigDict(extra="forbid")


Citations = Annotated[list[Annotated[int, Field(ge=1)]], Field(default_factory=list)]


class PublicEvidence(_Public):
    title: str | None = None
    url: str
    label: str


class StepItem(_Public):
    title: Label
    note: Prose | None = None
    citations: Citations


class BarItem(_Public):
    label: Label
    value: NonNegative
    citations: Citations


class BarScale(_Public):
    min: Literal[0] = 0
    max: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class TableCell(_Public):
    text: Label
    citations: Citations


class TableAssessment(_Public):
    status: AssessmentStatus
    label: Label
    reason: Prose
    citations: Citations


class TableRow(_Public):
    label: Label
    cells: list[TableCell] = Field(min_length=1, max_length=_MAX_COLUMNS)
    assessment: TableAssessment | None = None


class _PayloadBase(_Public):
    title: Label
    caption: Prose | None = None
    caption_citations: Citations
    evidence: list[PublicEvidence] = Field(default_factory=list)
    fallback_markdown: str = Field(min_length=1)

    def _citations(self) -> list[int]:
        return list(self.caption_citations)

    @model_validator(mode="after")
    def _citations_resolve(self) -> _PayloadBase:
        count = len(self.evidence)
        if any(number > count for number in self._citations()):
            raise ValueError("citation number has no evidence entry")
        return self


class StepsPayload(_PayloadBase):
    variant: Literal["steps"]
    steps: list[StepItem] = Field(min_length=1, max_length=_MAX_ENTRIES)

    def _citations(self) -> list[int]:
        return [*self.caption_citations, *(n for step in self.steps for n in step.citations)]


class BarsPayload(_PayloadBase):
    variant: Literal["bars"]
    unit: Unit
    basis: Basis
    assumptions: Prose | None = None
    scale: BarScale
    bars: list[BarItem] = Field(min_length=1, max_length=_MAX_ENTRIES)

    def _citations(self) -> list[int]:
        return [*self.caption_citations, *(n for bar in self.bars for n in bar.citations)]

    @model_validator(mode="after")
    def _shared_scale(self) -> BarsPayload:
        if any(bar.value > self.scale.max for bar in self.bars):
            raise ValueError("bar value exceeds the shared scale")
        if self.basis != "measured" and self.assumptions is None:
            raise ValueError("calculated or illustrative bars must disclose assumptions")
        return self


class ComparisonTablePayload(_PayloadBase):
    variant: Literal["comparison_table"]
    columns: list[Label] = Field(min_length=1, max_length=_MAX_COLUMNS)
    rows: list[TableRow] = Field(min_length=1, max_length=_MAX_ENTRIES)

    def _citations(self) -> list[int]:
        numbers = list(self.caption_citations)
        for row in self.rows:
            numbers.extend(n for cell in row.cells for n in cell.citations)
            if row.assessment is not None:
                numbers.extend(row.assessment.citations)
        return numbers

    @model_validator(mode="after")
    def _rectangular(self) -> ComparisonTablePayload:
        if any(len(row.cells) != len(self.columns) for row in self.rows):
            raise ValueError("every row needs one cell per column")
        return self


AnswerVisualPayload = Annotated[
    StepsPayload | BarsPayload | ComparisonTablePayload,
    Field(discriminator="variant"),
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


EvidenceResolver = Callable[[EvidenceRef], ResolvedEvidence | None]


class _EvidenceBinder:
    """Number each resolved source once, in order of first citation."""

    def __init__(self, resolve: EvidenceResolver) -> None:
        self._resolve = resolve
        self._numbers: dict[str, int] = {}
        self.evidence: list[dict[str, Any]] = []
        self.source_ids: list[str] = []

    def bind(self, refs: list[EvidenceRef]) -> list[int]:
        numbers: list[int] = []
        for ref in refs:
            key = f"{ref.kind}:{ref.id}"
            if key not in self._numbers:
                resolved = self._resolve(ref)
                if resolved is None:
                    raise AnswerVisualError(
                        f"unknown evidence {ref.id}; cite only S# IDs that web_search "
                        "returned in this turn, or omit evidence for illustrative content"
                    )
                self.evidence.append(
                    {
                        "title": resolved.title,
                        "url": resolved.url,
                        "label": _source_label(resolved.url),
                    }
                )
                self.source_ids.append(ref.id)
                self._numbers[key] = len(self.evidence)
            if self._numbers[key] not in numbers:
                numbers.append(self._numbers[key])
        return numbers


def build_answer_visual(
    spec: AnswerVisualSpec | dict[str, Any],
    *,
    resolve_evidence: EvidenceResolver,
    turn_key: str,
) -> dict[str, Any]:
    """Validate a visual and return its rich-item candidate.

    The candidate's ``_internal`` record names the turn-local source IDs the
    visual cites; finalization strips it before anything is persisted.

    Raises:
        AnswerVisualError: the input is invalid, cites unknown evidence, or its
            public form exceeds ``ANSWER_VISUAL_MAX_ITEM_BYTES``.
    """
    validated = _validate_spec(spec)
    binder = _EvidenceBinder(resolve_evidence)
    payload = _payload(validated, binder)
    payload["evidence"] = binder.evidence
    payload["fallback_markdown"] = fallback_markdown(payload)
    _validate_public_payload(payload)

    canonical = json.dumps(
        validated.model_dump(mode="json", exclude_none=True), sort_keys=True, ensure_ascii=False
    )
    digest = hashlib.sha256(f"{turn_key}\n{canonical}".encode()).hexdigest()[:24]
    item: dict[str, Any] = {
        "id": f"{ANSWER_VISUAL_ID_PREFIX}{digest}",
        "type": ANSWER_VISUAL_TYPE,
        "source": ANSWER_VISUAL_SOURCE,
        "display_policy": "inline_only",
        "title": validated.title,
        "payload": payload,
    }
    size = len(json.dumps(item, ensure_ascii=False).encode())
    if size > ANSWER_VISUAL_MAX_ITEM_BYTES:
        raise AnswerVisualError(
            f"visual is {size} bytes; the limit is {ANSWER_VISUAL_MAX_ITEM_BYTES} bytes. "
            "Shorten labels and notes or split it; never drop facts silently"
        )
    item["_internal"] = {"web_source_ids": list(binder.source_ids)}
    return item


def _validate_spec(spec: AnswerVisualSpec | dict[str, Any]) -> AnswerVisualSpec:
    if isinstance(spec, AnswerVisualSpec):
        return spec
    try:
        return AnswerVisualSpec.model_validate(spec)
    except ValidationError as exc:
        raise AnswerVisualError(_compact_errors(exc)) from None


def _validate_public_payload(payload: dict[str, Any]) -> None:
    from pydantic import TypeAdapter

    try:
        TypeAdapter(AnswerVisualPayload).validate_python(payload)
    except ValidationError as exc:
        raise AnswerVisualError(_compact_errors(exc)) from None


def _compact_errors(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "invalid visual"


def _payload(spec: AnswerVisualSpec, binder: _EvidenceBinder) -> dict[str, Any]:
    payload: dict[str, Any] = {"variant": spec.variant, "title": spec.title}
    if spec.caption is not None:
        payload["caption"] = spec.caption
    payload["caption_citations"] = binder.bind(spec.caption_evidence)
    if spec.variant == "steps":
        payload["steps"] = [
            _without_none(
                {"title": step.title, "note": step.note, "citations": binder.bind(step.evidence)}
            )
            for step in spec.steps or ()
        ]
    elif spec.variant == "bars":
        bars = spec.bars or []
        derived_max = max(bar.value for bar in bars) or 1
        payload.update(
            _without_none(
                {
                    "unit": spec.unit,
                    "basis": spec.basis,
                    "assumptions": spec.assumptions,
                    "scale": {"min": 0, "max": spec.max_value or derived_max},
                }
            )
        )
        payload["bars"] = [
            {"label": bar.label, "value": bar.value, "citations": binder.bind(bar.evidence)}
            for bar in bars
        ]
    else:
        payload["columns"] = list(spec.columns or ())
        payload["rows"] = [_row(row, binder) for row in spec.rows or ()]
    return payload


def _row(row: RowSpec, binder: _EvidenceBinder) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": row.label,
        "cells": [
            {"text": cell.text, "citations": binder.bind(cell.evidence)} for cell in row.cells
        ],
    }
    if row.assessment is not None:
        out["assessment"] = {
            "status": row.assessment.status,
            "label": row.assessment.label,
            "reason": row.assessment.reason,
            "citations": binder.bind(row.assessment.evidence),
        }
    return out


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _source_label(url: str) -> str:
    host = (urlsplit(url).hostname or "source").lower()
    return host.removeprefix("www.")


# ---------------------------------------------------------------------------
# Fallback Markdown
# ---------------------------------------------------------------------------


def fallback_markdown(payload: dict[str, Any]) -> str:
    """Render a validated public payload as evidence-bearing Markdown.

    The one fallback formatter: stream, history, and copy/export all use the
    ``fallback_markdown`` stored with the payload rather than re-rendering.
    """
    evidence = payload.get("evidence") or []
    blocks = [f"**{_text(payload['title'])}**"]
    variant = payload["variant"]
    if variant == "steps":
        blocks.append(_steps_markdown(payload, evidence))
    elif variant == "bars":
        blocks.extend(_bars_markdown(payload, evidence))
    else:
        blocks.append(_table_markdown(payload, evidence))
    caption = payload.get("caption")
    if caption:
        blocks.append(_with_citations(_text(caption), payload.get("caption_citations"), evidence))
    return "\n\n".join(blocks)


def _steps_markdown(payload: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    lines = []
    for number, step in enumerate(payload["steps"], start=1):
        text = _text(step["title"])
        if step.get("note"):
            text = f"{text} — {_text(step['note'])}"
        lines.append(f"{number}. {_with_citations(text, step.get('citations'), evidence)}")
    return "\n".join(lines)


def _bars_markdown(payload: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    unit = _cell(payload["unit"])
    cited = any(bar.get("citations") for bar in payload["bars"])
    header = ["Item", f"Value ({unit})", *(["Sources"] if cited else [])]
    rows = []
    for bar in payload["bars"]:
        row = [_cell(bar["label"]), _number(bar["value"])]
        if cited:
            row.append(_citation_links(bar.get("citations"), evidence))
        rows.append(row)
    basis = f"Basis: {payload['basis']}"
    if payload.get("assumptions"):
        basis = f"{basis} — {_text(payload['assumptions'])}"
    if not basis.endswith((".", "!", "?")):
        basis += "."
    scale = f"Scale: 0 to {_number(payload['scale']['max'])} {_text(payload['unit'])}."
    return [_markdown_table(header, rows), f"{basis} {scale}"]


def _table_markdown(payload: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    assessed = any(row.get("assessment") for row in payload["rows"])
    header = ["Item", *(_cell(column) for column in payload["columns"])]
    if assessed:
        header.append("Assessment")
    rows = []
    for row in payload["rows"]:
        cells = [_cell(row["label"])]
        cells.extend(
            _with_citations(_cell(cell["text"]), cell.get("citations"), evidence)
            for cell in row["cells"]
        )
        if assessed:
            assessment = row.get("assessment")
            cells.append(_assessment_text(assessment, evidence) if assessment else "")
        rows.append(cells)
    return _markdown_table(header, rows)


def _assessment_text(assessment: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    text = (
        f"{assessment['status'].capitalize()}: {_cell(assessment['label'])} — "
        f"{_cell(assessment['reason'])}"
    )
    return _with_citations(text, assessment.get("citations"), evidence)


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _with_citations(text: str, numbers: Any, evidence: list[dict[str, Any]]) -> str:
    links = _citation_links(numbers, evidence)
    return f"{text} {links}" if links else text


def _citation_links(numbers: Any, evidence: list[dict[str, Any]]) -> str:
    links = []
    for number in numbers or ():
        source = evidence[number - 1]
        url = str(source["url"]).replace("(", "%28").replace(")", "%29").replace(" ", "%20")
        links.append(f"[{_cell(source['label'])}]({url})")
    return " ".join(links)


def _text(value: Any) -> str:
    """Model text as inert Markdown: one line, and never a rich marker."""
    return " ".join(str(value).split()).replace("<!--", "&lt;!--")


def _cell(value: Any) -> str:
    return _text(value).replace("|", "\\|")


def _number(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".15g")


__all__ = [
    "ANSWER_VISUAL_ID_PREFIX",
    "ANSWER_VISUAL_MAX_ITEM_BYTES",
    "ANSWER_VISUAL_MAX_PER_ANSWER",
    "ANSWER_VISUAL_MAX_TOTAL_BYTES",
    "ANSWER_VISUAL_SOURCE",
    "ANSWER_VISUAL_TYPE",
    "AnswerVisualError",
    "AnswerVisualPayload",
    "AnswerVisualSpec",
    "BarsPayload",
    "ComparisonTablePayload",
    "EvidenceRef",
    "ResolvedEvidence",
    "StepsPayload",
    "build_answer_visual",
    "fallback_markdown",
]
