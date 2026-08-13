"""Normalize parser-specific output into immutable structural blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from app.services.document_blocks import NormalizedBlock


class DocumentNormalizer:
    """The single parser-output normalization boundary."""

    def normalize_text(self, text: str, *, source: str | None = None) -> list[NormalizedBlock]:
        if not text:
            return []
        metadata = {"source": source} if source else {}
        return [NormalizedBlock("text:0", "paragraph", text, metadata=metadata)]

    def normalize_markdown(
        self, markdown: str, *, source: str | None = None
    ) -> list[NormalizedBlock]:
        blocks: list[NormalizedBlock] = []
        section_levels: list[str] = []
        paragraph_lines: list[str] = []

        def emit_paragraph() -> None:
            if not paragraph_lines:
                return
            text = "\n".join(paragraph_lines).strip()
            paragraph_lines.clear()
            if text:
                metadata = {"source": source, "source_type": "markdown"}
                blocks.append(
                    NormalizedBlock(
                        f"markdown:{len(blocks)}",
                        "paragraph",
                        text,
                        section_path=tuple(section_levels),
                        metadata={
                            key: value for key, value in metadata.items() if value is not None
                        },
                    )
                )

        for line in markdown.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                emit_paragraph()
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                section_levels[level - 1 :] = [heading]
                metadata = {"source_type": "markdown", "heading_level": level}
                if source:
                    metadata["source"] = source
                blocks.append(
                    NormalizedBlock(
                        f"markdown:{len(blocks)}",
                        "heading",
                        heading,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
            elif line.strip():
                paragraph_lines.append(line)
            else:
                emit_paragraph()
        emit_paragraph()
        return blocks

    def normalize_mineru(
        self,
        entries: Iterable[Mapping[str, Any]],
        *,
        images_data: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[NormalizedBlock]:
        blocks: list[NormalizedBlock] = []
        section_levels: list[str] = []
        image_entries = list(images_data or [])

        for source_index, entry in enumerate(entries):
            source_type = str(entry.get("type", "text") or "text").lower()
            parser_page_idx = self._optional_int(entry.get("page_idx"))
            page = parser_page_idx + 1 if parser_page_idx is not None else None
            metadata: dict[str, Any] = {
                "source_type": source_type,
                "source_index": source_index,
            }
            if parser_page_idx is not None:
                metadata["parser_page_idx"] = parser_page_idx
            if "bbox" in entry:
                metadata["bbox"] = entry.get("bbox")

            if source_type == "text":
                text = str(entry.get("text", "") or "").strip()
                if not text:
                    continue
                level = self._optional_int(entry.get("text_level")) or 0
                if level > 0:
                    section_levels[level - 1 :] = [text]
                    kind = "heading"
                    metadata["heading_level"] = level
                else:
                    kind = "paragraph"
                blocks.append(
                    NormalizedBlock(
                        f"mineru:{source_index}",
                        kind,
                        text,
                        page_start=page,
                        page_end=page,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
                continue

            if source_type == "table":
                caption = self._as_list(entry.get("table_caption"))
                header = entry.get("table_header", [])
                body = entry.get("table_body", "")
                footnote = self._as_list(entry.get("table_footnote"))
                metadata.update(
                    {"caption": caption, "header": header, "body": body, "footnote": footnote}
                )
                if "img_path" in entry:
                    metadata["img_path"] = entry.get("img_path")
                text = self._table_text(caption, header, body, footnote)
                blocks.append(
                    NormalizedBlock(
                        f"mineru:{source_index}",
                        "table",
                        text or "[Table]",
                        page_start=page,
                        page_end=page,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
                continue

            if source_type == "image":
                img_path = str(entry.get("img_path", "") or "")
                caption = self._as_list(entry.get("image_caption"))
                footnote = self._as_list(entry.get("image_footnote"))
                metadata.update(
                    {"img_path": img_path, "caption": caption, "footnote": footnote}
                )
                matched_image = self._match_image(img_path, page, image_entries)
                if matched_image:
                    metadata.update(matched_image)
                description = " ".join(str(item) for item in [*caption, *footnote] if item).strip()
                text = f"[Image: {description}]" if description else "[Image]"
                blocks.append(
                    NormalizedBlock(
                        f"mineru:{source_index}",
                        "image",
                        text,
                        page_start=page,
                        page_end=page,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
                continue

            if source_type == "equation":
                text = str(entry.get("text", "") or entry.get("latex", "") or "[Equation]")
                for key in ("text_format", "latex"):
                    if key in entry:
                        metadata[key] = entry[key]
                blocks.append(
                    NormalizedBlock(
                        f"mineru:{source_index}",
                        "equation",
                        text,
                        page_start=page,
                        page_end=page,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
                continue

            text = str(entry.get("text", "") or "").strip()
            if text:
                blocks.append(
                    NormalizedBlock(
                        f"mineru:{source_index}",
                        "paragraph",
                        text,
                        page_start=page,
                        page_end=page,
                        section_path=tuple(section_levels),
                        metadata=metadata,
                    )
                )
        return blocks

    def normalize_excel(
        self,
        sheets: Iterable[tuple[str, Sequence[Sequence[str]]]],
        *,
        source: str | None = None,
    ) -> list[NormalizedBlock]:
        blocks: list[NormalizedBlock] = []
        for sheet_index, (sheet_name, rows) in enumerate(sheets):
            if not rows:
                continue
            heading = f"Sheet: {sheet_name}"
            display_page = sheet_index + 1
            common = {"source_type": "excel", "sheet_name": sheet_name, "sheet_index": sheet_index}
            if source:
                common["source"] = source
            blocks.append(
                NormalizedBlock(
                    f"excel:{sheet_index}:heading",
                    "heading",
                    heading,
                    page_start=display_page,
                    page_end=display_page,
                    section_path=(heading,),
                    metadata={**common, "heading_level": 1},
                )
            )
            normalized_rows = [[str(cell) for cell in row] for row in rows]
            header = normalized_rows[0]
            body = normalized_rows[1:]
            table_text = self._excel_table_markdown(header, body)
            blocks.append(
                NormalizedBlock(
                    f"excel:{sheet_index}:table:0",
                    "table",
                    table_text,
                    page_start=display_page,
                    page_end=display_page,
                    section_path=(heading,),
                    metadata={
                        **common,
                        "caption": [heading],
                        "header": header,
                        "body": body,
                        "footnote": [],
                    },
                )
            )
        return blocks

    def normalize_legacy_chunks(
        self, chunks: Iterable[Mapping[str, Any] | str]
    ) -> list[NormalizedBlock]:
        blocks: list[NormalizedBlock] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                text = str(chunk)
                data: Mapping[str, Any] = {}
            else:
                data = chunk
                text = str(data.get("text", "") or "")
                table_texts = []
                for table in data.get("tables") or []:
                    if isinstance(table, Mapping):
                        body = str(table.get("body") or table.get("table_body") or "").strip()
                        if body and body not in text:
                            table_texts.append(body)
                if table_texts:
                    text = "\n".join(part for part in [text.rstrip(), *table_texts] if part)
            if not text.strip():
                continue
            metadata = {
                key: value
                for key, value in data.items()
                if key not in {"text", "page_start", "page_end", "section_path"}
            }
            metadata.setdefault("source_chunk_index", index)
            metadata["legacy_prechunked"] = True
            parser_page_start = data.get("page_start")
            parser_page_end = data.get("page_end")
            if parser_page_start is not None:
                metadata["parser_page_start"] = parser_page_start
            if parser_page_end is not None:
                metadata["parser_page_end"] = parser_page_end
            kind = "table" if data.get("has_tables") else "paragraph"
            blocks.append(
                NormalizedBlock(
                    f"legacy:{index}",
                    kind,
                    text,
                    page_start=(
                        int(parser_page_start) + 1 if parser_page_start is not None else None
                    ),
                    page_end=(int(parser_page_end) + 1 if parser_page_end is not None else None),
                    section_path=tuple(data.get("section_path") or ()),
                    metadata=metadata,
                )
            )
        return blocks

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @classmethod
    def _table_text(cls, caption: list[Any], header: Any, body: Any, footnote: list[Any]) -> str:
        parts: list[str] = []
        if caption:
            parts.append("[Table: " + " ".join(str(item) for item in caption if item) + "]")
        if header:
            if isinstance(header, (list, tuple)):
                parts.append("| " + " | ".join(str(item) for item in header) + " |")
            else:
                parts.append(str(header))
        if body:
            parts.append(str(body))
        if footnote:
            note = " ".join(str(item) for item in footnote if item)
            parts.append(f"[Table footnote: {note}]")
        return "\n".join(parts)

    @staticmethod
    def _excel_table_markdown(header: list[str], body: list[list[str]]) -> str:
        width = max([len(header), *(len(row) for row in body)], default=0)
        escape = DocumentNormalizer._escape_markdown_cell
        padded_header = header + [f"Column {index + 1}" for index in range(len(header), width)]
        lines = [
            "| " + " | ".join(escape(cell) for cell in padded_header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for row in body:
            padded = row + [""] * (width - len(row))
            lines.append("| " + " | ".join(escape(cell) for cell in padded) + " |")
        return "\n".join(lines)

    @staticmethod
    def _escape_markdown_cell(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _match_image(
        img_path: str,
        page: int | None,
        images_data: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        wanted = PurePosixPath(img_path.replace("\\", "/")) if img_path else None
        for image in images_data:
            candidates = [image.get("relative_path"), image.get("img_path"), image.get("path")]
            if wanted and any(
                candidate
                and (
                    PurePosixPath(str(candidate).replace("\\", "/")) == wanted
                    or str(candidate).replace("\\", "/").endswith(str(wanted))
                )
                for candidate in candidates
            ):
                return dict(image)
        for image in images_data:
            if page is not None and image.get("page_number") == page:
                return dict(image)
        return None
