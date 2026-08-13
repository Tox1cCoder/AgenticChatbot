from __future__ import annotations

from app.services.document_normalizer import DocumentNormalizer


def test_mineru_heading_carries_section_path_to_later_blocks():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_mineru(
        [
            {"type": "text", "text_level": 1, "text": "Revenue", "page_idx": 0},
            {
                "type": "text",
                "text": "Quarterly results",
                "page_idx": 0,
                "bbox": [1, 2, 3, 4],
            },
        ]
    )

    assert blocks[0].kind == "heading"
    assert blocks[1].section_path == ("Revenue",)
    assert blocks[1].metadata["bbox"] == [1, 2, 3, 4]


def test_mineru_first_page_builds_one_based_chunk_and_retains_raw_index():
    from app.services.document_chunk_builder import DocumentChunkBuilder

    blocks = DocumentNormalizer().normalize_mineru(
        [{"type": "text", "text": "First page", "page_idx": 0}]
    )
    chunks = DocumentChunkBuilder().build(blocks)

    assert blocks[0].page_start == 1
    assert blocks[0].metadata["parser_page_idx"] == 0
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_table_retains_caption_header_body_and_footnote():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_mineru(
        [
            {
                "type": "table",
                "page_idx": 2,
                "bbox": [10, 20, 30, 40],
                "table_caption": ["Revenue by segment"],
                "table_header": ["Segment", "Revenue"],
                "table_body": "| Cloud | 12345 |",
                "table_footnote": ["USD millions"],
            }
        ]
    )

    table = blocks[0]
    assert table.kind == "table"
    assert table.page_start == 3
    assert table.page_end == 3
    assert table.metadata["parser_page_idx"] == 2
    assert table.metadata.keys() >= {"caption", "header", "body", "footnote", "bbox"}
    assert "Cloud" in table.text
    assert "USD millions" in table.text


def test_image_retains_parser_and_image_file_provenance():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_mineru(
        [
            {
                "type": "image",
                "page_idx": 4,
                "bbox": [5, 6, 7, 8],
                "img_path": "images/chart.png",
                "image_caption": ["Quarterly revenue chart"],
                "image_footnote": ["Source: Finance"],
            }
        ],
        images_data=[
            {
                "path": "/tmp/chart.png",
                "relative_path": "images/chart.png",
                "page_number": 4,
                "mime_type": "image/png",
            }
        ],
    )

    image = blocks[0]
    assert image.kind == "image"
    assert image.metadata["bbox"] == [5, 6, 7, 8]
    assert image.metadata["img_path"] == "images/chart.png"
    assert image.metadata["path"] == "/tmp/chart.png"
    assert image.metadata["mime_type"] == "image/png"
    assert image.metadata["caption"] == ["Quarterly revenue chart"]
    assert image.metadata["footnote"] == ["Source: Finance"]


def test_equation_retains_page_bbox_and_section_context():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_mineru(
        [
            {"type": "text", "text_level": 2, "text": "Method", "page_idx": 1},
            {
                "type": "equation",
                "text": "E = mc^2",
                "text_format": "latex",
                "page_idx": 1,
                "bbox": [11, 12, 13, 14],
            },
        ]
    )

    equation = blocks[1]
    assert equation.kind == "equation"
    assert equation.section_path == ("Method",)
    assert equation.metadata["bbox"] == [11, 12, 13, 14]
    assert equation.metadata["text_format"] == "latex"


def test_excel_sheet_becomes_heading_followed_by_table():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_excel(
        [("Q1", [["Region", "Revenue"], ["APAC", "120"]])],
        source="forecast.xlsx",
    )

    assert [block.kind for block in blocks] == ["heading", "table"]
    assert blocks[0].text == "Sheet: Q1"
    assert blocks[1].section_path == ("Sheet: Q1",)
    assert blocks[1].metadata["sheet_name"] == "Q1"
    assert blocks[1].metadata["header"] == ["Region", "Revenue"]
    assert blocks[1].metadata["body"] == [["APAC", "120"]]
    assert "| APAC | 120 |" in blocks[1].text


def test_excel_first_sheet_builds_one_based_chunks_and_retains_raw_index():
    from app.services.document_chunk_builder import DocumentChunkBuilder

    blocks = DocumentNormalizer().normalize_excel([("Q1", [["Revenue"], ["120"]])])
    chunks = DocumentChunkBuilder().build(blocks)

    assert all(block.page_start == 1 for block in blocks)
    assert all(block.metadata["sheet_index"] == 0 for block in blocks)
    assert all(chunk.page_start == 1 for chunk in chunks)


def test_markdown_fallback_emits_heading_and_paragraph_blocks():
    normalizer = DocumentNormalizer()

    blocks = normalizer.normalize_markdown("# Revenue\n\nQuarterly results\ncontinue here")

    assert [block.kind for block in blocks] == ["heading", "paragraph"]
    assert blocks[1].section_path == ("Revenue",)
    assert blocks[1].text == "Quarterly results\ncontinue here"
