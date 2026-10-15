from pathlib import Path

from app.chunking import chunk_document
from app.models import DocumentElement, ExtractedDocument, SourceLocation


def make_document(elements: list[DocumentElement]) -> ExtractedDocument:
    return ExtractedDocument(
        path=Path("report.pdf"), relative_path="report.pdf", content_hash="a" * 64,
        title="Report", format="pdf", elements=elements,
    )


def test_chunk_ids_are_stable_and_locations_are_preserved() -> None:
    document = make_document([
        DocumentElement("text", "First paragraph " * 8, SourceLocation(page=2, section="Overview"), ("Overview",)),
        DocumentElement("text", "Second paragraph " * 8, SourceLocation(page=2, section="Overview"), ("Overview",)),
    ])
    first = chunk_document(document, 20, 50)
    second = chunk_document(document, 20, 50)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].location.page == 2
    assert "# Overview" in first[0].text


def test_oversized_table_row_is_never_split() -> None:
    row = "value " * 200
    chunks = chunk_document(make_document([
        DocumentElement("table_row", row, SourceLocation(sheet="Data", row_start=7, row_end=7), ("Data",))
    ]), 20, 50)
    assert len(chunks) == 1
    assert chunks[0].location.row_start == 7
    assert row.strip() in chunks[0].text


def test_table_header_is_repeated_after_chunk_split() -> None:
    elements = [
        DocumentElement(
            "table_row", "Name | Description", SourceLocation(sheet="Data", row_start=1, row_end=1),
            ("Data",), {"role": "header"},
        )
    ]
    elements.extend(
        DocumentElement(
            "table_row", f"Item {index} | " + ("detail " * 20),
            SourceLocation(sheet="Data", row_start=index, row_end=index),
            ("Data",), {"role": "data"},
        )
        for index in range(2, 5)
    )
    chunks = chunk_document(make_document(elements), 20, 50)
    assert len(chunks) > 1
    assert all("Name | Description" in chunk.text for chunk in chunks)

