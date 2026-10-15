from __future__ import annotations

from dataclasses import replace

from .models import ContextChunk, DocumentElement, ExtractedDocument, SourceLocation
from .utils import approximate_tokens, stable_chunk_id


def _location_key(location: SourceLocation) -> str:
    return "|".join(
        str(value or "")
        for value in (
            location.page,
            location.sheet,
            location.section,
            location.row_start,
            location.row_end,
            location.cell_range,
        )
    )


def _merge_location(first: SourceLocation, last: SourceLocation) -> SourceLocation:
    return SourceLocation(
        page=first.page if first.page == last.page else first.page,
        sheet=first.sheet if first.sheet == last.sheet else first.sheet,
        section=first.section or last.section,
        row_start=first.row_start,
        row_end=last.row_end or last.row_start or first.row_end,
        cell_range=(
            first.cell_range
            if first.cell_range == last.cell_range
            else _merge_cell_ranges(first.cell_range, last.cell_range)
        ),
        bbox=first.bbox if first == last else None,
    )


def _merge_cell_ranges(first: str | None, last: str | None) -> str | None:
    if not first:
        return last
    if not last:
        return first
    return f"{first.split(':')[0]}:{last.split(':')[-1]}"


def chunk_document(
    document: ExtractedDocument, min_tokens: int = 350, max_tokens: int = 600
) -> list[ContextChunk]:
    del min_tokens  # A soft target: structural boundaries take precedence.
    chunks: list[ContextChunk] = []
    buffer: list[DocumentElement] = []
    buffer_tokens = 0
    table_headers: dict[str, DocumentElement] = {}

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        heading = buffer[0].heading_path
        prefix = "\n".join(f"# {part}" for part in heading)
        body = "\n\n".join(item.text.strip() for item in buffer if item.text.strip())
        text = f"{prefix}\n\n{body}".strip() if prefix else body
        location = _merge_location(buffer[0].location, buffer[-1].location)
        sequence = len(chunks)
        structural_key = f"{document.relative_path}:{_location_key(location)}:{' > '.join(heading)}:{buffer[0].kind}"
        chunks.append(
            ContextChunk(
                id=stable_chunk_id(document.content_hash, sequence, structural_key),
                file_hash=document.content_hash,
                sequence=sequence,
                text=text,
                token_count=approximate_tokens(text),
                kind=buffer[0].kind if all(x.kind == buffer[0].kind for x in buffer) else "mixed",
                heading_path=heading,
                location=location,
            )
        )
        buffer = []
        buffer_tokens = 0

    for element in document.elements:
        if not element.text.strip():
            continue
        if element.kind == "table_row" and element.metadata.get("role") == "header" and element.location.sheet:
            table_headers[element.location.sheet] = element
        tokens = approximate_tokens(element.text)
        boundary_changed = bool(buffer) and (
            element.heading_path != buffer[0].heading_path
            or element.location.sheet != buffer[0].location.sheet
            or (element.kind == "table_row" and buffer[0].kind != "table_row")
            or (element.kind != "table_row" and buffer[0].kind == "table_row")
        )
        if boundary_changed or (buffer and buffer_tokens + tokens > max_tokens):
            flush()
            if element.kind == "table_row" and element.metadata.get("role") != "header":
                header = table_headers.get(element.location.sheet or "")
                if header is not None:
                    buffer.append(header)
                    buffer_tokens = approximate_tokens(header.text)

        # Oversized prose is divided on words. Table rows remain indivisible.
        if tokens > max_tokens and element.kind != "table_row":
            words = element.text.split()
            part: list[str] = []
            for word in words:
                if part and approximate_tokens(" ".join(part + [word])) > max_tokens:
                    piece = replace(element, text=" ".join(part))
                    buffer = [piece]
                    buffer_tokens = approximate_tokens(piece.text)
                    flush()
                    part = []
                part.append(word)
            if part:
                buffer = [replace(element, text=" ".join(part))]
                buffer_tokens = approximate_tokens(buffer[0].text)
            continue

        buffer.append(element)
        buffer_tokens += tokens
    flush()
    return chunks

