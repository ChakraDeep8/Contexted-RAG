from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Protocol


class FileState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class SourceLocation:
    page: int | None = None
    sheet: str | None = None
    section: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    cell_range: str | None = None
    bbox: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentElement:
    kind: str
    text: str
    location: SourceLocation = field(default_factory=SourceLocation)
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedDocument:
    path: Path
    relative_path: str
    content_hash: str
    title: str
    format: str
    elements: list[DocumentElement]
    warnings: list[str] = field(default_factory=list)
    parser_version: str = "unknown"
    raw_document: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextChunk:
    id: str
    file_hash: str
    sequence: int
    text: str
    token_count: int
    kind: str
    heading_path: tuple[str, ...]
    location: SourceLocation
    needs_embedding: bool = True


@dataclass(slots=True)
class FileRecord:
    id: int
    path: str
    content_hash: str | None
    size: int
    modified_ns: int
    format: str
    state: FileState
    parser_version: str | None = None
    chunk_count: int = 0
    last_error: str | None = None
    indexed_at: str | None = None


@dataclass(slots=True)
class IngestionResult:
    path: str
    state: FileState
    action: str
    chunk_count: int = 0
    error: str | None = None


class Repository(Protocol):
    def list_files(self) -> list[FileRecord]: ...
    def list_active_chunks(self) -> list[ContextChunk]: ...
    def chunks_for_file(self, path: str) -> list[ContextChunk]: ...
    def chunks_requiring_embeddings(self, limit: int | None = None) -> Iterator[ContextChunk]: ...
    def resolve_chunk(self, chunk_id: str) -> tuple[FileRecord, ContextChunk] | None: ...
    def delete_file(self, path: str) -> str | None: ...

