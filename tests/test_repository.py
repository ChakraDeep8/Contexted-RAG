from pathlib import Path

from app.models import ContextChunk, FileState, SourceLocation
from app.repository import SQLiteRepository


def sample_chunk(identifier: str = "chunk-1", text: str = "hello") -> ContextChunk:
    return ContextChunk(
        id=identifier, file_hash="h" * 64, sequence=0, text=text, token_count=2,
        kind="text", heading_path=("Title",), location=SourceLocation(page=3, section="Title"),
    )


def test_replace_and_resolve_document(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "db.sqlite")
    repo.replace_document("a.pdf", "h" * 64, 10, 1, "pdf", "test", [sample_chunk()])
    record = repo.get_file("a.pdf")
    assert record and record.state == FileState.READY and record.chunk_count == 1
    resolved = repo.resolve_chunk("chunk-1")
    assert resolved and resolved[0].path == "a.pdf"
    assert resolved[1].location.page == 3


def test_failed_update_can_leave_previous_chunks(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "db.sqlite")
    repo.replace_document("a.pdf", "h" * 64, 10, 1, "pdf", "test", [sample_chunk()])
    repo.mark_processing("a.pdf", 20, 2, "pdf")
    repo.mark_failed("a.pdf", 20, 2, "pdf", "broken")
    record = repo.get_file("a.pdf")
    assert record and record.state == FileState.READY and record.last_error == "broken"
    assert repo.chunks_for_file("a.pdf")[0].text == "hello"


def test_delete_cascades_chunks(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "db.sqlite")
    repo.replace_document("a.pdf", "h" * 64, 10, 1, "pdf", "test", [sample_chunk()])
    assert repo.delete_file("a.pdf") == "h" * 64
    assert repo.list_active_chunks() == []

