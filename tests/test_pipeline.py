from __future__ import annotations

from pathlib import Path

from app.models import DocumentElement, ExtractedDocument, FileState, SourceLocation
from app.pipeline import IngestionPipeline


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def extract(self, path: Path, relative_path: str, content_hash: str) -> ExtractedDocument:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated extraction failure")
        return ExtractedDocument(
            path=path, relative_path=relative_path, content_hash=content_hash,
            title=path.stem, format=path.suffix.lstrip("."), parser_version="fake-1",
            elements=[DocumentElement("text", path.read_text(), SourceLocation(page=1))],
            raw_document={"text": path.read_text()},
        )


def test_create_unchanged_update_rename_delete(settings) -> None:
    settings.prepare()
    extractor = FakeExtractor()
    pipeline = IngestionPipeline(settings, extractor=extractor)
    source = settings.source_dir / "one.pdf"
    source.write_text("initial content", encoding="utf-8")

    result = pipeline.reconcile()
    assert result[0].action == "created"
    assert extractor.calls == 1
    assert pipeline.reconcile() == []
    assert extractor.calls == 1

    source.write_text("updated content", encoding="utf-8")
    result = pipeline.reconcile()
    assert result[0].action == "updated"
    assert extractor.calls == 2

    renamed = settings.source_dir / "renamed.pdf"
    source.rename(renamed)
    result = pipeline.reconcile()
    assert [item.action for item in result] == ["renamed"]
    assert extractor.calls == 2
    assert pipeline.repository.get_file("one.pdf") is None
    assert pipeline.repository.get_file("renamed.pdf") is not None

    renamed.unlink()
    result = pipeline.reconcile()
    assert result[0].action == "deleted"
    assert pipeline.repository.list_files() == []


def test_failed_update_preserves_last_good_chunks(settings) -> None:
    settings.prepare()
    extractor = FakeExtractor()
    pipeline = IngestionPipeline(settings, extractor=extractor)
    source = settings.source_dir / "one.pdf"
    source.write_text("good", encoding="utf-8")
    pipeline.reconcile()
    old_text = pipeline.repository.chunks_for_file("one.pdf")[0].text

    extractor.fail = True
    source.write_text("bad replacement", encoding="utf-8")
    result = pipeline.reconcile()
    assert result[0].action == "failed"
    record = pipeline.repository.get_file("one.pdf")
    assert record and record.state == FileState.READY and record.last_error
    assert pipeline.repository.chunks_for_file("one.pdf")[0].text == old_text


def test_bad_and_unsupported_files_do_not_block_others(settings) -> None:
    settings.prepare()
    extractor = FakeExtractor()
    pipeline = IngestionPipeline(settings, extractor=extractor)
    (settings.source_dir / "good.pdf").write_text("good", encoding="utf-8")
    (settings.source_dir / "notes.txt").write_text("unsupported", encoding="utf-8")
    results = pipeline.reconcile()
    assert {result.action for result in results} == {"created", "unsupported"}
    assert pipeline.repository.get_file("good.pdf").state == FileState.READY
    assert pipeline.repository.get_file("notes.txt").state == FileState.FAILED

