from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from watchfiles import awatch

from .chunking import chunk_document
from .config import Settings
from .extractors import DoclingExtractor, Extractor
from .models import FileState, IngestionResult
from .repository import SQLiteRepository
from .utils import SUPPORTED_EXTENSIONS, atomic_write_json, normalize_relative, sha256_file, wait_until_stable

LOGGER = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.settings = settings
        self.settings.prepare()
        self.repository = repository or SQLiteRepository(settings.database_path)
        self.extractor = extractor or DoclingExtractor()

    def discover(self) -> dict[str, Path]:
        discovered: dict[str, Path] = {}
        for path in self.settings.source_dir.rglob("*"):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(self.settings.source_dir).parts):
                discovered[normalize_relative(path, self.settings.source_dir)] = path
        return discovered

    def reconcile(self) -> list[IngestionResult]:
        discovered = self.discover()
        present_paths = set(discovered)
        results: list[IngestionResult] = []

        # Process additions/updates before deletions so content-identical renames are detected.
        for relative_path, absolute_path in sorted(discovered.items()):
            if absolute_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                if self.repository.get_file(relative_path) is None:
                    stat = absolute_path.stat()
                    message = f"Unsupported file type: {absolute_path.suffix or '(none)'}"
                    self.repository.mark_failed(relative_path, stat.st_size, stat.st_mtime_ns, absolute_path.suffix.lower(), message)
                    results.append(IngestionResult(relative_path, FileState.FAILED, "unsupported", error=message))
                continue
            result = self.ingest_file(absolute_path, present_paths)
            if result.action != "unchanged":
                results.append(result)

        for record in self.repository.list_files():
            if record.path not in present_paths:
                orphan_hash = self.repository.delete_file(record.path)
                if orphan_hash:
                    self._artifact_path(orphan_hash).unlink(missing_ok=True)
                results.append(IngestionResult(record.path, FileState.READY, "deleted"))
                LOGGER.info("Removed deleted source: %s", record.path)
        return results

    def ingest_file(self, path: Path, present_paths: set[str] | None = None) -> IngestionResult:
        relative_path = normalize_relative(path, self.settings.source_dir)
        present_paths = present_paths if present_paths is not None else set(self.discover())
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            stat = path.stat()
            error = f"Unsupported file type: {suffix or '(none)'}"
            self.repository.mark_failed(relative_path, stat.st_size, stat.st_mtime_ns, suffix, error)
            return IngestionResult(relative_path, FileState.FAILED, "unsupported", error=error)

        existing = self.repository.get_file(relative_path)
        try:
            wait_until_stable(path, self.settings.stable_checks, self.settings.stable_check_seconds)
            stat = path.stat()
            content_hash = sha256_file(path)
        except Exception as exc:
            return self._record_failure(relative_path, path, suffix, exc, "stability-check")

        if existing and existing.content_hash == content_hash and existing.state == FileState.READY:
            if existing.size != stat.st_size or existing.modified_ns != stat.st_mtime_ns:
                self.repository.rename_file(relative_path, relative_path, stat.st_size, stat.st_mtime_ns)
            return IngestionResult(relative_path, FileState.READY, "unchanged", existing.chunk_count)

        renamed = self.repository.find_missing_by_hash(content_hash, present_paths)
        if renamed:
            self.repository.rename_file(renamed.path, relative_path, stat.st_size, stat.st_mtime_ns)
            LOGGER.info("Renamed source without re-extraction: %s -> %s", renamed.path, relative_path)
            return IngestionResult(relative_path, FileState.READY, "renamed", renamed.chunk_count)

        action = "updated" if existing and existing.chunk_count else "created"
        job_id = self.repository.create_job(relative_path, action)
        self.repository.mark_processing(relative_path, stat.st_size, stat.st_mtime_ns, suffix.lstrip("."))
        last_error: Exception | None = None
        for attempt in range(1, self.settings.retry_attempts + 1):
            self.repository.start_job(job_id, attempt)
            try:
                document = self.extractor.extract(path, relative_path, content_hash)
                chunks = chunk_document(
                    document,
                    min_tokens=self.settings.min_chunk_tokens,
                    max_tokens=self.settings.max_chunk_tokens,
                )
                if not chunks:
                    raise ValueError("Extraction produced no indexable chunks")
                atomic_write_json(
                    self._artifact_path(content_hash),
                    {
                        "relative_path": relative_path,
                        "content_hash": content_hash,
                        "title": document.title,
                        "format": document.format,
                        "parser_version": document.parser_version,
                        "warnings": document.warnings,
                        "document": document.raw_document,
                    },
                )
                old_hash = existing.content_hash if existing else None
                self.repository.replace_document(
                    relative_path, content_hash, stat.st_size, stat.st_mtime_ns,
                    document.format, document.parser_version, chunks,
                )
                if old_hash and old_hash != content_hash and not self.repository.is_hash_referenced(old_hash):
                    self._artifact_path(old_hash).unlink(missing_ok=True)
                self.repository.finish_job(job_id)
                LOGGER.info("Indexed %s (%d chunks)", relative_path, len(chunks))
                return IngestionResult(relative_path, FileState.READY, action, len(chunks))
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.retry_attempts:
                    LOGGER.warning("Attempt %d failed for %s: %s", attempt, relative_path, exc)
                    time.sleep(self.settings.retry_base_seconds * (2 ** (attempt - 1)))

        error = str(last_error) if last_error else "Unknown extraction failure"
        self.repository.finish_job(job_id, error)
        self.repository.mark_failed(relative_path, stat.st_size, stat.st_mtime_ns, suffix.lstrip("."), error)
        LOGGER.error("Failed to index %s: %s", relative_path, error)
        record = self.repository.get_file(relative_path)
        state = record.state if record else FileState.FAILED
        return IngestionResult(relative_path, state, "failed", record.chunk_count if record else 0, error)

    async def watch(self) -> None:
        self.reconcile()
        LOGGER.info("Watching %s", self.settings.source_dir.resolve())
        async for _changes in awatch(
            self.settings.source_dir,
            debounce=500,
            step=100,
            recursive=True,
        ):
            try:
                await asyncio.to_thread(self.reconcile)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Watcher reconciliation failed; continuing")

    def _record_failure(self, relative_path: str, path: Path, suffix: str, exc: Exception, action: str) -> IngestionResult:
        try:
            stat = path.stat()
            size, modified_ns = stat.st_size, stat.st_mtime_ns
        except OSError:
            size, modified_ns = 0, 0
        error = str(exc) or exc.__class__.__name__
        job_id = self.repository.create_job(relative_path, action)
        self.repository.start_job(job_id, 1)
        self.repository.finish_job(job_id, error)
        self.repository.mark_failed(relative_path, size, modified_ns, suffix.lstrip("."), error)
        return IngestionResult(relative_path, FileState.FAILED, "failed", error=error)

    def _artifact_path(self, content_hash: str) -> Path:
        return self.settings.artifact_dir / f"{content_hash}.docling.json"

