from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .models import ContextChunk, FileRecord, FileState, JobState, SourceLocation


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT,
    size INTEGER NOT NULL DEFAULT 0,
    modified_ns INTEGER NOT NULL DEFAULT 0,
    format TEXT NOT NULL,
    state TEXT NOT NULL,
    parser_version TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    indexed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_state ON files(state);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    file_hash TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    kind TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    needs_embedding INTEGER NOT NULL DEFAULT 1,
    UNIQUE(file_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id, sequence);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(needs_embedding);

CREATE TABLE IF NOT EXISTS locations (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    page INTEGER,
    sheet TEXT,
    section TEXT,
    row_start INTEGER,
    row_end INTEGER,
    cell_range TEXT,
    bbox TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_path ON ingestion_jobs(path, id DESC);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            # A previous process may have exited while a job was marked active.
            connection.execute(
                "UPDATE ingestion_jobs SET state=?, finished_at=?, error=? WHERE state IN (?, ?)",
                (JobState.FAILED, _now(), "Interrupted by process shutdown", JobState.PENDING, JobState.PROCESSING),
            )
            connection.execute(
                "UPDATE files SET state=? WHERE state=? AND chunk_count=0",
                (FileState.PENDING, FileState.PROCESSING),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_job(self, path: str, action: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO ingestion_jobs(path,state,action,attempts) VALUES(?,?,?,0)",
                (path, JobState.PENDING, action),
            )
            return int(cursor.lastrowid)

    def start_job(self, job_id: int, attempt: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET state=?, attempts=?, started_at=? WHERE id=?",
                (JobState.PROCESSING, attempt, _now(), job_id),
            )

    def finish_job(self, job_id: int, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET state=?, finished_at=?, error=? WHERE id=?",
                (JobState.FAILED if error else JobState.SUCCEEDED, _now(), error, job_id),
            )

    def get_file(self, path: str) -> FileRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return self._file_record(row) if row else None

    def find_missing_by_hash(self, content_hash: str, present_paths: set[str]) -> FileRecord | None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM files WHERE content_hash=? ORDER BY updated_at DESC", (content_hash,)
            ).fetchall()
        return next((self._file_record(row) for row in rows if row["path"] not in present_paths), None)

    def mark_processing(self, path: str, size: int, modified_ns: int, fmt: str) -> None:
        now = _now()
        with self._connect() as connection:
            state = FileState.PROCESSING
            connection.execute(
                """INSERT INTO files(path,size,modified_ns,format,state,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_ns=excluded.modified_ns,
                     format=excluded.format, state=?, updated_at=excluded.updated_at""",
                (path, size, modified_ns, fmt, state, now, now, state),
            )

    def mark_failed(self, path: str, size: int, modified_ns: int, fmt: str, error: str) -> None:
        now = _now()
        with self._connect() as connection:
            existing = connection.execute("SELECT chunk_count FROM files WHERE path=?", (path,)).fetchone()
            state = FileState.READY if existing and existing["chunk_count"] else FileState.FAILED
            connection.execute(
                """INSERT INTO files(path,size,modified_ns,format,state,last_error,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET size=excluded.size, modified_ns=excluded.modified_ns,
                     format=excluded.format, state=?, last_error=excluded.last_error, updated_at=excluded.updated_at""",
                (path, size, modified_ns, fmt, state, error, now, now, state),
            )

    def replace_document(
        self,
        path: str,
        content_hash: str,
        size: int,
        modified_ns: int,
        fmt: str,
        parser_version: str,
        chunks: list[ContextChunk],
    ) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO files(path,content_hash,size,modified_ns,format,state,parser_version,
                                      chunk_count,indexed_at,created_at,updated_at,last_error)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)
                   ON CONFLICT(path) DO UPDATE SET content_hash=excluded.content_hash, size=excluded.size,
                     modified_ns=excluded.modified_ns, format=excluded.format, state=excluded.state,
                     parser_version=excluded.parser_version, chunk_count=excluded.chunk_count,
                     indexed_at=excluded.indexed_at, updated_at=excluded.updated_at, last_error=NULL""",
                (path, content_hash, size, modified_ns, fmt, FileState.READY, parser_version,
                 len(chunks), now, now, now),
            )
            file_id = connection.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]
            connection.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            for chunk in chunks:
                connection.execute(
                    """INSERT INTO chunks(id,file_id,file_hash,sequence,text,token_count,kind,
                                          heading_path,needs_embedding) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (chunk.id, file_id, chunk.file_hash, chunk.sequence, chunk.text, chunk.token_count,
                     chunk.kind, json.dumps(chunk.heading_path), int(chunk.needs_embedding)),
                )
                loc = chunk.location
                connection.execute(
                    """INSERT INTO locations(chunk_id,page,sheet,section,row_start,row_end,cell_range,bbox)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (chunk.id, loc.page, loc.sheet, loc.section, loc.row_start, loc.row_end,
                     loc.cell_range, json.dumps(loc.bbox) if loc.bbox else None),
                )

    def rename_file(self, old_path: str, new_path: str, size: int, modified_ns: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE files SET path=?, size=?, modified_ns=?, updated_at=? WHERE path=?",
                (new_path, size, modified_ns, _now(), old_path),
            )

    def delete_file(self, path: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT content_hash FROM files WHERE path=?", (path,)).fetchone()
            if not row:
                return None
            content_hash = row["content_hash"]
            connection.execute("DELETE FROM files WHERE path=?", (path,))
            remaining = connection.execute(
                "SELECT 1 FROM files WHERE content_hash=? LIMIT 1", (content_hash,)
            ).fetchone()
        return None if remaining else content_hash

    def is_hash_referenced(self, content_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM files WHERE content_hash=? LIMIT 1", (content_hash,)
            ).fetchone()
        return row is not None

    def list_files(self) -> list[FileRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM files ORDER BY path").fetchall()
        return [self._file_record(row) for row in rows]

    def list_active_chunks(self) -> list[ContextChunk]:
        return list(self._query_chunks("", ()))

    def chunks_for_file(self, path: str) -> list[ContextChunk]:
        return list(self._query_chunks("WHERE f.path=?", (path,)))

    def chunks_requiring_embeddings(self, limit: int | None = None) -> Iterator[ContextChunk]:
        suffix = "WHERE c.needs_embedding=1 ORDER BY f.path,c.sequence"
        parameters: tuple[object, ...] = ()
        if limit is not None:
            suffix += " LIMIT ?"
            parameters = (limit,)
        yield from self._query_chunks(suffix, parameters)

    def resolve_chunk(self, chunk_id: str) -> tuple[FileRecord, ContextChunk] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM files f JOIN chunks c ON c.file_id=f.id WHERE c.id=?", (chunk_id,)).fetchone()
        if not row:
            return None
        file = self.get_file(row["path"])
        chunks = list(self._query_chunks("WHERE c.id=?", (chunk_id,)))
        return (file, chunks[0]) if file and chunks else None

    def _query_chunks(self, suffix: str, parameters: tuple[object, ...]) -> Iterator[ContextChunk]:
        sql = """SELECT c.*,l.page,l.sheet,l.section,l.row_start,l.row_end,l.cell_range,l.bbox
                 FROM chunks c JOIN files f ON f.id=c.file_id
                 LEFT JOIN locations l ON l.chunk_id=c.id """ + suffix
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        for row in rows:
            yield ContextChunk(
                id=row["id"], file_hash=row["file_hash"], sequence=row["sequence"], text=row["text"],
                token_count=row["token_count"], kind=row["kind"],
                heading_path=tuple(json.loads(row["heading_path"])),
                location=SourceLocation(page=row["page"], sheet=row["sheet"], section=row["section"],
                    row_start=row["row_start"], row_end=row["row_end"], cell_range=row["cell_range"],
                    bbox=json.loads(row["bbox"]) if row["bbox"] else None),
                needs_embedding=bool(row["needs_embedding"]),
            )

    @staticmethod
    def _file_record(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            id=row["id"], path=row["path"], content_hash=row["content_hash"], size=row["size"],
            modified_ns=row["modified_ns"], format=row["format"], state=FileState(row["state"]),
            parser_version=row["parser_version"], chunk_count=row["chunk_count"],
            last_error=row["last_error"], indexed_at=row["indexed_at"],
        )

