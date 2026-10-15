from __future__ import annotations

from dataclasses import dataclass

from .models import ContextChunk, FileRecord
from .repository import SQLiteRepository
from .vector_store import Embedder, VectorStore


@dataclass(slots=True)
class SearchHit:
    file: FileRecord
    chunk: ContextChunk
    score: float
    dense_score: float | None
    lexical: bool


class HybridRetriever:
    def __init__(self, repository: SQLiteRepository, vector_store: VectorStore, embedder: Embedder):
        self.repository = repository
        self.vector_store = vector_store
        self.embedder = embedder

    def index_pending(self, batch_size: int = 64) -> int:
        indexed = 0
        while chunks := list(self.repository.chunks_requiring_embeddings(batch_size)):
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            self.vector_store.upsert(chunks, vectors)
            self.repository.mark_chunks_embedded([chunk.id for chunk in chunks])
            indexed += len(chunks)
        active = {chunk.id for chunk in self.repository.list_active_chunks()}
        stale = self.vector_store.stored_chunk_ids() - active
        self.vector_store.delete_chunk_ids(stale)
        return indexed

    def search(self, query: str, limit: int = 6, candidates: int = 20) -> list[SearchHit]:
        dense = self.vector_store.search(self.embedder.embed([query])[0], candidates)
        lexical = self.repository.lexical_search(query, candidates)
        scores: dict[str, float] = {}
        dense_scores = dict(dense)
        lexical_ids = {chunk_id for chunk_id, _ in lexical}
        for ranking in (dense, lexical):
            for rank, (chunk_id, _raw) in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)
        hits: list[SearchHit] = []
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            resolved = self.repository.resolve_chunk(chunk_id)
            if resolved:
                file, chunk = resolved
                hits.append(SearchHit(file, chunk, score, dense_scores.get(chunk_id), chunk_id in lexical_ids))
            if len(hits) >= limit:
                break
        return hits

