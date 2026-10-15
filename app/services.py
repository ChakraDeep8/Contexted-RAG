from __future__ import annotations

import threading

from .config import Settings
from .pipeline import IngestionPipeline
from .repository import SQLiteRepository
from .retrieval import HybridRetriever
from .vector_store import FastEmbedder, VectorStore


class ApplicationServices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.prepare()
        self.repository = SQLiteRepository(settings.database_path)
        self.ingestion = IngestionPipeline(settings, repository=self.repository)
        self._embedder = None
        self._vectors = None
        self._retriever = None
        self.lock = threading.Lock()

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._embedder = FastEmbedder(self.settings.embedding_model, self.settings.model_dir)
            self._vectors = VectorStore(self.settings.vector_dir, self._embedder.dimension)
            self._retriever = HybridRetriever(self.repository, self._vectors, self._embedder)
        return self._retriever

    def reconcile(self) -> tuple[int, int]:
        with self.lock:
            changed = len(self.ingestion.reconcile())
            has_pending = any(True for _ in self.repository.chunks_requiring_embeddings(1))
            embedded = self.retriever.index_pending() if has_pending or self.settings.vector_dir.exists() else 0
        return changed, embedded

    def close(self) -> None:
        if self._vectors is not None:
            self._vectors.close()

