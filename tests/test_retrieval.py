from pathlib import Path

from app.models import ContextChunk, SourceLocation
from app.repository import SQLiteRepository
from app.retrieval import HybridRetriever


class FakeEmbedder:
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float("invoice" in text.lower()), 0.0] for text in texts]


class FakeVectors:
    def __init__(self):
        self.vectors = {}

    def upsert(self, chunks, vectors):
        self.vectors.update({chunk.id: vector for chunk, vector in zip(chunks, vectors, strict=True)})

    def stored_chunk_ids(self):
        return set(self.vectors)

    def delete_chunk_ids(self, ids):
        for identifier in ids:
            self.vectors.pop(identifier, None)

    def search(self, vector, limit=20):
        return [(identifier, 0.9) for identifier in list(self.vectors)[:limit]]


def test_hybrid_index_and_exact_term_search(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "context.db")
    chunk = ContextChunk(
        id="invoice-chunk", file_hash="a" * 64, sequence=0,
        text="Invoice ABC-123 total is 425 dollars", token_count=9, kind="text",
        heading_path=(), location=SourceLocation(page=2),
    )
    repo.replace_document("invoice.pdf", "a" * 64, 100, 1, "pdf", "test", [chunk])
    retriever = HybridRetriever(repo, FakeVectors(), FakeEmbedder())
    assert retriever.index_pending() == 1
    assert list(repo.chunks_requiring_embeddings()) == []
    hits = retriever.search("ABC-123 invoice")
    assert hits[0].file.path == "invoice.pdf"
    assert hits[0].lexical is True
    assert hits[0].chunk.location.page == 2


def test_stale_vectors_are_removed(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "context.db")
    vectors = FakeVectors()
    vectors.vectors["gone"] = [1.0, 0.0, 0.0]
    HybridRetriever(repo, vectors, FakeEmbedder()).index_pending()
    assert vectors.vectors == {}

