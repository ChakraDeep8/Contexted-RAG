from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Protocol

from qdrant_client import QdrantClient, models

from .models import ContextChunk


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    dimension = 384

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]


class VectorStore:
    collection = "context_chunks"

    def __init__(self, path: Path, dimension: int = 384):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dimension = dimension
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"context-chunk:{chunk_id}"))

    def upsert(self, chunks: list[ContextChunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=self.point_id(chunk.id),
                vector=vector,
                payload={"chunk_id": chunk.id, "file_hash": chunk.file_hash},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 20) -> list[tuple[str, float]]:
        response = self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit, with_payload=True
        )
        return [
            (str(point.payload["chunk_id"]), float(point.score))
            for point in response.points
            if point.payload and point.payload.get("chunk_id")
        ]

    def delete_chunk_ids(self, chunk_ids: Iterable[str]) -> None:
        ids = [self.point_id(chunk_id) for chunk_id in chunk_ids]
        if ids:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )

    def stored_chunk_ids(self) -> set[str]:
        result: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            result.update(
                str(point.payload["chunk_id"])
                for point in points
                if point.payload and point.payload.get("chunk_id")
            )
            if offset is None:
                return result

    def close(self) -> None:
        self.client.close()

