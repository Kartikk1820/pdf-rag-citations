"""Qdrant vector store: collection management, upsert, and search."""

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_app.chunker import Chunk
from rag_app.config import get_settings


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    page: int
    score: float


@lru_cache
def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(*, recreate: bool = False) -> None:
    settings = get_settings()
    client = _client()
    name = settings.qdrant_collection
    exists = client.collection_exists(name)
    if exists and not recreate:
        return
    if exists:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embedding_dim, distance=Distance.COSINE
        ),
    )


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    settings = get_settings()
    points = [
        PointStruct(
            id=chunk.id,
            vector=vector,
            payload={"text": chunk.text, "source": chunk.source, "page": chunk.page},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    _client().upsert(collection_name=settings.qdrant_collection, points=points)


def search(query_vector: list[float], *, top_k: int) -> list[Hit]:
    settings = get_settings()
    results = _client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        Hit(
            text=p.payload["text"],
            source=p.payload["source"],
            page=p.payload["page"],
            score=p.score,
        )
        for p in results
    ]


def count() -> int:
    settings = get_settings()
    return _client().count(collection_name=settings.qdrant_collection).count
