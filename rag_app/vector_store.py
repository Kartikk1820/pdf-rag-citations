"""Qdrant vector store: collection management, upsert, and search.

Runs in *embedded* mode (``QdrantClient(path=...)``) — a local on-disk
directory, with no server process to install, run, or containerize.
"""

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from rag_app.chunker import Chunk
from rag_app.config import get_settings


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    source: str
    page: int
    score: float


@lru_cache
def _client() -> QdrantClient:
    settings = get_settings()
    from pathlib import Path

    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def _source_filter(sources: list[str] | None) -> Filter | None:
    if not sources:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sources))])


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


def search(
    query_vector: list[float], *, top_k: int, sources: list[str] | None = None
) -> list[Hit]:
    settings = get_settings()
    results = _client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=_source_filter(sources),
        with_payload=True,
    ).points
    return [
        Hit(
            id=str(p.id),
            text=p.payload["text"],
            source=p.payload["source"],
            page=p.payload["page"],
            score=p.score,
        )
        for p in results
    ]


def count() -> int:
    settings = get_settings()
    if not _client().collection_exists(settings.qdrant_collection):
        return 0
    return _client().count(settings.qdrant_collection).count


def _scroll_sources() -> list[str]:
    settings = get_settings()
    if not _client().collection_exists(settings.qdrant_collection):
        return []
    sources: list[str] = []
    offset = None
    while True:
        points, offset = _client().scroll(
            collection_name=settings.qdrant_collection,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        sources.extend(p.payload["source"] for p in points)
        if offset is None:
            break
    return sources


def list_sources() -> list[str]:
    """Distinct document names currently indexed (for the document filter UI)."""
    return sorted(set(_scroll_sources()))


def count_by_source() -> dict[str, int]:
    """Indexed chunk count per document name (for the Document Library screen)."""
    counts: dict[str, int] = {}
    for source in _scroll_sources():
        counts[source] = counts.get(source, 0) + 1
    return counts
