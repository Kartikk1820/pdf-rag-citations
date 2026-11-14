"""Lexical (BM25) index over the same chunks stored in Qdrant.

Dense embeddings are good at paraphrase/semantic matches but routinely miss
exact keyword hits (product codes, names, numbers). BM25 catches those, so the
retrieval pipeline fuses both rather than relying on vectors alone. Qdrant's
embedded mode has no full-text index, so chunks are additionally persisted
here (a small on-disk JSON file) to build an in-memory ``rank_bm25`` index.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_app.chunker import Chunk
from rag_app.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _store_path() -> Path:
    return Path(get_settings().storage_dir) / "chunks.jsonl"


def save_chunks(chunks: list[Chunk]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                    }
                )
                + "\n"
            )
    _load_index.cache_clear()


def _load_chunks() -> list[dict]:
    path = _store_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass(frozen=True)
class _Index:
    bm25: BM25Okapi | None
    records: list[dict]


@lru_cache
def _load_index() -> _Index:
    records = _load_chunks()
    if not records:
        return _Index(bm25=None, records=[])
    corpus = [_tokenize(r["text"]) for r in records]
    return _Index(bm25=BM25Okapi(corpus), records=records)


def search(query: str, *, top_k: int, sources: list[str] | None = None) -> list[dict]:
    """Returns dicts with id/text/source/page/score, highest score first."""
    index = _load_index()
    if index.bm25 is None:
        return []
    scores = index.bm25.get_scores(_tokenize(query))
    ranked = sorted(
        range(len(index.records)), key=lambda i: scores[i], reverse=True
    )
    allowed = set(sources) if sources else None
    results = []
    for i in ranked:
        record = index.records[i]
        if allowed is not None and record["source"] not in allowed:
            continue
        if scores[i] <= 0:
            continue
        results.append({**record, "score": float(scores[i])})
        if len(results) >= top_k:
            break
    return results


def reset() -> None:
    _load_index.cache_clear()
