"""Cross-encoder reranking (ONNX, via fastembed — no torch).

Dense/BM25 retrieval scores rank candidates against the whole corpus in
isolation; a cross-encoder reads the question and each candidate *together*,
which is a much stronger relevance signal and is what the pipeline uses to
decide "is this actually answerable from the corpus", not raw embedding
similarity.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag_app.config import get_settings


@lru_cache
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=get_settings().reranker_model)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Ranked:
    index: int
    score: float  # sigmoid-normalized, 0..1


def rerank(query: str, documents: list[str]) -> list[Ranked]:
    """Returns ``documents`` indices sorted best-first with normalized scores."""
    if not documents:
        return []
    raw_scores = list(_model().rerank(query, documents))
    ranked = [
        Ranked(index=i, score=_sigmoid(s)) for i, s in enumerate(raw_scores)
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
