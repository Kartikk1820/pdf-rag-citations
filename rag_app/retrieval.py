"""Hybrid retrieval pipeline: multi-query expansion -> dense + BM25 fusion ->
cross-encoder rerank.

This is the core upgrade over naive top-k dense search:

1. (optional) the question is rewritten into 2-3 alternative phrasings by the
   LLM, so retrieval is not at the mercy of the user's exact wording.
2. Every query variant is searched both densely (embeddings, via Qdrant) and
   lexically (BM25, exact keyword matches embeddings tend to miss), and the
   ranked lists are merged with Reciprocal Rank Fusion (RRF) -- a
   score-scale-free way to combine heterogeneous rankers.
3. The fused candidate pool is reranked by a cross-encoder that reads the
   question and each candidate together, which is a materially stronger
   relevance signal than cosine similarity and is what the pipeline uses to
   decide whether the corpus actually answers the question at all.

Reranking is scored against every query variant (the original phrasing plus
each rewrite), taking the best score per candidate. MS-MARCO-style
cross-encoders are trained on terse, search-box-style queries, so a
conversational wrapper ("can you tell me in brief what's in...") can score a
genuinely on-topic passage far lower than a terser rephrasing of the same
question would -- occasionally dropping a correct match below
SCORE_THRESHOLD entirely. Reusing the rewrites already generated for
retrieval to also score reranking (instead of only the verbatim original
question) fixes that without any extra LLM calls.
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field

from rag_app import embeddings, lexical_store, reranker, vector_store
from rag_app.config import get_settings
from rag_app.llm import call_structured
from rag_app.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt

_RRF_K = 60


class _RewriteResult(BaseModel):
    queries: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    source: str
    page: int
    retrieval_score: float  # best raw score seen pre-rerank (for display)
    rerank_score: float = 0.0  # cross-encoder relevance, 0..1


def rewrite_queries(
    question: str, history: list[tuple[str, str]] | None = None
) -> list[str]:
    try:
        result = call_structured(
            REWRITE_SYSTEM_PROMPT,
            build_rewrite_prompt(question, history),
            _RewriteResult,
            temperature=0.4,
        )
    except Exception:
        return []
    return [q.strip() for q in result.queries if q.strip()][:3]


def _rrf_fuse(ranked_lists: list[list[str]], weights: list[float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_ids, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (_RRF_K + rank)
    return scores


def _max_rerank_scores(queries: list[str], texts: list[str]) -> list[float]:
    """Best cross-encoder score per text, across every query variant."""
    best = [0.0] * len(texts)
    for q in queries:
        for r in reranker.rerank(q, texts):
            if r.score > best[r.index]:
                best[r.index] = r.score
    return best


def candidates(
    question: str,
    *,
    sources: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> list[Candidate]:
    settings = get_settings()
    queries = [question]
    if settings.enable_query_rewrite:
        queries += rewrite_queries(question, history)

    records: dict[str, dict] = {}
    ranked_lists: list[list[str]] = []
    weights: list[float] = []

    for q in queries:
        dense_hits = vector_store.search(
            embeddings.embed_query(q), top_k=settings.retrieval_k, sources=sources
        )
        for h in dense_hits:
            records[h.id] = {"text": h.text, "source": h.source, "page": h.page}
        ranked_lists.append([h.id for h in dense_hits])
        weights.append(1.0 - settings.hybrid_bm25_weight)

        if settings.enable_hybrid:
            lexical_hits = lexical_store.search(
                q, top_k=settings.retrieval_k, sources=sources
            )
            for h in lexical_hits:
                records.setdefault(
                    h["id"],
                    {"text": h["text"], "source": h["source"], "page": h["page"]},
                )
            ranked_lists.append([h["id"] for h in lexical_hits])
            weights.append(settings.hybrid_bm25_weight)

    if not records:
        return []

    fused_scores = _rrf_fuse(ranked_lists, weights)
    top_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    top_ids = top_ids[: settings.retrieval_k]

    pool = [
        Candidate(
            id=i,
            text=records[i]["text"],
            source=records[i]["source"],
            page=records[i]["page"],
            retrieval_score=fused_scores[i],
        )
        for i in top_ids
    ]

    scores = _max_rerank_scores(queries, [c.text for c in pool])
    reranked_pool = [
        Candidate(
            id=c.id,
            text=c.text,
            source=c.source,
            page=c.page,
            retrieval_score=c.retrieval_score,
            rerank_score=scores[i],
        )
        for i, c in enumerate(pool)
    ]
    reranked_pool.sort(key=lambda c: c.rerank_score, reverse=True)
    return reranked_pool[: settings.top_k]
