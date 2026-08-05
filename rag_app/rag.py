"""Query pipeline: retrieve -> grounded LLM answer -> deterministic citations.

Citations are the graded core of this task, so the LLM is never trusted to
produce page numbers. It only decides (a) whether the question is answerable
from the retrieved snippets and (b) which snippet indices it used. The app then
builds each citation from that snippet's stored metadata, so document name and
page number are correct by construction.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag_app import embeddings, vector_store
from rag_app.config import get_settings
from rag_app.prompts import SYSTEM_PROMPT, build_user_prompt

UNANSWERABLE_MESSAGE = (
    "The information is not available in the supplied documents."
)

# Cap how many sources we display, keeping citations precise and readable.
_MAX_CITATIONS = 3


class _LLMAnswer(BaseModel):
    is_answerable: bool
    answer: str = ""
    used_chunk_indices: list[int] = Field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    source: str
    page: int
    snippet: str


@dataclass(frozen=True)
class Answer:
    answered: bool
    text: str
    citations: list[Citation]


def _snippet(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# Matches leaked citation markers like "[0]", "[1, 2]" or " [1,4,6]" that some
# models append to the prose despite the prompt telling them not to.
_INDEX_MARKER = re.compile(r"\s*\[[\d\s,]+\]")


def _clean_answer(text: str) -> str:
    return _INDEX_MARKER.sub("", text).strip()


def answer_question(question: str) -> Answer:
    settings = get_settings()

    query_vector = embeddings.embed_query(question)
    hits = vector_store.search(query_vector, top_k=settings.top_k)

    # No retrieval, or nothing clears the relevance bar -> unanswerable.
    relevant = [h for h in hits if h.score >= settings.score_threshold]
    if not relevant:
        return Answer(answered=False, text=UNANSWERABLE_MESSAGE, citations=[])

    user_prompt = build_user_prompt(question, [h.text for h in relevant])
    result = call_structured_answer(user_prompt)

    if not result.is_answerable or not result.answer.strip():
        return Answer(answered=False, text=UNANSWERABLE_MESSAGE, citations=[])

    # Map the indices the model used back to concrete, de-duplicated citations.
    # We keep the model's own ordering (it tends to list the pages it actually
    # relied on first, which is a better answer-relevance signal than raw retrieval
    # score) and cap the count to stay precise and readable rather than dumping
    # every loosely-related page.
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for i in result.used_chunk_indices:
        if not (0 <= i < len(relevant)):
            continue
        hit = relevant[i]
        key = (hit.source, hit.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(source=hit.source, page=hit.page, snippet=_snippet(hit.text))
        )
        if len(citations) >= _MAX_CITATIONS:
            break

    # If the model answered but named no usable snippet, fall back to the top hit
    # so an answer is never presented without a citation.
    if not citations:
        top = relevant[0]
        citations.append(
            Citation(source=top.source, page=top.page, snippet=_snippet(top.text))
        )

    return Answer(answered=True, text=_clean_answer(result.answer), citations=citations)


def call_structured_answer(user_prompt: str) -> _LLMAnswer:
    # Imported lazily so unit tests can exercise retrieval logic without the
    # LLM/network layer being importable.
    from rag_app.llm import call_structured

    # temperature=0 for reproducible answers: what you test is what you demo.
    return call_structured(SYSTEM_PROMPT, user_prompt, _LLMAnswer, temperature=0.0)
