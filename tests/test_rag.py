"""Offline unit tests: no network, no Qdrant, no LLM required.

We monkeypatch the retrieval and LLM boundaries so the tests focus on the parts
that decide correctness: chunk/page metadata preservation, deterministic
citation assembly, and the unanswerable path.
"""

from rag_app import rag
from rag_app.chunker import chunk_pages
from rag_app.pdf_loader import Page
from rag_app.rag import UNANSWERABLE_MESSAGE, _LLMAnswer
from rag_app.vector_store import Hit


def test_chunker_preserves_source_and_page():
    pages = [
        Page(source="doc.pdf", page=1, text="alpha " * 300),
        Page(source="doc.pdf", page=2, text="beta " * 300),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    assert len(chunks) > 2  # long pages produced multiple chunks
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "doc.pdf" for c in chunks)
    assert all(c.id for c in chunks)  # every chunk has an id


def test_chunker_no_infinite_loop_on_early_separator():
    # A separator right at the start once caused non-termination.
    text = "a. " + ("x" * 500)
    pages = [Page(source="d.pdf", page=1, text=text)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=30)
    assert len(chunks) >= 1
    # Terminates and covers all content (>= because overlap duplicates some chars).
    assert "".join(c.text for c in chunks).count("x") >= 500


def _patch_retrieval(monkeypatch, hits, llm_answer):
    monkeypatch.setattr(rag.embeddings, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(rag.vector_store, "search", lambda v, top_k: hits)
    monkeypatch.setattr(rag, "call_structured_answer", lambda prompt: llm_answer)


def test_answer_builds_citation_from_metadata(monkeypatch):
    hits = [
        Hit(text="Employees receive 24 days of annual leave.", source="hb.pdf", page=17, score=0.9),
        Hit(text="Unrelated text about parking.", source="hb.pdf", page=3, score=0.5),
    ]
    llm = _LLMAnswer(is_answerable=True, answer="24 annual leave days.", used_chunk_indices=[0])
    _patch_retrieval(monkeypatch, hits, llm)

    result = rag.answer_question("What is the leave policy?")
    assert result.answered
    assert result.text == "24 annual leave days."
    assert len(result.citations) == 1
    # Page number comes from metadata, not from the model.
    assert result.citations[0].source == "hb.pdf"
    assert result.citations[0].page == 17


def test_unanswerable_when_model_says_so(monkeypatch):
    hits = [Hit(text="Something.", source="hb.pdf", page=1, score=0.9)]
    llm = _LLMAnswer(is_answerable=False, answer="", used_chunk_indices=[])
    _patch_retrieval(monkeypatch, hits, llm)

    result = rag.answer_question("What is the CEO's blood type?")
    assert not result.answered
    assert result.text == UNANSWERABLE_MESSAGE
    assert result.citations == []


def test_unanswerable_when_nothing_clears_threshold(monkeypatch):
    # All hits below SCORE_THRESHOLD -> we don't even call the LLM.
    hits = [Hit(text="Weakly related.", source="hb.pdf", page=1, score=0.05)]

    def _fail(prompt):
        raise AssertionError("LLM should not be called when no hit is relevant")

    monkeypatch.setattr(rag.embeddings, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(rag.vector_store, "search", lambda v, top_k: hits)
    monkeypatch.setattr(rag, "call_structured_answer", _fail)

    result = rag.answer_question("Totally unrelated question?")
    assert not result.answered
    assert result.text == UNANSWERABLE_MESSAGE


def test_leaked_index_markers_are_stripped_from_answer(monkeypatch):
    hits = [Hit(text="Torture convention text.", source="c.pdf", page=5, score=0.9)]
    llm = _LLMAnswer(
        is_answerable=True,
        answer="It referenced the UN Convention against Torture [1, 4, 6].",
        used_chunk_indices=[0],
    )
    _patch_retrieval(monkeypatch, hits, llm)

    result = rag.answer_question("Which convention?")
    assert result.text == "It referenced the UN Convention against Torture."
    assert "[" not in result.text


def test_answered_but_no_indices_falls_back_to_top_hit(monkeypatch):
    hits = [Hit(text="Top relevant snippet.", source="a.pdf", page=5, score=0.8)]
    llm = _LLMAnswer(is_answerable=True, answer="Some answer.", used_chunk_indices=[])
    _patch_retrieval(monkeypatch, hits, llm)

    result = rag.answer_question("A question?")
    assert result.answered
    assert len(result.citations) == 1
    assert result.citations[0].page == 5
