"""Offline unit tests: no network, no Qdrant, no LLM required.

We monkeypatch the retrieval and LLM boundaries so the tests focus on the parts
that decide correctness: chunk/page metadata preservation, deterministic
citation assembly, the unanswerable/refusal path, and the general-knowledge
fallback path.
"""

from types import SimpleNamespace

from rag_app import rag
from rag_app.chunker import chunk_pages
from rag_app.pdf_loader import Page
from rag_app.rag import UNANSWERABLE_MESSAGE, _FallbackAnswer, _LLMAnswer
from rag_app.retrieval import Candidate


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


def _candidate(text, source, page, score) -> Candidate:
    return Candidate(
        id=f"{source}:{page}:{text[:8]}",
        text=text,
        source=source,
        page=page,
        retrieval_score=score,
        rerank_score=score,
    )


def _patch_retrieval(monkeypatch, cands, llm_answer):
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: cands)
    monkeypatch.setattr(rag, "call_structured_answer", lambda prompt: llm_answer)


def _fake_settings(**overrides):
    base = {"score_threshold": 0.3, "enable_general_fallback": True}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_answer_builds_citation_from_metadata(monkeypatch):
    cands = [
        _candidate("Employees receive 24 days of annual leave.", "hb.pdf", 17, 0.9),
        _candidate("Unrelated text about parking.", "hb.pdf", 3, 0.5),
    ]
    llm = _LLMAnswer(is_answerable=True, answer="24 annual leave days.", used_chunk_indices=[0])
    _patch_retrieval(monkeypatch, cands, llm)

    result = rag.answer_question("What is the leave policy?")
    assert result.answered
    assert result.mode == "grounded"
    assert result.text == "24 annual leave days."
    assert len(result.citations) == 1
    # Page number comes from metadata, not from the model.
    assert result.citations[0].source == "hb.pdf"
    assert result.citations[0].page == 17
    assert result.citations[0].confidence == 0.9


def test_refused_when_model_says_unanswerable_and_fallback_disabled(monkeypatch):
    cands = [_candidate("Something.", "hb.pdf", 1, 0.9)]
    llm = _LLMAnswer(is_answerable=False, answer="", used_chunk_indices=[])
    _patch_retrieval(monkeypatch, cands, llm)
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings(enable_general_fallback=False))

    result = rag.answer_question("What is the CEO's blood type?")
    assert not result.answered
    assert result.mode == "refused"
    assert result.text == UNANSWERABLE_MESSAGE
    assert result.citations == []


def test_refused_when_nothing_clears_threshold_and_fallback_disabled(monkeypatch):
    # All candidates below SCORE_THRESHOLD -> the grounded LLM call is never made.
    cands = [_candidate("Weakly related.", "hb.pdf", 1, 0.05)]

    def _fail(prompt):
        raise AssertionError("Grounded LLM call should not be made when no hit is relevant")

    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: cands)
    monkeypatch.setattr(rag, "call_structured_answer", _fail)
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings(enable_general_fallback=False))

    result = rag.answer_question("Totally unrelated question?")
    assert not result.answered
    assert result.mode == "refused"
    assert result.text == UNANSWERABLE_MESSAGE


def test_chitchat_gets_a_conversational_reply_not_a_refusal(monkeypatch):
    # Nothing in the corpus matches "hi" -- with the fallback enabled (the
    # default), that should read as a normal reply, not a cold refusal.
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(
        rag.llm,
        "call_structured",
        lambda *a, **kw: _FallbackAnswer(intent="chitchat", answer="Hi! Ask me about your documents."),
    )

    result = rag.answer_question("hi")
    assert result.answered
    assert result.mode == "chitchat"
    assert result.citations == []
    assert result.text == "Hi! Ask me about your documents."


def test_general_knowledge_fallback_for_offcorpus_question(monkeypatch):
    # cands empty -> no document was even plausibly relevant, so this is a
    # genuine off-corpus question, not an "ambiguous, did you mean" case.
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(
        rag.llm,
        "call_structured",
        lambda *a, **kw: _FallbackAnswer(intent="general", answer="Paris is the capital of France."),
    )

    result = rag.answer_question("What is the capital of France?")
    assert result.answered
    assert result.mode == "general"
    assert result.citations == []
    assert result.text == "Paris is the capital of France."


def test_document_list_intent_answers_from_real_index_not_llm(monkeypatch):
    # The LLM must never be trusted to name what's indexed -- same philosophy
    # as citations. It should answer "document_list" but leave the actual
    # listing to the app.
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(rag.vector_store, "list_sources", lambda: ["handbook.pdf", "chapter2.pdf"])
    monkeypatch.setattr(
        rag.llm,
        "call_structured",
        lambda *a, **kw: _FallbackAnswer(intent="document_list", answer=""),
    )

    result = rag.answer_question("what documents do you have?")
    assert result.mode == "document_list"
    assert "handbook.pdf" in result.text
    assert "chapter2.pdf" in result.text


def test_ambiguous_question_suggests_documents_instead_of_guessing(monkeypatch):
    # Retrieval found candidates across two documents, but none cleared the
    # relevance bar -- this should offer the closest documents rather than
    # calling the general-knowledge LLM path at all.
    cands = [
        _candidate("Chapter 1 content.", "ch1.pdf", 1, 0.05),
        _candidate("Chapter 2 content.", "ch2.pdf", 1, 0.03),
    ]
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: cands)
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(
        rag.llm,
        "call_structured",
        lambda *a, **kw: _FallbackAnswer(intent="general", answer="should not be used"),
    )

    result = rag.answer_question("can you tell me in brief what's in chapter 1")
    assert result.mode == "ambiguous"
    assert result.citations == []
    assert [s.source for s in result.suggestions] == ["ch1.pdf", "ch2.pdf"]


def test_leaked_index_markers_are_stripped_from_answer(monkeypatch):
    cands = [_candidate("Torture convention text.", "c.pdf", 5, 0.9)]
    llm = _LLMAnswer(
        is_answerable=True,
        answer="It referenced the UN Convention against Torture [1, 4, 6].",
        used_chunk_indices=[0],
    )
    _patch_retrieval(monkeypatch, cands, llm)

    result = rag.answer_question("Which convention?")
    assert result.text == "It referenced the UN Convention against Torture."
    assert "[" not in result.text


def test_answered_but_no_indices_falls_back_to_top_hit(monkeypatch):
    cands = [_candidate("Top relevant snippet.", "a.pdf", 5, 0.8)]
    llm = _LLMAnswer(is_answerable=True, answer="Some answer.", used_chunk_indices=[])
    _patch_retrieval(monkeypatch, cands, llm)

    result = rag.answer_question("A question?")
    assert result.answered
    assert len(result.citations) == 1
    assert result.citations[0].page == 5


def test_stream_answer_emits_citations_before_tokens(monkeypatch):
    cands = [_candidate("Streaming works.", "s.pdf", 2, 0.95)]
    llm = _LLMAnswer(is_answerable=True, answer="It streams.", used_chunk_indices=[0])
    _patch_retrieval(monkeypatch, cands, llm)
    monkeypatch.setattr(rag.llm, "stream_chat", lambda *a, **kw: iter(["It ", "streams."]))

    events = list(rag.stream_answer("Does it stream?"))
    kinds = [e.kind for e in events]
    assert kinds[0] == "citations"
    assert kinds[-1] == "done"
    assert events[0].answered is True
    assert events[0].mode == "grounded"
    assert events[0].citations[0].page == 2
    assert "".join(e.text for e in events if e.kind == "token") == "It streams."


def test_stream_answer_general_mode_has_no_citations(monkeypatch):
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(rag, "_fallback_classify", lambda *a, **kw: "general")
    monkeypatch.setattr(rag.llm, "stream_chat", lambda *a, **kw: iter(["Paris."]))

    events = list(rag.stream_answer("What is the capital of France?"))
    assert events[0].kind == "citations"
    assert events[0].mode == "general"
    assert events[0].citations == []
    assert "".join(e.text for e in events if e.kind == "token") == "Paris."


def test_stream_answer_document_list_mode_skips_llm_narration(monkeypatch):
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(rag, "_fallback_classify", lambda *a, **kw: "document_list")
    monkeypatch.setattr(rag.vector_store, "list_sources", lambda: ["handbook.pdf"])

    def _fail(*a, **kw):
        raise AssertionError("document_list must not call the LLM for narration")

    monkeypatch.setattr(rag.llm, "stream_chat", _fail)

    events = list(rag.stream_answer("what documents do you have?"))
    assert events[0].mode == "document_list"
    full_text = "".join(e.text for e in events if e.kind == "token")
    assert "handbook.pdf" in full_text


def test_stream_answer_ambiguous_mode_suggests_documents(monkeypatch):
    cands = [
        _candidate("Chapter 1 content.", "ch1.pdf", 1, 0.05),
        _candidate("Chapter 2 content.", "ch2.pdf", 1, 0.03),
    ]
    monkeypatch.setattr(rag.retrieval, "candidates", lambda *a, **kw: cands)
    monkeypatch.setattr(rag, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(rag, "_fallback_classify", lambda *a, **kw: "general")

    def _fail(*a, **kw):
        raise AssertionError("ambiguous mode must not call the LLM for narration")

    monkeypatch.setattr(rag.llm, "stream_chat", _fail)

    events = list(rag.stream_answer("what's in chapter 1"))
    assert events[0].mode == "ambiguous"
    assert [s.source for s in events[0].suggestions] == ["ch1.pdf", "ch2.pdf"]
