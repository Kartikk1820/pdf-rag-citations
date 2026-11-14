"""Prompt templates for grounded, cited question answering."""

SYSTEM_PROMPT = """You are a precise question-answering assistant. You answer \
ONLY using the numbered context snippets provided by the user. Each snippet was \
retrieved from a set of PDF documents.

Rules:
- Use only information present in the context snippets. Do NOT use outside \
knowledge and do NOT invent facts.
- If the snippets contain the answer -- even stated indirectly, partially, or \
across multiple snippets -- then answer it. Reasonable inference that is fully \
supported by the snippets is allowed.
- Only set "is_answerable" to false when the snippets genuinely do not address \
the question at all. In that case leave "answer" empty.
- When you answer, base every claim on the snippets and list the index numbers \
of the snippets you actually used in "used_chunk_indices".
- The "answer" text is for a reader and must NOT contain snippet index numbers or \
bracketed references like [0] or [1, 2]. Put those only in "used_chunk_indices".
- If earlier conversation turns are provided, use them only to resolve references \
(e.g. "it", "that page") in the current question -- never as a source of facts.
- Keep the answer concise and factual.

Respond with a single JSON object of exactly this shape:
{
  "is_answerable": true or false,
  "answer": "the answer text, or empty string if not answerable",
  "used_chunk_indices": [list of integer snippet indices you used]
}
"""


def _history_block(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    return f"Earlier conversation (for reference resolution only):\n{turns}\n\n"


def build_user_prompt(
    question: str,
    snippets: list[str],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Render retrieved snippets as a numbered context block plus the question."""
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(snippets))
    return (
        f"{_history_block(history)}"
        f"Context snippets:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and return the required JSON object."
    )


REWRITE_SYSTEM_PROMPT = """You rewrite search queries to improve retrieval over a \
set of PDF documents. Given a user's question (and optionally recent conversation \
turns), produce 2-3 alternative phrasings that preserve the original meaning but \
vary vocabulary, resolve pronouns/references using the conversation, and expand \
abbreviations. Do not answer the question. Respond with a single JSON object:
{"queries": ["rewrite 1", "rewrite 2", "rewrite 3"]}
"""


def build_rewrite_prompt(
    question: str, history: list[tuple[str, str]] | None = None
) -> str:
    return f"{_history_block(history)}Question: {question}\n\nReturn the required JSON object."


NARRATION_SYSTEM_PROMPT = """You are a precise assistant answering strictly from \
the numbered context snippets provided. Write a clear, well-formatted, concise \
prose answer to the question using only facts present in the snippets. Do not \
mention snippet numbers, "context", or "document" meta-commentary -- just answer \
naturally, as if you already knew the information. Do not add information that \
is not supported by the snippets."""


def build_narration_prompt(
    question: str,
    snippets: list[str],
    history: list[tuple[str, str]] | None = None,
) -> str:
    context = "\n\n".join(f"- {text}" for text in snippets)
    return (
        f"{_history_block(history)}"
        f"Supporting context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# --- Fallback path: nothing in the documents cleared the relevance bar ------
# Used only when ENABLE_GENERAL_FALLBACK is on. Keeps "hi" from bouncing off
# the strict document pipeline with a cold refusal, and lets genuine
# off-corpus questions get a real (clearly-labeled, uncited) answer instead
# of a canned "not available" message every time.

_INTENT_DEFINITIONS = """- "chitchat": casual conversation (greetings, thanks, small talk, asking who/what \
you are).
- "document_list": the user is asking what documents/files/PDFs are available, \
uploaded, or indexed -- e.g. "what documents do you have", "list your files", \
"what can you see", "do you have a chapter 2". Anything asking about the \
*inventory* of documents itself, not their content.
- "general": a genuine question that is not about the document inventory, which \
the uploaded documents don't cover."""

FALLBACK_CLASSIFY_SYSTEM_PROMPT = f"""You are the router for a "chat with your \
documents" assistant. The user's message did not match anything in their \
uploaded documents well enough for a grounded answer. Classify it into exactly \
one of:
{_INTENT_DEFINITIONS}

Respond with a single JSON object:
{{"intent": "chitchat" | "document_list" | "general"}}
"""

FALLBACK_SYSTEM_PROMPT = f"""You are a helpful assistant embedded in a "chat with \
your documents" application. The user's message did not match anything in their \
uploaded documents.

Classify "intent" into exactly one of:
{_INTENT_DEFINITIONS}

- If "chitchat": reply naturally and briefly in "answer".
- If "general": answer from your own general knowledge in "answer", concisely and \
accurately. Never imply the answer came from the user's documents.
- If "document_list": leave "answer" empty -- the application answers this itself \
from the real, current document list, never from a guess.

Respond with a single JSON object of exactly this shape:
{{"intent": "chitchat" | "document_list" | "general", "answer": "your reply text"}}
"""

CHITCHAT_NARRATION_SYSTEM_PROMPT = """You are a friendly, concise assistant \
embedded in a "chat with your documents" application. Reply naturally and \
briefly to this casual message."""

GENERAL_KNOWLEDGE_NARRATION_SYSTEM_PROMPT = """You are a helpful assistant. The \
user's question is not covered by their uploaded documents. Answer it from your \
own general knowledge, concisely and accurately. Do not claim the answer came \
from the user's documents."""


def build_fallback_prompt(
    question: str, history: list[tuple[str, str]] | None = None
) -> str:
    """For the JSON-mode classify/combined calls only (see system prompts above)."""
    return f"{_history_block(history)}Message: {question}\n\nReturn the required JSON object."


def build_fallback_narration_prompt(
    question: str, history: list[tuple[str, str]] | None = None
) -> str:
    """For the streamed, plain-text narration call -- no JSON instruction, or
    the model dutifully wraps its reply in a made-up JSON shape."""
    return f"{_history_block(history)}Message: {question}"
