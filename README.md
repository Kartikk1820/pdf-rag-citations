# PDF RAG with Citations

A Retrieval-Augmented Generation system that answers questions **only** from a
set of PDF documents and returns accurate, page-level citations with every
answer. If the answer isn't in the PDFs, it says so rather than guessing.

It ships as a full web application — Dashboard, Document Library, a streaming
Chat UI, an Analytics screen, and Settings — on top of a hybrid-search +
reranking retrieval pipeline. Everything runs locally with nothing but Python:
no Docker, no server to install, no Node build step.

> **New to this project?** `ABOUT.md` is a plain-English overview of what it does and why it's
> more than a basic "chat with your PDF" demo. `SETUP.md` is a copy-paste setup guide for anyone
> running it for the first time, on any computer.

## Screens

| Screen | What it does |
| --- | --- |
| **Dashboard** | Document/chunk counts, pipeline configuration, answer-rate and latency at a glance. |
| **Document Library** | Drag-and-drop PDF upload, one-click ingest, per-document chunk counts and status. |
| **Chat** | Streaming, multi-turn chat with a document filter, per-citation confidence bars, and direct links into the source PDF page. |
| **Analytics** | Per-query latency, retrieval confidence, and answer/no-match outcomes — observability, not just a chat box. |
| **Settings** | Switch LLM provider/model and tune retrieval (hybrid weight, rerank threshold, candidate pool size) without touching `.env` by hand. |

## Retrieval Pipeline

```
INGEST:  data/*.pdf --> pypdf (per page) --> chunker (page-aware, ~800 chars)
         --> fastembed (bge-small, 384-d) --> embedded Qdrant
         --> BM25 lexical index (rank_bm25, persisted alongside)

QUERY:   question --> LLM query rewrite (2-3 alternate phrasings)
         --> [dense search + BM25 search] per phrasing
         --> Reciprocal Rank Fusion (merge rankings, scale-free)
         --> cross-encoder rerank (ONNX, fastembed -- no torch)
         --> relevance-threshold check (reranker score, not raw cosine)
         --> LLM structured decision: answerable? which chunks?
         --> citations built deterministically from chunk metadata
         --> LLM streams the prose answer, token by token
```

**Why citations are reliable:** the LLM never emits page numbers. It only
decides whether the question is answerable and which retrieved snippets it
used; the app then attaches each citation from that snippet's stored
`{source, page}` metadata, so citations are correct by construction.

**Why hybrid + rerank, not just embeddings:** dense vectors are good at
paraphrase matches but routinely miss exact keywords (names, codes, numbers).
BM25 catches those. The two rankings are fused with Reciprocal Rank Fusion
(score-scale-free), then a cross-encoder reads the question and each
candidate *together* — a materially stronger relevance signal than cosine
similarity — and that score, not raw embedding similarity, is what decides
whether the corpus can answer the question at all.

**Why two LLM calls for an answer, not one:** the *decision* (answerable?
which chunks?) and the *narration* (the prose shown to the user) are
deliberately split. The CLI uses one call for both (cheap, fast, good enough
for a terminal). The web UI calls the decision first — so citations are known
deterministically before any text is shown — then streams a second,
unconstrained completion for the prose, which is what makes the chat feel
responsive instead of a multi-second blank wait.

## Libraries

| Concern | Choice |
| --- | --- |
| PDF parsing | `pypdf` |
| Embeddings | `fastembed` — **`BAAI/bge-small-en-v1.5`** (384-d, local/offline, free) |
| Lexical search | `rank_bm25` (pure Python, in-process) |
| Reranking | `fastembed` ONNX cross-encoder — **`Xenova/ms-marco-MiniLM-L-6-v2`** (local/offline, free, no torch) |
| Vector store | `qdrant-client` in **embedded mode** (local directory, no server) |
| LLM | OpenAI-compatible client → **OpenRouter** (free tier) or **Groq** (fallback) |
| Web app | `FastAPI` + `Jinja2` + hand-written CSS/JS (no Node, no CDN dependency) |
| CLI | `typer` |
| Analytics | `sqlite3` (stdlib) |
| Robustness | `tenacity` (retries), `json-repair` (JSON-mode recovery), `pydantic` (validation) |

## Assumptions

- PDFs are text-based (not scanned images); no OCR is performed.
- Citation granularity is the **page**.
- A single Qdrant collection holds all documents; re-running ingest rebuilds it.
- Embeddings and reranking run locally (no API cost, no rate limit, works offline).
- The web app is a local, single-user tool — chat history and analytics are
  process-local, not a multi-tenant session store.
- Qdrant's embedded mode locks its storage directory to one process at a
  time: don't run the CLI and the web app against the same `storage/` folder
  simultaneously.

## How to run

**Prerequisites:** Python 3.11+. Nothing else — no Docker, no Node.

```bash
# 1. Create a venv and install dependencies
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Configure an LLM provider
cp .env.example .env
# Edit .env: paste a free key from https://openrouter.ai/keys (or use Groq)
# -- or just start the app and set it from the Settings screen instead.

# 3. Run it
uvicorn web.app:app --reload
```

Open `http://127.0.0.1:8000`, upload PDFs on the Document Library screen,
click **Ingest All**, then ask questions on the Chat screen.

> OpenRouter's set of free models rotates over time. If the configured model
> returns a 404/429, pick any current `:free` model from
> https://openrouter.ai/models?max_price=0 and update it from the Settings
> screen, or switch to Groq.

### CLI (still available)

```bash
python -m rag_app.cli ingest
python -m rag_app.cli query "What is the leave policy?"
python -m rag_app.cli chat
```

Run the tests (offline, no network needed): `pytest -q`
