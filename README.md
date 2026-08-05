# PDF RAG with Citations

A small Retrieval-Augmented Generation app that answers questions **only** from a
set of PDF documents and returns an accurate citation (document name, page number,
and the retrieved snippet) with every answer. If the answer isn't in the PDFs, it
says so rather than guessing.

## Architecture

```
INGEST:  data/*.pdf --> pypdf (per page) --> chunker (page-aware, ~800 chars)
         --> fastembed (bge-small, 384-d) --> Qdrant collection

QUERY:   question --> fastembed --> Qdrant top-k search --> score threshold
         --> LLM (JSON mode: is_answerable? which chunks used?)
         --> citations built from the used chunks' stored metadata
```

**Why citations are reliable:** the LLM never emits page numbers. It only decides
whether the question is answerable and which retrieved snippets it used; the app
then attaches each citation from that snippet's stored `{source, page}` metadata,
so citations are correct by construction. If nothing clears the similarity
threshold, or the model reports the question isn't answerable, the app returns
*"The information is not available in the supplied documents."*

## Libraries

| Concern        | Choice |
| -------------- | ------ |
| PDF parsing    | `pypdf` |
| Embeddings     | `fastembed` — **`BAAI/bge-small-en-v1.5`** (384-d, local/offline, free) |
| Vector DB      | `qdrant-client` + Qdrant (local Docker) |
| LLM            | OpenAI-compatible client → **OpenRouter** `meta-llama/llama-3.3-70b-instruct:free` (default) or **Groq** `llama-3.3-70b-versatile` (fallback) |
| CLI            | `typer` |
| Robustness     | `tenacity` (retries), `json-repair` (JSON-mode recovery), `pydantic` (validation) |

## Assumptions

- PDFs are text-based (not scanned images); no OCR is performed.
- Citation granularity is the **page** — sufficient for the required document/page
  references.
- A single Qdrant collection holds all documents; re-running `ingest` rebuilds it.
- Embeddings run locally because OpenRouter offers no reliable embeddings endpoint;
  this also keeps ingestion free and offline.

## How to run

**Prerequisites:** Python 3.11+, Docker.

```bash
# 1. Start Qdrant (local, free)
docker compose up -d

# 2. Install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure the LLM provider
cp .env.example .env
# Edit .env: set LLM_PROVIDER=openrouter and paste a free key from
# https://openrouter.ai/keys  (or LLM_PROVIDER=groq with a Groq key)

# 4. Put the supplied PDFs in ./data, then ingest
python -m rag_app.cli ingest

# 5. Ask questions
python -m rag_app.cli query "What is the leave policy?"
python -m rag_app.cli chat      # interactive loop
```

Example output:

```
Answer:
Employees are entitled to 24 annual leave days.

Sources:
  - employee_handbook.pdf  (Page 17)
    Retrieved Text: "All full-time employees receive 24 days of annual leave..."
```

Run the tests (offline, no network needed): `pytest -q`
