"""FastAPI web app: Dashboard, Document Library, Chat, Analytics, Settings.

Server-rendered Jinja2 pages (no Node/build step) with small bits of vanilla
JS for the chat stream, upload, and settings form. State (chat history) is a
module-level list — this is a local, single-user demo app, not a multi-tenant
service, so a real session store would be solving a problem that doesn't
exist here.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_app import analytics, documents, ingest, rag, vector_store
from rag_app.config import EDITABLE_KEYS, get_settings, is_llm_configured

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="PDF RAG with Citations")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/files", StaticFiles(directory=str(documents.data_dir())), name="files")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory chat log: [{"question", "answer", "citations", "answered"}, ...]
_CHAT_LOG: list[dict] = []
_MAX_HISTORY_TURNS = 6


def _base_ctx(request: Request, active: str) -> dict:
    try:
        settings = get_settings()
        provider, model = settings.provider, settings.model
    except ValueError:
        provider, model = "not configured", ""
    return {"request": request, "active": active, "provider": provider, "model": model}


@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    ctx = _base_ctx(request, "dashboard")
    docs = documents.list_documents()
    settings = None
    settings_error = None
    try:
        settings = get_settings()
        if not is_llm_configured(settings):
            settings_error = (
                f"No API key set for provider '{settings.provider}'. "
                "Add one on the Settings page to enable Chat."
            )
    except ValueError as exc:
        settings_error = str(exc)

    stats = analytics.summary()
    ctx.update(
        {
            "doc_count": len(docs),
            "ingested_count": sum(1 for d in docs if d.ingested),
            "chunk_count": sum(d.chunks for d in docs),
            "settings": settings,
            "settings_error": settings_error,
            "stats": stats,
        }
    )
    return templates.TemplateResponse(request, "dashboard.html", ctx)


@app.get("/documents", response_class=HTMLResponse)
def documents_page(request: Request, flash: str | None = None, flash_kind: str = "success"):
    ctx = _base_ctx(request, "documents")
    ctx.update(
        {
            "docs": documents.list_documents(),
            "flash": flash,
            "flash_kind": flash_kind,
        }
    )
    return templates.TemplateResponse(request, "documents.html", ctx)


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile]):
    saved = 0
    for f in files:
        content = await f.read()
        try:
            documents.save_upload(f.filename, content)
            saved += 1
        except ValueError:
            continue
    flash = f"Uploaded {saved} file(s). Click \"Ingest All\" to index them."
    return RedirectResponse(url=f"/documents?flash={flash}", status_code=303)


@app.post("/documents/ingest")
def ingest_documents():
    try:
        result = ingest.ingest(documents.data_dir())
        flash = (
            f"Ingested {result.chunks} chunks from {result.pages} pages "
            f"across {result.documents} document(s)."
        )
        kind = "success"
    except FileNotFoundError as exc:
        flash, kind = str(exc), "error"
    return RedirectResponse(url=f"/documents?flash={flash}&flash_kind={kind}", status_code=303)


@app.post("/documents/delete")
def delete_document(filename: str = Form(...)):
    documents.delete_document(filename)
    flash = f"Deleted {filename}. Re-ingest to update the index."
    return RedirectResponse(url=f"/documents?flash={flash}", status_code=303)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    ctx = _base_ctx(request, "chat")
    ctx.update(
        {
            "sources": vector_store.list_sources(),
            "history": _CHAT_LOG,
        }
    )
    return templates.TemplateResponse(request, "chat.html", ctx)


@app.post("/chat/reset")
def chat_reset():
    _CHAT_LOG.clear()
    return {"ok": True}


def _ndjson(obj: dict) -> str:
    return json.dumps(obj) + "\n"


@app.post("/chat/ask")
async def chat_ask(request: Request):
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    sources = payload.get("sources") or None

    def event_stream():
        if not question:
            yield _ndjson({"kind": "done"})
            return

        timer = analytics.start_timer()
        history = [(t["question"], t["answer"]) for t in _CHAT_LOG[-_MAX_HISTORY_TURNS:]]
        tokens: list[str] = []
        citations_payload: list[dict] = []
        suggestions_payload: list[dict] = []
        answered = False
        mode = "grounded"

        try:
            for event in rag.stream_answer(question, sources=sources, history=history):
                if event.kind == "citations":
                    answered = event.answered
                    mode = event.mode
                    citations_payload = [asdict(c) for c in event.citations]
                    suggestions_payload = [asdict(s) for s in event.suggestions]
                    yield _ndjson(
                        {
                            "kind": "citations",
                            "citations": citations_payload,
                            "suggestions": suggestions_payload,
                            "answered": answered,
                            "mode": mode,
                        }
                    )
                elif event.kind == "token":
                    tokens.append(event.text)
                    yield _ndjson({"kind": "token", "text": event.text})
                elif event.kind == "done":
                    yield _ndjson({"kind": "done"})
        except Exception as exc:  # LLM/network failure mid-stream: surface it, don't just drop the connection
            yield _ndjson({"kind": "error", "text": f"Request failed: {exc}"})
            yield _ndjson({"kind": "done"})
            return

        full_text = "".join(tokens)
        _CHAT_LOG.append(
            {
                "question": question,
                "answer": full_text,
                "citations": citations_payload,
                "suggestions": suggestions_payload,
                "answered": answered,
                "mode": mode,
            }
        )
        if len(_CHAT_LOG) > 50:
            del _CHAT_LOG[: len(_CHAT_LOG) - 50]

        best_score = max((c["confidence"] for c in citations_payload), default=None)
        analytics.log_query(
            question=question,
            answered=answered,
            latency_ms=timer.elapsed_ms(),
            best_score=best_score,
            num_candidates=len(citations_payload),
            citations=citations_payload,
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request):
    ctx = _base_ctx(request, "analytics")
    ctx.update({"stats": analytics.summary(), "recent": analytics.recent(30), "now": time.time()})
    return templates.TemplateResponse(request, "analytics.html", ctx)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, flash: str | None = None):
    ctx = _base_ctx(request, "settings")
    values = {key: os.getenv(key, "") for key in EDITABLE_KEYS}
    try:
        current = get_settings()
    except ValueError:
        current = None
    ctx.update({"values": values, "current": current, "flash": flash})
    return templates.TemplateResponse(request, "settings.html", ctx)


_CHECKBOX_FLAGS = ("ENABLE_HYBRID", "ENABLE_QUERY_REWRITE", "ENABLE_GENERAL_FALLBACK")


@app.post("/settings")
async def save_settings(request: Request):
    form = await request.form()
    updates = {}
    for key in EDITABLE_KEYS:
        if key in form:
            value = form[key]
            if key.endswith("_API_KEY") and not value:
                continue  # don't blank out a key the user left untouched
            if key in _CHECKBOX_FLAGS:
                value = "true" if form.get(key) == "on" else "false"
            updates[key] = str(value)
    for flag in _CHECKBOX_FLAGS:
        if flag not in form:
            updates[flag] = "false"

    from rag_app.config import update_settings

    update_settings(updates)
    return RedirectResponse(url="/settings?flash=Settings saved.", status_code=303)
