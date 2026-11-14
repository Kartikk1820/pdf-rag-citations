"""Lightweight SQLite query log: latency, retrieval quality, and outcomes per
question. Powers the Analytics screen -- the thing that turns "a RAG script"
into "a RAG product with observability".
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rag_app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    answered INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    best_score REAL,
    num_candidates INTEGER,
    provider TEXT,
    model TEXT,
    citations_json TEXT
);
"""


def _db_path() -> Path:
    return Path(get_settings().storage_dir) / "analytics.db"


@contextmanager
def _conn():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class Timer:
    start: float

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start) * 1000


def start_timer() -> Timer:
    return Timer(start=time.monotonic())


def log_query(
    *,
    question: str,
    answered: bool,
    latency_ms: float,
    best_score: float | None,
    num_candidates: int,
    citations: list[dict],
) -> None:
    settings = get_settings()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO query_log "
            "(ts, question, answered, latency_ms, best_score, num_candidates, "
            " provider, model, citations_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                question,
                int(answered),
                latency_ms,
                best_score,
                num_candidates,
                settings.provider,
                settings.model,
                json.dumps(citations),
            ),
        )


def recent(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, question, answered, latency_ms, best_score, num_candidates, "
            "provider, model, citations_json FROM query_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "ts": r[0],
            "question": r[1],
            "answered": bool(r[2]),
            "latency_ms": r[3],
            "best_score": r[4],
            "num_candidates": r[5],
            "provider": r[6],
            "model": r[7],
            "citations": json.loads(r[8]) if r[8] else [],
        }
        for r in rows
    ]


def summary() -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*), SUM(answered), AVG(latency_ms), AVG(best_score) "
            "FROM query_log"
        ).fetchone()
    total, answered, avg_latency, avg_score = row
    total = total or 0
    answered = answered or 0
    return {
        "total_queries": total,
        "answered_queries": answered,
        "answer_rate": (answered / total) if total else 0.0,
        "avg_latency_ms": avg_latency or 0.0,
        "avg_best_score": avg_score or 0.0,
    }
