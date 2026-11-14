"""Filesystem bookkeeping for the Document Library screen: what PDFs exist on
disk under ``data_dir``, and whether each one has been ingested yet.
"""

from dataclasses import dataclass
from pathlib import Path

from rag_app import vector_store
from rag_app.config import get_settings


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    size_bytes: int
    chunks: int
    ingested: bool


def data_dir() -> Path:
    path = Path(get_settings().data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_documents() -> list[DocumentInfo]:
    chunk_counts = vector_store.count_by_source()
    infos = []
    for pdf_path in sorted(data_dir().glob("*.pdf")):
        chunks = chunk_counts.get(pdf_path.name, 0)
        infos.append(
            DocumentInfo(
                name=pdf_path.name,
                size_bytes=pdf_path.stat().st_size,
                chunks=chunks,
                ingested=chunks > 0,
            )
        )
    return infos


def save_upload(filename: str, content: bytes) -> Path:
    safe_name = Path(filename).name  # strip any path components
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    dest = data_dir() / safe_name
    dest.write_bytes(content)
    return dest


def delete_document(filename: str) -> None:
    safe_name = Path(filename).name
    target = data_dir() / safe_name
    if target.exists() and target.parent == data_dir():
        target.unlink()
