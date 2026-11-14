"""Ingestion pipeline: PDFs -> pages -> chunks -> embeddings -> Qdrant."""

import logging
from dataclasses import dataclass
from pathlib import Path

from rag_app import embeddings, lexical_store, vector_store
from rag_app.chunker import chunk_pages
from rag_app.pdf_loader import load_pdfs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    documents: int
    pages: int
    chunks: int


def ingest(data_dir: Path) -> IngestResult:
    pages = load_pdfs(data_dir)
    documents = len({p.source for p in pages})
    logger.info("Loaded %d pages from %d documents", len(pages), documents)

    chunks = chunk_pages(pages)
    logger.info("Split into %d chunks; embedding...", len(chunks))

    vectors = embeddings.embed_documents([c.text for c in chunks])

    # Fresh collection each ingest keeps the index consistent with the PDFs on disk.
    vector_store.ensure_collection(recreate=True)
    vector_store.upsert_chunks(chunks, vectors)
    lexical_store.save_chunks(chunks)  # rebuilds the BM25 side of hybrid search

    return IngestResult(documents=documents, pages=len(pages), chunks=len(chunks))
