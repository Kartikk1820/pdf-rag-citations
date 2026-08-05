"""Split page text into overlapping chunks that retain source + page metadata.

A small recursive character splitter: it prefers to break on paragraph, then
line, then sentence, then word boundaries before finally hard-cutting. Chunking
happens strictly *within* a page so every resulting chunk carries the exact page
number needed for citations.
"""

import uuid
from dataclasses import dataclass

from rag_app.pdf_loader import Page

# Separators tried in order, from coarsest to finest.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer to break on the coarsest separator inside the window.
            window = text[start:end]
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > 0:
                    end = start + idx + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        # Advance with overlap, but always make forward progress past `start`.
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[Page], *, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in _split_text(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    source=page.source,
                    page=page.page,
                )
            )
    return chunks
