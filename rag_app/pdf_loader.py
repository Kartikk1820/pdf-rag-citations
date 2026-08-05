"""Parse PDFs into per-page text, preserving 1-based page numbers.

Page numbers are the backbone of citation accuracy, so we keep each page as a
distinct unit and never let text bleed across page boundaries.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str  # PDF file name, e.g. "employee_handbook.pdf"
    page: int  # 1-based page number
    text: str


def _normalize(text: str) -> str:
    # Collapse runs of whitespace/newlines that PDF extraction tends to produce,
    # while keeping the text readable.
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Page]:
    """Extract non-empty pages from a single PDF."""
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _normalize(raw)
        if text:
            pages.append(Page(source=path.name, page=idx, text=text))
    return pages


def load_pdfs(data_dir: Path) -> list[Page]:
    """Load every ``*.pdf`` under ``data_dir`` (sorted for deterministic order)."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Place the supplied PDFs there."
        )
    pages: list[Page] = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path))
    return pages
