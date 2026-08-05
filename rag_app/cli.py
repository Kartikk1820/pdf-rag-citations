"""Command-line interface: ingest, query, and interactive chat.

Usage:
    python -m rag_app.cli ingest
    python -m rag_app.cli query "What is the leave policy?"
    python -m rag_app.cli chat
"""

import logging
from pathlib import Path

import typer

from rag_app.rag import Answer, answer_question

app = typer.Typer(add_completion=False, help="Ask questions over your PDFs, with citations.")

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Keep the terminal output clean: silence the per-request HTTP chatter from the
# HTTP/LLM/vector-store client libraries. Our own INFO logs (e.g. ingest progress)
# still show.
for _noisy in ("httpx", "httpcore", "openai", "qdrant_client"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _print_answer(result: Answer) -> None:
    typer.echo("\nAnswer:")
    typer.echo(result.text)
    if result.citations:
        typer.echo("\nSources:")
        for c in result.citations:
            typer.echo(f"  - {c.source}  (Page {c.page})")
            typer.echo(f'    Retrieved Text: "{c.snippet}"')
    typer.echo("")


@app.command()
def ingest(
    data_dir: Path = typer.Option(
        Path("data"), "--data-dir", "-d", help="Folder containing the PDF files."
    ),
) -> None:
    """Parse PDFs, embed them, and store the vectors in Qdrant."""
    from rag_app.ingest import ingest as run_ingest

    result = run_ingest(data_dir)
    typer.echo(
        f"Ingested {result.chunks} chunks from {result.pages} pages "
        f"across {result.documents} document(s)."
    )


@app.command()
def query(question: str = typer.Argument(..., help="The question to ask.")) -> None:
    """Answer a single question and print its citations."""
    _print_answer(answer_question(question))


@app.command()
def chat() -> None:
    """Interactive question/answer loop (Ctrl-C or 'exit' to quit)."""
    typer.echo("Ask a question about your documents. Type 'exit' to quit.")
    while True:
        try:
            question = typer.prompt("\nQuestion")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nBye.")
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        _print_answer(answer_question(question))


if __name__ == "__main__":
    app()
