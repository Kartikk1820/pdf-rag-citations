# Demo Video Script (1–2 min, no narration required)

Goal: show the app running locally, three answered questions with citations, and
one unanswerable question handled correctly.

## Before recording
- `docker compose up -d` (Qdrant running).
- `.env` configured, PDFs in `data/`.
- Terminal font large enough to read citations on screen.

## Shot list

1. **(0:00–0:15) Show it's local.** Terminal in the project folder. Run:
   ```
   docker compose ps
   python -m rag_app.cli ingest
   ```
   Let the "Ingested N chunks from M documents" line show.

2. **(0:15–1:15) Three real questions.** Use `chat` or three `query` calls.
   Pick questions whose answers are clearly in the PDFs. For each, point the
   cursor at the **Answer** and the **Sources** (document name + page + snippet).
   ```
   python -m rag_app.cli query "<question 1 from the PDFs>"
   python -m rag_app.cli query "<question 2 from the PDFs>"
   python -m rag_app.cli query "<question 3 from the PDFs>"
   ```

3. **(1:15–1:40) The unanswerable case.** Ask something plausibly related but not
   in any PDF:
   ```
   python -m rag_app.cli query "<something not in the documents>"
   ```
   Show it responds: *"The information is not available in the supplied documents."*
   with no fabricated citation.

4. **(optional, 1:40–2:00)** `pytest -q` to show the tests pass.

## Tip
Open one cited PDF to the referenced page to visually confirm the citation is real.
This directly demonstrates citation accuracy (30% of the grade).
