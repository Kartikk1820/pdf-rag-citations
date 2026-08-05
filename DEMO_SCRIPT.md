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

2. **(0:15–1:15) Three real questions.** These are verified — each answers from a
   *different* PDF (docs 02, 04, 07), showing multi-document retrieval. For each,
   point the cursor at the **Answer** and the **Sources** (document name + page +
   snippet).
   ```
   python -m rag_app.cli query "Under which section of the Passports Act was Maneka Gandhi's passport impounded?"
   python -m rag_app.cli query "Which international convention was referenced in the Ashwani Kumar case?"
   python -m rag_app.cli query "Why was the road safety writ petition dismissed?"
   ```

3. **(1:15–1:40) The unanswerable case.** Ask something plainly not in the PDFs:
   ```
   python -m rag_app.cli query "What is the capital city of Australia?"
   ```
   Show it responds: *"The information is not available in the supplied documents."*
   with no fabricated citation.

4. **(optional, 1:40–2:00)** `pytest -q` to show the tests pass.

## Tip
Open one cited PDF to the referenced page to visually confirm the citation is real.
This directly demonstrates citation accuracy (30% of the grade).
