# Setup Guide (No Technical Background Needed)

This walks through getting the project running from a completely fresh computer. Every command
below is meant to be copy-pasted exactly as written into a terminal.

No Docker. No Node.js. No database to install. Just Python.

---

## Step 0: Get the project onto the new laptop

**If the project is on GitHub** (recommended — keeps both laptops in sync):

```bash
git clone https://github.com/Kartikk1820/pdf-rag-citations.git
cd pdf-rag-citations
```

**If you're just copying the folder** (USB drive, AirDrop, Google Drive, etc.): copy the whole
`pdf-rag-citations` folder over, then open a terminal and `cd` into it, e.g.:

```bash
cd ~/Downloads/pdf-rag-citations
```

---

## Step 1: Install Python (if it isn't already)

Check if you already have it:

```bash
python3 --version
```

- If you see something like `Python 3.11.x` or `Python 3.12.x` — you're good, skip to Step 2.
- If you get "command not found" or a version below 3.11:
  - **Mac:** download and install from https://www.python.org/downloads/ (the macOS installer).
  - **Windows:** download and install from https://www.python.org/downloads/windows/.
    **Important:** on the first install screen, tick the box that says **"Add Python to PATH"**
    before clicking Install.

Close and reopen your terminal after installing, then re-run `python3 --version` to confirm.

---

## Step 2: Create a virtual environment and install dependencies

A "virtual environment" just keeps this project's Python packages separate from everything else
on your computer. Run these commands one at a time, from inside the `pdf-rag-citations` folder:

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This takes a minute or two the first time (it downloads the AI models used for search and
reranking too). You'll know it worked if there are no red error messages at the end.

> Every time you come back to work on this later, you need to re-run the `activate` line (the
> second command above) in any new terminal window before running the app. You do **not** need
> to repeat the `pip install` every time — just once.

---

## Step 3: Get a free AI API key

The app needs an AI provider to generate answers. **OpenRouter** is recommended — it's free and
takes under a minute:

1. Go to https://openrouter.ai/keys
2. Sign in (Google/GitHub sign-in works)
3. Click "Create Key", give it any name, click Create
4. Copy the key it shows you (starts with `sk-or-`)

Keep this tab open — you'll paste the key in Step 5, either into a file or directly into the app.

(Alternative: https://console.groq.com/keys also offers a free key if you'd rather use Groq.)

---

## Step 4: Start the app

Still in the same terminal (with the virtual environment active from Step 2):

```bash
cp .env.example .env
uvicorn web.app:app --reload
```

You should see a line like `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal window
open — closing it stops the app.

Open your browser to: **http://127.0.0.1:8000**

---

## Step 5: Add your API key (one-time)

The easiest way — do this directly in the app, no file editing:

1. In the browser, click **Settings** in the left sidebar.
2. Paste your API key from Step 3 into the "OpenRouter API Key" box.
3. Click **Save Settings**.

That's it — no restart needed.

*(If you'd rather edit the file instead: open the `.env` file in the project folder with any text
editor, find the line `OPENROUTER_API_KEY=your_openrouter_key_here`, and replace
`your_openrouter_key_here` with your actual key, then save the file.)*

---

## Step 6: Try it out

1. Click **Document Library** in the sidebar.
2. Click the **Upload PDF** button (or drag a PDF file onto the box) and pick a PDF from your
   computer.
3. Click **Ingest All** — this reads the PDF and builds the search index. Takes a few seconds.
4. Click **Chat** in the sidebar.
5. Type a question about what's in the PDF and press Enter (or click Ask).

You should see the answer stream in, with a source citation below it linking to the exact page.

---

## Stopping and restarting later

- **To stop:** go back to the terminal and press `Ctrl + C`.
- **To start again later** (even after restarting your computer):
  ```bash
  cd pdf-rag-citations          # wherever you put the folder
  source .venv/bin/activate     # Windows: .venv\Scripts\activate
  uvicorn web.app:app --reload
  ```
  Then open http://127.0.0.1:8000 again. Your uploaded PDFs, index, and settings are all saved —
  nothing needs to be redone.

> **Only run one copy of the app at a time.** The search index (Qdrant, embedded) is a local
> file that only one running process can open at once. If you start the app in a second terminal
> window (or forget one is already running from earlier and start another), the second one will
> fail with an error mentioning `storage/qdrant is already accessed by another instance`. If that
> happens: find and close the other running copy (check other terminal windows/tabs for one still
> printing `Uvicorn running on...`, or press `Ctrl+C` in it), then start the app again.

---

## Common problems

| Problem | Fix |
|---|---|
| `command not found: python3` | Python isn't installed, or wasn't added to PATH. Redo Step 1. |
| `command not found: uvicorn` | The virtual environment isn't active. Re-run the `activate` line from Step 2. |
| Browser says "can't connect" / page won't load | Check the terminal — is it still running and did it print "Uvicorn running on..."? If it crashed, scroll up in the terminal to read the error message. |
| `storage/qdrant is already accessed by another instance` | You have two copies of the app running at once (two terminal windows, or one you forgot about). Close the other one — see the note above — then restart. |
| "Missing API key" message when asking a question | Go to Settings in the app and make sure a real key is pasted in (not the placeholder text). |
| Port 8000 already in use | Run `uvicorn web.app:app --reload --port 8001` instead, and open `http://127.0.0.1:8001`. |
| Uploaded a PDF but Chat says nothing is indexed | Did you click **Ingest All** on the Document Library page after uploading? Upload and indexing are two separate steps. |
| A free AI model stopped working (404/429 error) | Free models on OpenRouter/Groq occasionally get retired. Go to Settings, change the model name to any other free one (e.g. pick one from https://openrouter.ai/models?max_price=0), and Save. |

---

For how the project actually works under the hood, see `README.md` and `ABOUT.md`.
