# Evidence Engine

**An AI-powered assistant for systematic literature reviews.** Search multiple
databases, screen abstracts and full texts, refine PICO questions, snowball
citations, extract tables and data, and generate PRISMA-compliant reports —
all from one desktop app that runs on your own computer.

Screening runs locally by default (LEADS-Mistral-7B via Ollama), so your data
never leaves your machine unless you choose a cloud model.

---

## Get started

### 1. Install the two prerequisites (one time)

You need **Node.js** and **Python 3**. Pick your OS:

**macOS** (with [Homebrew](https://brew.sh)):
```bash
brew install node python
```

**Windows** (with [winget](https://learn.microsoft.com/windows/package-manager/winget/), built into Windows 10/11):
```powershell
winget install OpenJS.NodeJS Python.Python.3.12
```

Or download the installers directly: [Node.js (LTS)](https://nodejs.org/en/download) · [Python 3](https://www.python.org/downloads/).

> Optional: [Google Chrome](https://www.google.com/chrome/) for a clean app
> window (otherwise it opens in your default browser). You do **not** need to
> install Ollama — the app sets up the local AI model automatically on first run.

### 2. Get the app

**Option A — Download (easiest):** grab the latest zip from the
[**Releases page**](https://github.com/haile-teshome/Evidence-Engine/releases),
and unzip it anywhere.

**Option B — Clone (recommended if you have Git):**
```bash
git clone https://github.com/haile-teshome/Evidence-Engine.git
```
Cloning avoids the macOS security prompt in step 4.

### 3. Launch it

- **macOS** — double-click **`Evidence Engine.app`** (or `launch.command`).
- **Windows** — double-click **`launch.bat`**.

The first launch installs dependencies and builds an optimized version of the
app (a few minutes, one time). After that it starts in seconds and runs the
fast production build. **To quit, just close the app window** — everything shuts
down on its own.

### 4. macOS security prompt (only if you downloaded the zip)

Because the app isn't from the App Store, macOS may block the first open.
**Right-click** `Evidence Engine.app` → **Open** → **Open**. Only needed once.
(If it still refuses: `xattr -dr com.apple.quarantine "Evidence Engine.app"`.)

📖 Full walkthrough and troubleshooting: **[GETTING-STARTED.md](GETTING-STARTED.md)**

---

## AI models

- **Local (default, private):** on first launch the app auto-installs
  [Ollama](https://ollama.com) and downloads **LEADS-Mistral-7B** (~4 GB) for
  screening. Nothing leaves your computer; no manual setup.
- **Cloud (optional):** pick Claude / GPT / Gemini in the sidebar. These need an
  API key in `Backend/.env` (copy from `Backend/.env.example`):
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`.
- **Sessions & reviewers:** saved automatically to a local database
  (`~/.evidence-engine/evidence.db`) by the backend. No account or cloud service.
  Add reviewer profiles from the top-right menu for local dual-review projects.

---

## What's inside

| Path | What it is |
|---|---|
| `Backend/` | FastAPI server (port 8000). Local LEADS screening pipeline + cloud-LLM routing, streamed progress, server-side cancel. |
| `src/`, `index.html`, `vite.config.ts` | React + Vite frontend (shadcn UI), served as a production build by the launcher. |
| `launch.mjs`, `launch.command`, `launch.bat` | Cross-platform launcher: starts backend + frontend, opens the app window, and shuts down on close. |
| `Backend/store.py` | Local SQLite persistence for sessions + multi-reviewer projects (replaces the former cloud backend). |

Screening defaults to **LEADS-Mistral-7B** (chosen from a separate benchmark:
recall 1.000, specificity 0.676, MCC +0.260, WSS@95 0.61 on van_Dis_2020).

---

## For developers

Run the dev server with hot reload instead of the packaged launcher:

```bash
npm install
npm run dev                       # Vite dev server (http://localhost:5173)

# in a second terminal — backend
cd Backend && python3 -m uvicorn api:app --reload --port 8000
```

Build the production bundle: `npm run build` (output in `dist/`).
Type-check: `npm run typecheck`.

---

## Privacy

Evidence Engine runs entirely on your computer. Database searches go directly to
the public APIs (PubMed, Europe PMC, etc.). With the default local model, paper
text is never sent to any third party. Sessions and reviewer data stay in a
local SQLite database on your machine. Cloud models are opt-in.
