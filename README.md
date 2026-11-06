# Evidence Engine

**An AI-powered assistant for systematic literature reviews.** Search multiple
databases, screen abstracts and full texts, refine PICO questions, snowball
citations, extract tables and data, and generate PRISMA-compliant reports,
all from one desktop app that runs on your own computer.

Screening runs locally by default (LEADS-Mistral-7B via Ollama), so your data
never leaves your machine unless you choose a cloud model.

---

## Install

Download one file for your computer from the
[**Releases page**](https://github.com/haile-teshome/Evidence-Engine/releases).
You do **not** need to install Node.js, Python or Ollama: a download carries its
own runtimes.

| Your computer | Download | Then |
|---|---|---|
| **Mac** (Apple Silicon: M1 and later) | `...-darwin-arm64.dmg` | Open it, drag **Evidence Engine** to Applications |
| **Mac** (Intel) | `...-darwin-x64.dmg` | Open it, drag **Evidence Engine** to Applications |
| **Windows** | `...-win-x64-Setup.exe` | Run it, then open Evidence Engine from the Start Menu |
| **Linux** | `...-linux-x64.zip` | Unzip it, run `launch.sh` |

Not sure which Mac you have? Apple menu, then About This Mac. Anything listing an
M1, M2, M3 or M4 chip is Apple Silicon.

### The first open needs one extra step

These builds are not code-signed, so your computer will warn you about an
unidentified developer. It is one click, once:

- **macOS:** control-click (or right-click) **Evidence Engine** in Applications
  and choose **Open**, then **Open** again. If you are not offered an Open
  button, go to **System Settings → Privacy & Security**, scroll to the bottom
  and click **Open Anyway**.
- **Windows:** on the blue SmartScreen box, click **More info**, then **Run
  anyway**.

### The first launch takes a few minutes

It builds its own private Python environment from files inside the app, then
installs [Ollama](https://ollama.com) and downloads the local AI model (about
4 GB). The window opens by itself when it is ready, and after that it starts in
seconds. **To quit, close the app window** and everything shuts down on its own.

No account, no sign-up, no API keys.

### Updating

The app tells you when a new version is out and links to the download. Install
it over the top of the old one. Your reviews are kept in `~/.evidence-engine`
(`%USERPROFILE%\.evidence-engine` on Windows), **outside** the app, so replacing
it keeps everything. To turn the check off, set `EE_UPDATE_CHECK=0`.

📖 Full walkthrough and troubleshooting: **[GETTING-STARTED.md](GETTING-STARTED.md)**

---

## AI models

- **Local (default, private):** on first launch the app auto-installs
  [Ollama](https://ollama.com) and downloads **LEADS-Mistral-7B** (~4 GB) for
  screening. Nothing leaves your computer; no manual setup.
- **Cloud (optional):** pick Claude / GPT / Gemini in the sidebar. These need an
  API key, entered in the app's API keys dialog (stored in your OS keychain),
  or set in `~/.evidence-engine/.env` as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
  or `GEMINI_API_KEY`.
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

Working from a checkout needs **Node.js** and **Python 3** (`brew install node python`,
or `winget install OpenJS.NodeJS Python.Python.3.12`). A checkout keeps its venv in
`Backend/.venv` and its keys in `Backend/.env`, both unchanged.

Run the dev server with hot reload instead of the packaged launcher:

```bash
npm install
npm run dev                       # Vite dev server (http://localhost:5173)

# in a second terminal: backend
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
