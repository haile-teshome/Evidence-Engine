# Evidence Engine — Systematic Review Platform

An LLM-powered systematic-review screening platform. Abstract / full-text screening, PICO refinement, multi-database search, citation snowballing, table + text extraction, and PRISMA reporting — all driven by a local FastAPI backend + a React/Vite frontend.

The default screening model is **LEADS-mistral-7b**, run locally via Ollama with the LEADS-native PICO-element prompt and a +0.20 score threshold. This was selected based on a separate benchmark (recall=1.000, specificity=0.676, MCC=+0.260, WSS@95=0.61 on van_Dis_2020). The benchmark workspace lives outside this repo at `~/Desktop/screening-benchmark/`.

## Components

| Path | What it is |
|---|---|
| `Backend/` | FastAPI HTTP layer. Routes screening to the LEADS-native pipeline when the model is `"leads"`, or to a generic prompt for cloud LLMs (Claude / GPT / Gemini). Serves SSE-streamed progress + server-side cancel. |
| `src/`, `index.html`, `vite.config.ts`, `package.json` | React + Vite frontend (shadcn UI). Talks to the Backend over REST/SSE. |
| `supabase/`, `utils/` | Optional Supabase edge functions + client. Used for auth + session storage when configured; the app runs fine without it. |

## Quick start

```bash
./setup.sh                  # installs Backend (venv) + frontend (pnpm)
./setup.sh --pull-ollama    # additionally pull the default LEADS-mistral GGUF (~4 GB)
```

Then in two terminals:

```bash
# Backend (FastAPI on :8000)
cd Backend && source .venv/bin/activate && bash run_api.sh

# Frontend (Vite on :5173) — in a second terminal
pnpm dev
```

Open <http://localhost:5173>.

### Setup flags

```bash
./setup.sh --help            # show all flags
./setup.sh --backend-only    # just the FastAPI server
./setup.sh --frontend-only   # just the React app
./setup.sh --pull-ollama     # also pull the LEADS-mistral Ollama model
```

## Prerequisites

- **Python ≥ 3.10**
- **Node.js ≥ 18** with **pnpm** (the setup script will `npm i -g pnpm` if missing)
- **Ollama** — required for the default screening model. Install from <https://ollama.com>, then `./setup.sh --pull-ollama` (or pull `hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest` manually).
- **(Optional) API keys** — only needed if you switch off the default LEADS model. Populate `Backend/.env` (copied from `Backend/.env.example`) with any of:
  - `ANTHROPIC_API_KEY` (Claude)
  - `OPENAI_API_KEY` (GPT)
  - `GEMINI_API_KEY` (Gemini)
- **(Optional) Supabase** — only needed for cross-device session sync + auth. Create `.env.local` at the project root with `VITE_SUPABASE_PROJECT_ID` and `VITE_SUPABASE_ANON_KEY`. The app renders and screens fine without these.

## Repo layout

```
.
├── setup.sh                    # one-shot installer
├── README.md                   # this file
├── ATTRIBUTIONS.md
├── .gitignore
├── package.json, pnpm-lock.yaml, vite.config.ts, tsconfig.json, …
├── index.html, default_shadcn_theme.css, postcss.config.mjs
├── src/                        # React app
│   ├── app/
│   │   ├── App.tsx, components/, pages/, lib/
│   └── vite-env.d.ts
├── Backend/                    # FastAPI server
│   ├── api.py                  # HTTP layer + LEADS routing
│   ├── leads_screening.py      # LEADS-native pipeline (prompt + scoring)
│   ├── app.py, utils.py, data_services.py, …
│   ├── requirements.txt, run_api.sh
│   └── .env.example
├── supabase/                   # Supabase edge functions (optional)
└── utils/                      # frontend Supabase client (optional)
```
