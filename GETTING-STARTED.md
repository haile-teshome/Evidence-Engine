# Evidence Engine — Getting Started

A one-click research assistant. The launcher starts everything, opens the app in
its own window, and shuts it all down when you close that window.

## 1. Install the prerequisites (one time)

You need two free tools. Install both, then continue.

| Tool | Why | Get it |
|---|---|---|
| **Node.js (LTS)** | runs the launcher + the app UI | https://nodejs.org/en/download |
| **Python 3.10+** | runs the backend | https://www.python.org/downloads/ |

Optional:
- **Google Chrome / Edge / Brave** — used for the clean app window. Without it, the app opens in your default browser instead.
- **Ollama** (https://ollama.com) — only if you want local, offline AI models. Cloud models work without it.

> Tip: on the download pages, pick the installer for your operating system and accept the defaults. On Windows, keep "Add to PATH" checked.

## 2. Launch it

Put this whole folder wherever you like, then:

- **macOS** — double-click **`Evidence Engine.app`** (or **`launch.command`**).
- **Windows** — double-click **`launch.bat`**.

The **first** launch installs dependencies and builds an optimized version of
the app automatically (a few minutes, one time). Later launches take a few
seconds. The app is rebuilt only when its code changes, so day-to-day it runs
the fast production build, not the slower developer server.

**To quit:** just close the app window. Everything shuts down on its own.

## 3. macOS security prompt (first launch only)

Because the app isn't from the App Store, macOS may block the first open.
If you see "unidentified developer" or "cannot be opened":

1. **Right-click** `Evidence Engine.app` → **Open** → **Open** in the dialog.

That's it, only needed once. (If it still refuses after downloading as a zip,
open Terminal and run: `xattr -dr com.apple.quarantine "Evidence Engine.app"`.)

## Troubleshooting

- **"Node.js is required"** — install Node from the link above, then relaunch.
- **"Python 3 is required"** — install Python from the link above, then relaunch.
- **Nothing opens / stuck on "Waiting for services"** — close the window and
  relaunch; the first run may still be installing dependencies.
- **Local AI model** — handled automatically. On first launch the app installs
  Ollama (if needed), starts it, and downloads the default screening model
  (~4 GB) in the background. Local screening works once that finishes; you can use
  a cloud model in the sidebar in the meantime. If you're offline, the download
  is skipped and retried next launch.
  - Prefer no local download? Pick Claude / GPT / Gemini in the sidebar and add an
    API key in `Backend/.env` (copy from `Backend/.env.example`).
- **Ports busy** — if you already have something on ports 5180 or 8000, close it
  and relaunch.
- **"permission denied" / launcher won't run (macOS)** — the zip may have dropped
  the executable flag. In Terminal, `cd` into the folder and run:
  `chmod +x launch.command "Evidence Engine.app/Contents/MacOS/EvidenceEngine"`
