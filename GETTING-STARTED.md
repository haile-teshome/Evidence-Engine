# Evidence Engine: Getting Started

A one-click research assistant. It starts everything, opens in its own window,
and shuts it all down when you close that window.

**You do not need to install Node.js, Python or Ollama.** A download from the
[Releases page](https://github.com/haile-teshome/Evidence-Engine/releases)
carries its own runtimes.

## 1. Download and install

| Your computer | Download | Then |
|---|---|---|
| **Mac** (Apple Silicon: M1 and later) | `...-darwin-arm64.dmg` | Open it, drag **Evidence Engine** onto **Applications** |
| **Mac** (Intel) | `...-darwin-x64.dmg` | Open it, drag **Evidence Engine** onto **Applications** |
| **Windows** | `...-win-x64-Setup.exe` | Run it, then open **Evidence Engine** from the Start Menu |
| **Linux** | `...-linux-x64.zip` | Unzip anywhere, run `launch.sh` |

Not sure which Mac you have? Apple menu, then About This Mac. Anything listing
an M1, M2, M3 or M4 chip is Apple Silicon.

## 2. Get past the first-open warning

These builds are not code-signed, so your computer warns you about an
unidentified developer. It is not a sign anything is wrong, and you only do this
once.

**macOS.** Control-click (or right-click) **Evidence Engine** in Applications and
choose **Open**, then **Open** again in the dialog.

If you are not offered an Open button, open **System Settings**, go to
**Privacy & Security**, scroll to the bottom, and click **Open Anyway** next to
Evidence Engine. You may be asked for your password.

If it still refuses, open Terminal and run:

```bash
xattr -dr com.apple.quarantine "/Applications/Evidence Engine.app"
```

**Windows.** On the blue SmartScreen box, click **More info**, then **Run
anyway**.

## 3. The first launch

It sets itself up, which takes a few minutes:

- builds its own private Python environment from files inside the app, with no
  internet needed for that part
- installs [Ollama](https://ollama.com) and downloads the local screening model,
  about 4 GB

The window opens by itself when the app is ready, and the model keeps downloading
in the background. You can use a cloud model in the sidebar in the meantime. If
you are offline, the model download is skipped and retried next launch.

Later launches take a few seconds.

**To quit:** close the app window. Everything shuts down on its own.

No account, no sign-up and no API keys. The default model runs on your own
machine, so your articles and decisions are not sent anywhere.

## 4. Where your work is kept, and updating

Everything you do is stored **outside** the app, in `.evidence-engine` in your
home folder (`%USERPROFILE%\.evidence-engine` on Windows): your reviews,
projects, screening decisions and cached article text.

That is what makes updating safe. The app tells you when a new version is out and
links to the download. Install it over the top of the old one:

- **macOS:** drag the new app into Applications and choose Replace.
- **Windows:** run the new installer.

Nothing is lost. To turn the update check off, set `EE_UPDATE_CHECK=0`.

To uninstall, delete the app. To remove your reviews too, also delete the
`.evidence-engine` folder in your home folder. (In Finder, press
Command-Shift-G and enter `~/.evidence-engine`.)

## Troubleshooting

- **Nothing opens, or it sits on "Waiting for services"** — the first run may
  still be building its Python environment. Give it a few minutes, then close
  and reopen. If it keeps happening, the log says why:
  `~/.evidence-engine/launcher.log` and `~/.evidence-engine/backend.log`.
- **"Another program is using port 8000"** — the app moves to a free port by
  itself, so this message means it found something answering on its own port
  that was not the app. Quit and reopen.
- **"The local engine stopped unexpectedly"** — the backend crashed.
  `~/.evidence-engine/backend.log` holds the reason.
- **Local AI model** — handled automatically on first launch. Prefer not to
  download it? Pick Claude, GPT or Gemini in the sidebar and add an API key
  through the app's API keys dialog, which stores it in your operating system's
  keychain.
- **"permission denied" (macOS, zip download only)** — a zip can drop the
  executable flag. In Terminal, `cd` into the folder and run:
  `chmod +x launch.command "Evidence Engine.app/Contents/MacOS/EvidenceEngine"`.

## Running from the source code

Only needed if you are developing the app. A checkout **does** need Node.js and
Python 3.10+ (`brew install node python`, or
`winget install OpenJS.NodeJS Python.Python.3.12`).

```bash
npm install
npm run dev                  # Vite dev server on http://localhost:5173
cd Backend && python3 -m uvicorn api:app --reload --port 8000
```

Or `./run.sh` to serve the built frontend and the API from one process. A
checkout keeps its environment in `Backend/.venv` and its keys in `Backend/.env`,
exactly as before.
