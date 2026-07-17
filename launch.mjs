#!/usr/bin/env node
// Cross-platform Evidence Engine launcher (macOS + Windows + Linux).
// Starts the backend + frontend, opens the app in its own browser window, and
// shuts everything down when that window is closed. Run via launch.command
// (macOS) or launch.bat (Windows), or directly: `node launch.mjs`.

import { spawn, spawnSync } from "node:child_process";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const PROJECT = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = path.join(PROJECT, "Backend");
const FRONTEND_PORT = 5180;
const BACKEND_PORT = 8000;
const APP_URL = `http://localhost:${FRONTEND_PORT}/`;
// The window is opened at this URL: the `?new=1` flag tells the app a fresh
// launch happened so it starts on Home with a new session instead of restoring
// the last one (an in-app refresh has no flag and still restores). Health checks
// use the bare APP_URL.
const APP_LAUNCH_URL = `${APP_URL}?new=1`;
const IS_WIN = process.platform === "win32";
const OLLAMA_URL = "http://localhost:11434/";
const EE_CACHE = path.join(os.homedir(), ".evidence-engine");     // downloaded tools
const DEFAULT_MODEL = "hf.co/mradermacher/leads-mistral-7b-v1-GGUF";  // default local screener

const started = [];               // child processes we spawned (to clean up)
let heldWindow = null;            // the Chrome child — kept referenced so Node
                                  // (v22) doesn't GC and tear it down mid-run.
let cleaned = false;

function log(m) { process.stdout.write(m + "\n"); }

function httpOk(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => { res.resume(); resolve(res.statusCode > 0 && res.statusCode < 500); });
    req.on("error", () => resolve(false));
    req.setTimeout(2000, () => { req.destroy(); resolve(false); });
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitHealthy(urls, timeoutSec) {
  process.stdout.write("Waiting for services");
  for (let i = 0; i < timeoutSec; i++) {
    const oks = await Promise.all(urls.map(httpOk));
    if (oks.every(Boolean)) { log(" ready."); return true; }
    process.stdout.write(".");
    await sleep(1000);
  }
  log("");
  return false;
}

const VENV_DIR = path.join(BACKEND_DIR, ".venv");
const VENV_PY = IS_WIN
  ? path.join(VENV_DIR, "Scripts", "python.exe")
  : path.join(VENV_DIR, "bin", "python");

// Self-contained bundle layout (produced by packaging/build-*.sh). When present,
// these let the app run with NO system Node/Python install. Absent → the
// launcher falls back to system runtimes (developer / from-source mode).
const RUNTIME_DIR = path.join(PROJECT, "runtime");
const BUNDLED_PY = IS_WIN
  ? path.join(RUNTIME_DIR, "python", "python.exe")
  : path.join(RUNTIME_DIR, "python", "bin", "python3");
const WHEELS_DIR = path.join(BACKEND_DIR, "wheels");   // vendored deps for offline install

function pyVersionOk(cmd) {
  // Returns the (major, minor) if runnable and >= 3.9, else null.
  const r = spawnSync(cmd, ["-c", "import sys;print(sys.version_info[0],sys.version_info[1])"],
    { encoding: "utf8" });
  if (r.status !== 0 || !r.stdout) return null;
  const [maj, min] = r.stdout.trim().split(/\s+/).map(Number);
  return maj === 3 && min >= 9 ? { maj, min } : null;
}

// Find a base interpreter capable of creating a venv. Prefer newer versions;
// the plain `python3` on a Mac's Finder PATH is often an old/broken system one,
// so we try explicit versioned names first.
function findBasePython() {
  // Prefer the bundled interpreter so a self-contained install never depends on
  // (or is broken by) whatever Python happens to be on the machine.
  if (fs.existsSync(BUNDLED_PY) && pyVersionOk(BUNDLED_PY)) return BUNDLED_PY;
  const cands = IS_WIN
    ? ["python", "py", "python3", "python3.12", "python3.11"]
    : ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"];
  let best = null, bestVer = -1;
  for (const c of cands) {
    const v = pyVersionOk(c);
    if (!v) continue;
    const canVenv = spawnSync(c, ["-c", "import venv"], { stdio: "ignore" });
    if (canVenv.status !== 0) continue;
    const score = v.min;              // higher minor == newer 3.x
    if (score > bestVer) { best = c; bestVer = score; }
  }
  return best;
}

function venvHasDeps() {
  if (!fs.existsSync(VENV_PY)) return false;
  const dep = spawnSync(VENV_PY, ["-c", "import fastapi, uvicorn"], { stdio: "ignore" });
  return dep.status === 0;
}

// Create Backend/.env on first run so the user never has to. No API keys are
// required for the default local model; the only useful value is a real contact
// email for NCBI/PubMed + Unpaywall, which we auto-fill from git if available.
function ensureBackendEnv() {
  const envPath = path.join(BACKEND_DIR, ".env");
  const examplePath = path.join(BACKEND_DIR, ".env.example");
  if (fs.existsSync(envPath) || !fs.existsSync(examplePath)) return;
  try {
    let content = fs.readFileSync(examplePath, "utf8");
    const gitEmail = (spawnSync("git", ["config", "user.email"], { encoding: "utf8" }).stdout || "").trim();
    if (gitEmail && /^ENTREZ_EMAIL=/m.test(content)) {
      content = content.replace(/^ENTREZ_EMAIL=.*$/m, `ENTREZ_EMAIL=${gitEmail}`);
    }
    fs.writeFileSync(envPath, content);
    log("Created Backend/.env (no API keys needed — the default model runs locally).");
  } catch { /* ignore — the backend falls back to built-in defaults */ }
}

// Ensure a working, isolated backend interpreter. Returns the python path to run
// the backend with, or null if we couldn't build one.
function ensureBackendPython() {
  if (venvHasDeps()) return VENV_PY;

  const base = findBasePython();
  if (!base) return null;

  if (!fs.existsSync(VENV_PY)) {
    log("First run: creating an isolated Python environment for the backend...");
    const mk = spawnSync(base, ["-m", "venv", VENV_DIR], { stdio: "ignore" });
    if (mk.status !== 0 || !fs.existsSync(VENV_PY)) return null;
  }

  const req = path.join(BACKEND_DIR, "requirements.txt");
  if (fs.existsSync(req)) {
    log("Installing backend dependencies (one time, a couple of minutes)...");
    // Offline install from vendored wheels when the bundle ships them; otherwise
    // fall back to PyPI (developer / from-source mode).
    const offline = fs.existsSync(WHEELS_DIR);
    const pipArgs = ["-m", "pip", "install"];
    if (offline) pipArgs.push("--no-index", "--find-links", WHEELS_DIR);
    else spawnSync(VENV_PY, ["-m", "pip", "install", "--upgrade", "pip"], { stdio: "ignore" });
    pipArgs.push("-r", req);
    const inst = spawnSync(VENV_PY, pipArgs, { stdio: "inherit" });
    if (inst.status !== 0) return null;
  }
  return venvHasDeps() ? VENV_PY : null;
}

function findChrome() {
  const home = os.homedir();
  let cands = [];
  if (process.platform === "darwin") {
    cands = [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ];
  } else if (IS_WIN) {
    const pf = process.env["ProgramFiles"] || "C:\\Program Files";
    const pf86 = process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)";
    const la = process.env["LOCALAPPDATA"] || path.join(home, "AppData", "Local");
    cands = [
      path.join(pf, "Google\\Chrome\\Application\\chrome.exe"),
      path.join(pf86, "Google\\Chrome\\Application\\chrome.exe"),
      path.join(la, "Google\\Chrome\\Application\\chrome.exe"),
      path.join(pf86, "Microsoft\\Edge\\Application\\msedge.exe"),
      path.join(pf, "Microsoft\\Edge\\Application\\msedge.exe"),
      path.join(pf, "BraveSoftware\\Brave-Browser\\Application\\brave.exe"),
    ];
  } else {
    for (const c of ["google-chrome", "chromium", "chromium-browser", "brave-browser", "microsoft-edge"]) {
      const w = spawnSync("which", [c], { encoding: "utf8" });
      if (w.status === 0 && w.stdout.trim()) return w.stdout.trim();
    }
  }
  return cands.find((p) => fs.existsSync(p)) || null;
}

// Newest mtime under a directory (recursive), skipping node_modules/dist/.git.
function newestMtime(dir, skip = new Set(["node_modules", "dist", ".git"])) {
  let newest = 0;
  let entries = [];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return 0; }
  for (const e of entries) {
    if (skip.has(e.name)) continue;
    const full = path.join(dir, e.name);
    try {
      if (e.isDirectory()) newest = Math.max(newest, newestMtime(full, skip));
      else newest = Math.max(newest, fs.statSync(full).mtimeMs);
    } catch { /* unreadable; ignore */ }
  }
  return newest;
}

// Rebuild if there is no build yet, or if any source file is newer than it.
function needsBuild() {
  const built = path.join(PROJECT, "dist", "index.html");
  if (!fs.existsSync(built)) return true;
  const builtAt = fs.statSync(built).mtimeMs;
  const srcNewest = Math.max(
    newestMtime(path.join(PROJECT, "src")),
    ...["index.html", "vite.config.ts", "package.json", "tailwind.config.js"]
      .map((f) => { try { return fs.statSync(path.join(PROJECT, f)).mtimeMs; } catch { return 0; } }),
  );
  return srcNewest > builtAt;
}

function findOllama() {
  const home = os.homedir();
  const cands = IS_WIN
    ? [path.join(process.env["LOCALAPPDATA"] || path.join(home, "AppData", "Local"), "Programs", "Ollama", "ollama.exe"),
       path.join(EE_CACHE, "ollama.exe"), "ollama.exe", "ollama"]
    : ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama",
       path.join(EE_CACHE, "Ollama.app", "Contents", "Resources", "ollama"), "ollama"];
  for (const c of cands) {
    if (c.includes("/") || c.includes("\\")) {
      if (fs.existsSync(c)) return c;
    } else {
      const w = spawnSync(IS_WIN ? "where" : "which", [c], { encoding: "utf8" });
      if (w.status === 0 && w.stdout.trim()) return w.stdout.trim().split(/\r?\n/)[0];
    }
  }
  return null;
}

// Download + install Ollama automatically. Returns a path to its binary, or null.
function installOllama() {
  try { fs.mkdirSync(EE_CACHE, { recursive: true }); } catch { /* exists */ }

  if (process.platform === "darwin") {
    const brew = spawnSync("which", ["brew"], { encoding: "utf8" });
    if (brew.status === 0 && brew.stdout.trim()) {
      log("Installing Ollama via Homebrew...");
      spawnSync("brew", ["install", "ollama"], { stdio: "ignore" });
      const f = findOllama(); if (f) return f;
    }
    const app = path.join(EE_CACHE, "Ollama.app");
    const bin = path.join(app, "Contents", "Resources", "ollama");
    if (fs.existsSync(bin)) return bin;
    log("Downloading Ollama (~20 MB, one time)...");
    const zip = path.join(EE_CACHE, "ollama-darwin.zip");
    if (spawnSync("curl", ["-fsSL", "-o", zip, "https://ollama.com/download/Ollama-darwin.zip"], { stdio: "ignore" }).status !== 0) return null;
    spawnSync("ditto", ["-x", "-k", zip, EE_CACHE], { stdio: "ignore" });
    try { fs.rmSync(zip, { force: true }); } catch { /* ignore */ }
    spawnSync("xattr", ["-dr", "com.apple.quarantine", app], { stdio: "ignore" });   // let the binary run
    return fs.existsSync(bin) ? bin : null;
  }

  if (IS_WIN) {
    const setup = path.join(EE_CACHE, "OllamaSetup.exe");
    log("Downloading Ollama installer (one time)...");
    if (spawnSync("curl", ["-fsSL", "-o", setup, "https://ollama.com/download/OllamaSetup.exe"], { stdio: "ignore" }).status !== 0) return null;
    log("Installing Ollama...");
    spawnSync(setup, ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], { stdio: "ignore" });
    return findOllama();
  }

  log("Installing Ollama...");
  spawnSync("sh", ["-c", "curl -fsSL https://ollama.com/install.sh | sh"], { stdio: "ignore" });
  return findOllama();
}

// Pull the default screening model if it isn't present. Runs in the BACKGROUND
// (it's ~4 GB) so the app opens immediately; Ollama resumes the pull if quit.
async function ensureModel(ollamaBin) {
  try {
    const r = await fetch(`${OLLAMA_URL}api/tags`);
    const j = await r.json();
    if ((j.models || []).some((m) => (m.name || "").startsWith(DEFAULT_MODEL))) return;
  } catch { /* server not answering tags yet — pull anyway */ }
  log(`Downloading the default AI model in the background (~4 GB): ${DEFAULT_MODEL}`);
  log("The app is usable now; local screening works once the download finishes.");
  const p = spawn(ollamaBin, ["pull", DEFAULT_MODEL], { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN });
  started.push(p);
}

function openDefaultBrowser(url) {
  if (process.platform === "darwin") spawn("open", [url], { stdio: "ignore", detached: true }).unref();
  else if (IS_WIN) spawn("cmd", ["/c", "start", "", url], { stdio: "ignore", detached: true }).unref();
  else spawn("xdg-open", [url], { stdio: "ignore", detached: true }).unref();
}

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon",
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
  ".webp": "image/webp", ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
  ".map": "application/json", ".csv": "text/csv", ".wasm": "application/wasm", ".txt": "text/plain",
};

// Serve the prebuilt SPA from distDir and reverse-proxy /api/* to the backend.
// Runs in-process (dies with the launcher). Using this instead of `vite preview`
// means the packaged bundle needs no node_modules at runtime — only the built
// dist/. Streaming-safe so backend SSE endpoints work.
function serveFrontend(distDir, port) {
  const server = http.createServer((req, res) => {
    if (req.url.startsWith("/api")) {
      const up = http.request(
        { host: "localhost", port: BACKEND_PORT, path: req.url, method: req.method, headers: req.headers },
        (r) => { res.writeHead(r.statusCode || 502, r.headers); r.pipe(res); },
      );
      up.on("error", () => { if (!res.headersSent) res.writeHead(502); res.end("backend not ready"); });
      req.pipe(up);
      return;
    }
    let rel = decodeURIComponent((req.url.split("?")[0] || "/"));
    if (rel.endsWith("/")) rel += "index.html";
    let file = path.join(distDir, rel);
    if (!file.startsWith(distDir)) { res.writeHead(403); res.end(); return; }   // no traversal
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) file = path.join(distDir, "index.html");   // SPA fallback
      res.writeHead(200, { "Content-Type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream" });
      fs.createReadStream(file).on("error", () => { if (!res.headersSent) res.writeHead(500); res.end(); }).pipe(res);
    });
  });
  server.on("error", (e) => log("Frontend server error: " + e.message));
  server.listen(port, "localhost");
  return server;
}

function killChild(child) {
  if (!child || child.killed) return;
  try {
    if (IS_WIN) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    else process.kill(-child.pid, "SIGTERM");   // whole process group (detached)
  } catch { /* already gone */ }
}

const APP_PROFILE = path.join(os.homedir(), ".evidence-engine-appwin");
const DEBUG_PORT = 9765;   // Chrome DevTools port for the app window (see monitorWindow)

// Is any Chrome process still using our dedicated app profile? Chrome's command
// line carries `--user-data-dir=...evidence-engine-appwin`, and so do all its
// helper/renderer processes, so this is true exactly while the window is open.
function appWindowAlive() {
  const marker = path.basename(APP_PROFILE);
  try {
    if (IS_WIN) {
      const r = spawnSync("wmic", ["process", "get", "CommandLine"], { encoding: "utf8" });
      return (r.stdout || "").includes(marker);
    }
    const r = spawnSync("pgrep", ["-f", marker], { encoding: "utf8" });
    return r.status === 0 && (r.stdout || "").trim() !== "";
  } catch { return false; }
}

// A previous unclean exit can leave Chrome's Singleton* lock files behind, which
// makes the next launch fail to open a window (it silently hands off instead).
// Clear them only when no Chrome process is actually using the profile.
function clearStaleProfileLock() {
  if (appWindowAlive()) return;
  for (const f of ["SingletonLock", "SingletonSocket", "SingletonCookie"]) {
    try { fs.rmSync(path.join(APP_PROFILE, f), { force: true }); } catch { /* ignore */ }
  }
}

// Mark the profile as having exited cleanly so Chrome never shows a
// "restore pages?" bubble or reopens an extra empty window on the next launch —
// that stray second window is what kept the instance (and the launcher) alive
// after the app window was closed.
function markProfileCleanExit() {
  const prefs = path.join(APP_PROFILE, "Default", "Preferences");
  try {
    if (!fs.existsSync(prefs)) return;
    const d = JSON.parse(fs.readFileSync(prefs, "utf8"));
    d.profile = d.profile || {};
    d.profile.exit_type = "Normal";
    d.profile.exited_cleanly = true;
    fs.writeFileSync(prefs, JSON.stringify(d));
  } catch { /* worst case: a restore bubble appears once — harmless */ }
}

// How many app WINDOWS/tabs are open, via Chrome's DevTools endpoint. We can't
// use process presence: on macOS, closing the app window leaves the Chrome
// instance running windowless (background processes persist), so a
// process-based check would think the window is still open forever. The
// DevTools "page" targets disappear the instant the window is closed.
async function appPageCount() {
  try {
    const res = await fetch(`http://localhost:${DEBUG_PORT}/json`);
    if (!res.ok) return 0;
    const list = await res.json();
    return (list || []).filter((t) => t.type === "page").length;
  } catch {
    return 0;   // DevTools unreachable → instance gone → treat as closed
  }
}

// Reliable shutdown for macOS/Linux: wait for the app window to appear, then for
// it to close (no DevTools "page" targets left), then tear everything down.
async function monitorWindow() {
  let appeared = false;
  for (let i = 0; i < 60 && !appeared; i++) { appeared = (await appPageCount()) >= 1; if (!appeared) await sleep(500); }
  if (!appeared) return;   // never confirmed a window — leave Ctrl-C in charge
  // Require several consecutive empty reads before declaring the window closed,
  // so a transient DevTools hiccup doesn't shut a live app down by mistake.
  let gone = 0;
  while (gone < 4) {
    await sleep(1000);
    if ((await appPageCount()) >= 1) gone = 0; else gone++;
  }
  log("App window closed.");
  cleanup(0);
}

// Other launch.mjs processes for THIS project (excluding ourselves). A previous
// run that didn't shut down cleanly leaves one of these holding the ports.
function priorLauncherPids() {
  try {
    const r = spawnSync("pgrep", ["-f", "launch.mjs"], { encoding: "utf8" });
    return (r.stdout || "").split(/\s+/).filter(Boolean).map(Number)
      .filter((p) => p && p !== process.pid)
      .filter((p) => {
        const c = spawnSync("ps", ["-o", "command=", "-p", String(p)], { encoding: "utf8" });
        return (c.stdout || "").includes(PROJECT);
      });
  } catch { return []; }
}

// Single-owner startup: shut down any previous Evidence Engine instance (its
// launcher, servers, and app window) so this run starts fresh and fully owns
// everything it spawns. This is what makes "close the window → everything dies"
// and "reopen → the app opens" reliable, instead of a stale Chrome hijacking the
// launch and opening a blank default-profile window.
async function reclaimFromPriorInstances() {
  if (IS_WIN) return;   // Windows keeps its existing child-exit flow
  const pids = priorLauncherPids();
  if (pids.length) {
    log("Closing a previous Evidence Engine instance...");
    // SIGTERM lets each old launcher run its own cleanup (killing its detached
    // backend/frontend process groups) before exiting.
    for (const pid of pids) { try { process.kill(pid, "SIGTERM"); } catch { /* gone */ } }
    for (let i = 0; i < 40 && priorLauncherPids().length; i++) await sleep(250);
  }
  // A launcher-less Chrome can still hold our profile (it survives its launcher).
  if (appWindowAlive()) {
    try { spawnSync("pkill", ["-f", path.basename(APP_PROFILE)], { stdio: "ignore" }); } catch { /* ignore */ }
    for (let i = 0; i < 20 && appWindowAlive(); i++) await sleep(200);
  }
  clearStaleProfileLock();
}

function cleanup(code = 0) {
  if (cleaned) return;
  cleaned = true;
  log("\nShutting down Evidence Engine...");
  for (const c of started) killChild(c);
  // Also kill the app-window Chrome instance itself — on macOS it lingers
  // windowless after the window is closed, which would keep the Dock icon lit
  // and hold the profile. (On Windows it exits on its own.)
  if (!IS_WIN) { try { spawnSync("pkill", ["-f", path.basename(APP_PROFILE)], { stdio: "ignore" }); } catch { /* ignore */ } }
  log("Stopped.");
  process.exit(code);
}

async function main() {
  log("=== Evidence Engine ===");
  process.on("SIGINT", () => cleanup(0));
  process.on("SIGTERM", () => cleanup(0));

  // Start as the single owner: reclaim (shut down) any leftover instance so this
  // run owns its servers and window, and closing the window later frees all of it.
  await reclaimFromPriorInstances();

  const spawnOpts = { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN };

  // 2) backend — run from an isolated venv so it always uses a working Python
  //    with correctly-built deps, regardless of what `python3` resolves to when
  //    launched from Finder/Explorer (which often finds an old system Python).
  ensureBackendEnv();
  if (await httpOk(`http://localhost:${BACKEND_PORT}/docs`)) {
    log(`Backend already running on :${BACKEND_PORT}`);
  } else {
    const backendPy = ensureBackendPython();
    if (!backendPy) {
      log("Could not set up the Python backend. Install Python 3.10+ from");
      log("https://www.python.org/downloads/ (or `brew install python`), then run this again.");
      openDefaultBrowser("https://www.python.org/downloads/");
      cleanup(1); return;
    }
    log(`Starting backend on :${BACKEND_PORT}...`);
    const b = spawn(backendPy, ["-m", "uvicorn", "api:app", "--app-dir", BACKEND_DIR, "--port", String(BACKEND_PORT)], spawnOpts);
    started.push(b);
  }

  // 3) frontend — serve the prebuilt production build with a tiny built-in static
  //    server (no runtime dependency on node_modules). vite is only a BUILD tool:
  //    a packaged bundle ships dist/ prebuilt and no node_modules, so it skips
  //    straight to serving; from-source runs build when the toolchain is present.
  const distIndex = path.join(PROJECT, "dist", "index.html");
  const viteJs = path.join(PROJECT, "node_modules", "vite", "bin", "vite.js");
  const wantBuild = !fs.existsSync(distIndex) || (fs.existsSync(viteJs) && needsBuild());
  if (wantBuild) {
    if (!fs.existsSync(viteJs) && !fs.existsSync(path.join(PROJECT, "node_modules"))) {
      log("First run: installing frontend dependencies (a few minutes)...");
      const r = spawnSync(IS_WIN ? "npm.cmd" : "npm", ["install"], { cwd: PROJECT, stdio: "inherit" });
      if (r.status !== 0) { log("npm install failed."); process.exit(1); }
    }
    if (fs.existsSync(viteJs)) {
      log("Building the app (first run or sources changed; ~a few seconds)...");
      const bres = spawnSync(process.execPath, [viteJs, "build"], { cwd: PROJECT, stdio: "ignore" });
      if (bres.status !== 0) { log("Build failed."); cleanup(1); return; }
    }
  }
  if (!fs.existsSync(distIndex)) {
    log("No frontend build (dist/) found and no build toolchain available."); cleanup(1); return;
  }
  {
    log(`Serving the app on :${FRONTEND_PORT}...`);
    serveFrontend(path.join(PROJECT, "dist"), FRONTEND_PORT);
  }

  // 4) Wait only for the FRONTEND (the cached production build) — that comes up
  //    in ~a second. Do NOT block on the backend or Ollama: the backend is
  //    already spawned and keeps importing its heavy deps (langchain/torch) in
  //    the background while the window is already on screen. The app shows a
  //    "starting the local engine" indicator until the backend answers. This is
  //    what makes Evidence Engine appear fast instead of waiting on imports.
  const ok = await waitHealthy([APP_URL], 60);
  if (!ok) { log("Frontend did not come up in time."); cleanup(1); return; }

  // 5) open in its own window and wait; closing it shuts everything down
  clearStaleProfileLock();   // recover from any previous unclean exit
  markProfileCleanExit();    // never restore a stray empty window on launch
  const browser = findChrome();
  if (browser) {
    log("Opening Evidence Engine — CLOSE THE APP WINDOW to shut everything down.");
    const win = spawn(browser, [
      `--app=${APP_LAUNCH_URL}`,
      `--user-data-dir=${APP_PROFILE}`,
      `--remote-debugging-port=${DEBUG_PORT}`,
      "--no-first-run", "--no-default-browser-check",
      "--disable-session-crashed-bubble", "--hide-crash-restore-bubble",
    ], { stdio: "ignore" });
    heldWindow = win;   // keep a live reference so the child isn't garbage-collected
    win.on("error", () => { openDefaultBrowser(APP_LAUNCH_URL); keepAlive(); });
    if (IS_WIN) {
      // Windows: rely on the child's exit (handoff-aware), as before.
      const openedAt = Date.now();
      win.on("exit", () => {
        if (Date.now() - openedAt < 4000) {
          log("An Evidence Engine window is already open.");
          if (started.length === 0) process.exit(0);
          keepAlive();
        } else {
          cleanup(0);
        }
      });
    } else {
      // macOS/Linux: Chrome can let our spawned child exit immediately after
      // handing the URL to another process, so we cannot trust its exit event.
      // Stay alive and drive shutdown off whether the profile's window is still
      // open. This is what makes closing the window reliably free everything so
      // a reopen works without killing processes by hand.
      keepAlive();
      monitorWindow();
    }
  } else {
    openDefaultBrowser(APP_LAUNCH_URL);
    keepAlive();
  }

  // 6) Warm up the backend + Ollama in the BACKGROUND. The window is already
  //    open, so none of this delays first paint; the UI reflects progress via
  //    its own /api/health poll.
  warmUp();
}

// Background warm-up (runs after the window is already open). The backend is
// already spawned; wait for it to finish importing, then install/start Ollama
// and pull the default screening model. Nothing here blocks the app window.
async function warmUp() {
  // Backend is already starting; give it time to finish importing heavy deps.
  for (let i = 0; i < 180 && !(await httpOk(`http://localhost:${BACKEND_PORT}/api/health`)); i++) await sleep(1000);

  // Ollama powers the default local (private) screening model. Fully automatic:
  // install it if missing, start it, and pull the default model in the background.
  let ollamaBin = findOllama();
  if (!(await httpOk(OLLAMA_URL))) {
    if (!ollamaBin) ollamaBin = installOllama();
    if (ollamaBin) {
      log("Starting Ollama (local AI models)...");
      const o = spawn(ollamaBin, ["serve"], { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN });
      started.push(o);
      for (let i = 0; i < 20 && !(await httpOk(OLLAMA_URL)); i++) await sleep(1000);
    }
  }
  if (ollamaBin && (await httpOk(OLLAMA_URL))) {
    await ensureModel(ollamaBin);
  } else {
    log("Note: couldn't set up Ollama automatically (offline?). Local models need it —");
    log("      install from https://ollama.com, or pick a cloud model in the sidebar.");
  }
}

function keepAlive() {
  log(`\nEvidence Engine is running at ${APP_URL}`);
  log(">>> Press Ctrl-C in this window to shut everything down. <<<");
  setInterval(() => {}, 1 << 30);   // stay alive until SIGINT
}

main().catch((e) => { log("Launcher error: " + e.message); cleanup(1); });
