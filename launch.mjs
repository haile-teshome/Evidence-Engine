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
const IS_WIN = process.platform === "win32";

const started = [];               // child processes we spawned (to clean up)
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
    spawnSync(VENV_PY, ["-m", "pip", "install", "--upgrade", "pip"], { stdio: "ignore" });
    const inst = spawnSync(VENV_PY, ["-m", "pip", "install", "-r", req], { stdio: "inherit" });
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

function openDefaultBrowser(url) {
  if (process.platform === "darwin") spawn("open", [url], { stdio: "ignore", detached: true }).unref();
  else if (IS_WIN) spawn("cmd", ["/c", "start", "", url], { stdio: "ignore", detached: true }).unref();
  else spawn("xdg-open", [url], { stdio: "ignore", detached: true }).unref();
}

function killChild(child) {
  if (!child || child.killed) return;
  try {
    if (IS_WIN) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    else process.kill(-child.pid, "SIGTERM");   // whole process group (detached)
  } catch { /* already gone */ }
}

function cleanup(code = 0) {
  if (cleaned) return;
  cleaned = true;
  log("\nShutting down Evidence Engine...");
  for (const c of started) killChild(c);
  log("Stopped.");
  process.exit(code);
}

async function main() {
  log("=== Evidence Engine ===");
  process.on("SIGINT", () => cleanup(0));
  process.on("SIGTERM", () => cleanup(0));

  // 1) first-run deps
  if (!fs.existsSync(path.join(PROJECT, "node_modules"))) {
    log("First run: installing frontend dependencies (a few minutes)...");
    const r = spawnSync(IS_WIN ? "npm.cmd" : "npm", ["install"], { cwd: PROJECT, stdio: "inherit" });
    if (r.status !== 0) { log("npm install failed."); process.exit(1); }
  }
  const spawnOpts = { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN };

  // 2) backend — run from an isolated venv so it always uses a working Python
  //    with correctly-built deps, regardless of what `python3` resolves to when
  //    launched from Finder/Explorer (which often finds an old system Python).
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

  // 3) frontend — serve a PRODUCTION build (fast: minified assets + React in
  //    production mode), not the dev server. Build once and cache in dist/;
  //    rebuild only when a source file is newer than the last build.
  const viteJs = path.join(PROJECT, "node_modules", "vite", "bin", "vite.js");
  if (await httpOk(APP_URL)) {
    log(`Frontend already running on :${FRONTEND_PORT}`);
  } else {
    if (needsBuild()) {
      log("Building the app (first run or sources changed; ~a few seconds)...");
      const bres = spawnSync(process.execPath, [viteJs, "build"], { cwd: PROJECT, stdio: "ignore" });
      if (bres.status !== 0 || !fs.existsSync(path.join(PROJECT, "dist", "index.html"))) {
        log("Build failed."); cleanup(1); return;
      }
    }
    log(`Starting frontend on :${FRONTEND_PORT}...`);
    const f = spawn(process.execPath, [viteJs, "preview", "--port", String(FRONTEND_PORT), "--strictPort"], spawnOpts);
    started.push(f);
  }

  // 4) wait until healthy
  const ok = await waitHealthy([APP_URL, `http://localhost:${BACKEND_PORT}/docs`], 90);
  if (!ok) { log("Services did not come up in time."); cleanup(1); return; }

  if (!(await httpOk("http://localhost:11434/"))) {
    log("Note: Ollama not running (:11434). Local-model features need it — open the Ollama app.");
  }

  // 5) open in its own window and wait; closing it shuts everything down
  const browser = findChrome();
  if (browser) {
    log("Opening Evidence Engine — CLOSE THE APP WINDOW to shut everything down.");
    const win = spawn(browser, [
      `--app=${APP_URL}`,
      `--user-data-dir=${path.join(os.homedir(), ".evidence-engine-appwin")}`,
      "--no-first-run", "--no-default-browser-check",
    ], { stdio: "ignore" });
    const openedAt = Date.now();
    win.on("exit", () => {
      // If Chrome exits almost immediately it didn't own the window — it handed
      // the URL off to an Evidence Engine window that's ALREADY open (same
      // profile). Tearing the servers down here would break that live window,
      // which is the "it won't open" bug. So only shut down on a real close.
      if (Date.now() - openedAt < 4000) {
        log("An Evidence Engine window is already open.");
        if (started.length === 0) process.exit(0);   // another launcher owns the servers
        keepAlive();                                  // we own them; idle until closed/Ctrl-C
      } else {
        cleanup(0);
      }
    });
    win.on("error", () => { openDefaultBrowser(APP_URL); keepAlive(); });
  } else {
    openDefaultBrowser(APP_URL);
    keepAlive();
  }
}

function keepAlive() {
  log(`\nEvidence Engine is running at ${APP_URL}`);
  log(">>> Press Ctrl-C in this window to shut everything down. <<<");
  setInterval(() => {}, 1 << 30);   // stay alive until SIGINT
}

main().catch((e) => { log("Launcher error: " + e.message); cleanup(1); });
