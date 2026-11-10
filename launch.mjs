#!/usr/bin/env node
// Cross-platform Evidence Engine launcher (macOS + Windows + Linux).
// Starts the backend + frontend, opens the app in its own browser window, and
// shuts everything down when that window is closed. Run via launch.command
// (macOS) or launch.bat (Windows), or directly: `node launch.mjs`.

import { spawn, spawnSync } from "node:child_process";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const PROJECT = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = path.join(PROJECT, "Backend");
// Both ports are PREFERENCES, not requirements. Anything already listening on
// 5180 (a dev `vite preview`, another app) used to make the launcher wait 60 s
// and then exit with no window at all, which looks exactly like the app being
// broken. Both now shift to the next free port instead.
// The backend side is pinned to ONE address. uvicorn binds 127.0.0.1 by default
// while "localhost" resolves to ::1 first here, so the health probe, the proxy and
// the server could each end up on a different address family of the same port.
// (The app window deliberately stays on localhost: that is its browser origin, and
// changing it would discard whatever the app has stored there.)
const LOCAL = "127.0.0.1";
const FRONTEND_PORT_PREF = 5180;
let FRONTEND_PORT = FRONTEND_PORT_PREF;
let BACKEND_PORT = 8000;   // resolved at startup; may shift if 8000 is taken by a foreign process
// Recomputed once the frontend port is known. The `?new=1` flag on the launch URL
// tells the app a fresh launch happened so it starts on Home with a new session
// instead of restoring the last one (an in-app refresh has no flag and still
// restores). Health checks use the bare appUrl().
const appUrl = () => `http://localhost:${FRONTEND_PORT}/`;
const appLaunchUrl = () => `${appUrl()}?new=1`;
const IS_WIN = process.platform === "win32";
const OLLAMA_URL = "http://localhost:11434/";
const EE_CACHE = path.join(os.homedir(), ".evidence-engine");     // downloaded tools
const LOCK_FILE = path.join(EE_CACHE, "launcher.pid");            // single-instance guard
const LOG_FILE = path.join(EE_CACHE, "launcher.log");             // what this run did
const BACKEND_LOG = path.join(EE_CACHE, "backend.log");           // the backend's own output
try { fs.mkdirSync(EE_CACHE, { recursive: true }); } catch { /* exists */ }
// Two models, because the stages want different things of them. LEADS-Mistral
// is fine-tuned for per-PICO screening verdicts and wins on that; asked to
// extract fields from a paper it returns no usable JSON at all (measured: 0 of
// 8 known values, four times slower), which is why the backend routes every
// non-screening task to a general-purpose model. That model has to be on the
// machine for extraction to work, so it is downloaded here too.
const MODELS = [
  { tag: "hf.co/mradermacher/leads-mistral-7b-v1-GGUF", what: "screening" },   // ~4 GB
  { tag: "qwen2.5:7b", what: "extraction and everything else" },               // ~4.7 GB
];

const started = [];               // child processes we spawned (to clean up)
// Set when the backend process exits by itself. Everything the app asks of /api
// then gets a clear 503 instead of a dead socket, so the UI can explain itself.
let backendFailure = null;
let shuttingDown = false;         // suppresses the failure notice during a normal quit
let heldWindow = null;            // the Chrome child — kept referenced so Node
                                  // (v22) doesn't GC and tear it down mid-run.
let cleaned = false;

// The .app starts us with stdout pointed at /dev/null, so stdout alone left no
// trace of what a launch did. Everything is mirrored into LOG_FILE, which is the
// file the failure messages tell people to read.
function log(m) {
  process.stdout.write(m + "\n");
  try { fs.appendFileSync(LOG_FILE, "[" + new Date().toISOString() + "] " + m + "\n"); } catch { /* best effort */ }
}

// Fresh backend.log per run, handed to the child as its stdout/stderr.
function openBackendLog() {
  try {
    fs.writeFileSync(BACKEND_LOG, "[" + new Date().toISOString() + "] backend starting\n");
    return fs.openSync(BACKEND_LOG, "a");
  } catch { return null; }
}

function httpOk(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => { res.resume(); resolve(res.statusCode > 0 && res.statusCode < 500); });
    req.on("error", () => resolve(false));
    req.setTimeout(2000, () => { req.destroy(); resolve(false); });
  });
}

// True only if OUR FastAPI backend answers on this port. A dumb static server or
// other program squatting the port returns a 200/404 too, so we must confirm the
// health endpoint returns our JSON shape — otherwise the launcher would mistake a
// foreign process for the backend and never start the real one (the "sessions
// gone / stuck on Starting engine" bug).
function isOurBackend(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://${LOCAL}:${port}/api/health`, (res) => {
      if (res.statusCode !== 200) { res.resume(); return resolve(false); }
      let body = "";
      res.on("data", (c) => { body += c; if (body.length > 4096) req.destroy(); });
      res.on("end", () => { try { resolve(JSON.parse(body)?.ok === true); } catch { resolve(false); } });
    });
    req.on("error", () => resolve(false));
    req.setTimeout(2000, () => { req.destroy(); resolve(false); });
  });
}

// True if we can open a TCP connection to this port, i.e. something is already
// serving it.
function connectable(port, host) {
  return new Promise((resolve) => {
    const s = net.connect({ port, host });
    const done = (v) => { s.destroy(); resolve(v); };
    s.once("connect", () => done(true));
    s.once("error", () => done(false));
    s.setTimeout(1000, () => done(false));
  });
}

// Checked on both address families: a listener bound to only one of them (a
// `vite preview` on ::1, a `python3 -m http.server` on 0.0.0.0) still owns the
// port number as far as the app is concerned.
async function portBusy(port) {
  const [v4, v6] = await Promise.all([connectable(port, "127.0.0.1"), connectable(port, "::1")]);
  return v4 || v6;
}

// A port is usable only if nobody is serving it AND we can bind it. Binding on
// its own is not a valid test: on macOS a bind to 127.0.0.1 succeeds even while
// another process serves the same port on 0.0.0.0, so the launcher would adopt a
// foreign server's port and then proxy the app's /api calls straight to it. That
// is how a stray `python3 -m http.server` came to answer "501 Unsupported method
// ('POST')" inside the app.
async function portFree(port, host = LOCAL) {
  if (await portBusy(port)) return false;
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.once("listening", () => srv.close(() => resolve(true)));
    srv.listen({ port, host, exclusive: true });
  });
}

// Pick the port the app window is served on: the preferred one when free, else the
// next free port. Without this a foreign listener on 5180 stops the app starting.
async function pickFrontendPort() {
  if (await portFree(FRONTEND_PORT_PREF, "localhost")) return FRONTEND_PORT_PREF;
  for (let p = FRONTEND_PORT_PREF + 1; p < FRONTEND_PORT_PREF + 40; p++) {
    if (await portFree(p, "localhost")) return p;
  }
  return 0;   // let the OS choose; serveFrontend reports the bound port
}

// Pick the backend port: reuse ours if already running, else the default if free,
// else the next free port so a foreign program on 8000 can't break the app.
async function resolveBackendPort() {
  if (await isOurBackend(BACKEND_PORT)) return BACKEND_PORT;
  if (await portFree(BACKEND_PORT)) return BACKEND_PORT;
  for (let p = BACKEND_PORT + 1; p < BACKEND_PORT + 20; p++) {
    if (await portFree(p)) {
      log(`Port ${BACKEND_PORT} is in use by another program; using ${p} for the backend.`);
      return p;
    }
  }
  return BACKEND_PORT;
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

// Mutable state must never live inside the app bundle. An installed app (in
// /Applications, or Program Files) is not the user's to write into, and
// replacing the app to update it would discard the venv along with it. A source
// checkout keeps using Backend/.venv when it already exists, so run.sh,
// setup.sh and the test suite are unaffected.
const DEV_VENV = path.join(BACKEND_DIR, ".venv");
const VENV_DIR = fs.existsSync(DEV_VENV) ? DEV_VENV : path.join(EE_CACHE, "venv");
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

// Create the env file on first run so the user never has to. No API keys are
// required for the default local model; the only useful value is a real contact
// email for NCBI/PubMed + Unpaywall, which we auto-fill from git if available.
// Written beside the database rather than into the app, for the same reason as
// the venv: an installed app is read-only and gets replaced wholesale.
function ensureBackendEnv() {
  const devEnv = path.join(BACKEND_DIR, ".env");
  const envPath = fs.existsSync(devEnv) ? devEnv : path.join(EE_CACHE, ".env");
  const examplePath = path.join(BACKEND_DIR, ".env.example");
  if (fs.existsSync(envPath) || !fs.existsSync(examplePath)) return;
  try {
    let content = fs.readFileSync(examplePath, "utf8");
    const gitEmail = (spawnSync("git", ["config", "user.email"], { encoding: "utf8" }).stdout || "").trim();
    if (gitEmail && /^ENTREZ_EMAIL=/m.test(content)) {
      content = content.replace(/^ENTREZ_EMAIL=.*$/m, `ENTREZ_EMAIL=${gitEmail}`);
    }
    fs.writeFileSync(envPath, content);
    log(`Created ${envPath} (no API keys needed: the default model runs locally).`);
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

// Docling downloads its layout and table models from Hugging Face the first time
// it reads a document. Doing that here, once, in the background, means the first
// scanned PDF a reviewer opens is read straight away, and the app keeps working
// with no network afterwards. The marker file is what keeps it to once: the
// models live in the Hugging Face cache, which this launcher does not own.
const DOCLING_MARK = path.join(EE_CACHE, "docling-models-fetched");

function warmDocling(venvPy) {
  if (!venvPy || fs.existsSync(DOCLING_MARK)) return;
  const code = "from docling.document_converter import DocumentConverter; DocumentConverter()";
  const p = spawn(venvPy, ["-c", code], { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN });
  started.push(p);
  p.on("exit", (status) => {
    if (status === 0) {
      log("Document reader ready (scanned PDFs can be read).");
      try { fs.writeFileSync(DOCLING_MARK, new Date().toISOString()); } catch { /* retry next run */ }
    } else {
      log(`Document reader models not fetched (exit ${status}); will try again next launch.`);
    }
  });
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

// Pull whichever default models are missing. Runs in the BACKGROUND (about
// 9 GB between them) so the app opens immediately; Ollama resumes a pull if the
// app is quit part-way. One at a time: two concurrent pulls share the same
// bandwidth and finish later than two sequential ones.
async function ensureModel(ollamaBin) {
  let installed = [];
  try {
    const r = await fetch(`${OLLAMA_URL}api/tags`);
    installed = ((await r.json()).models || []).map((m) => m.name || "");
  } catch { /* server not answering tags yet — pull anyway */ }

  const missing = MODELS.filter((m) => !installed.some((t) => t.startsWith(m.tag)));
  if (!missing.length) return;

  for (const m of missing) {
    log(`Downloading the local model for ${m.what} in the background: ${m.tag}`);
  }
  log("The app is usable now; local AI works once the downloads finish.");

  const pullNext = (i) => {
    if (i >= missing.length) return;
    const p = spawn(ollamaBin, ["pull", missing[i].tag],
                    { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN });
    started.push(p);
    p.on("exit", (code) => {
      log(code === 0 ? `Downloaded ${missing[i].tag}`
                     : `Could not download ${missing[i].tag} (exit ${code})`);
      pullNext(i + 1);
    });
  };
  pullNext(0);
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
  // Chrome will not treat the page as installable unless the manifest arrives
  // as a manifest; served as a generic blob it is ignored.
  ".webmanifest": "application/manifest+json",
};

// Text-ish assets worth gzipping (images/fonts/wasm are already compressed).
const COMPRESSIBLE = new Set([".html", ".js", ".mjs", ".css", ".json", ".svg", ".map", ".csv", ".txt",
                              ".webmanifest"]);

// Send one static file with the same performance headers a real static server
// gives you: gzip when the client accepts it, long-lived immutable caching for
// content-hashed assets (and no-cache for index.html), and ETag revalidation.
// Without these the browser re-downloads the whole (multi-MB, uncompressed)
// bundle on every load, which is why the launcher felt slower than the app in a
// normal browser.
function sendFile(file, stat, req, res) {
  const ext = path.extname(file).toLowerCase();
  const etag = `W/"${stat.size.toString(16)}-${Math.round(stat.mtimeMs).toString(16)}"`;
  const isHtml = ext === ".html";
  // Vite emits content-hashed filenames under /assets/, so those are safe to
  // cache forever; index.html must always be revalidated so new builds show up.
  const hashed = file.includes(`${path.sep}assets${path.sep}`) && !isHtml;
  const headers = {
    "Content-Type": MIME[ext] || "application/octet-stream",
    "ETag": etag,
    "Cache-Control": isHtml ? "no-cache" : hashed ? "public, max-age=31536000, immutable" : "public, max-age=3600",
  };
  if (req.headers["if-none-match"] === etag) { res.writeHead(304, headers); res.end(); return; }
  const acceptsGzip = /\bgzip\b/.test(req.headers["accept-encoding"] || "");
  const onErr = () => { if (!res.headersSent) res.writeHead(500); res.end(); };
  if (acceptsGzip && COMPRESSIBLE.has(ext)) {
    headers["Content-Encoding"] = "gzip";
    headers["Vary"] = "Accept-Encoding";
    res.writeHead(200, headers);
    fs.createReadStream(file).on("error", onErr).pipe(zlib.createGzip()).on("error", onErr).pipe(res);
  } else {
    headers["Content-Length"] = stat.size;
    res.writeHead(200, headers);
    fs.createReadStream(file).on("error", onErr).pipe(res);
  }
}

// Serve the prebuilt SPA from distDir and reverse-proxy /api/* to the backend.
// Runs in-process (dies with the launcher). Using this instead of `vite preview`
// means the packaged bundle needs no node_modules at runtime — only the built
// dist/. Streaming-safe so backend SSE endpoints work.
function serveFrontend(distDir, port) {
  // Other launchers read the file, not our variable, so it has to stay fresh
  // for as long as a window is holding a stream open.
  const refresh = setInterval(() => { if (liveWindows > 0) recordBeat(true); }, 5000);
  if (typeof refresh.unref === "function") refresh.unref();
  const server = http.createServer((req, res) => {
    if (req.url.startsWith("/api")) {
      const up = http.request(
        { host: LOCAL, port: BACKEND_PORT, path: req.url, method: req.method, headers: req.headers },
        (r) => {
          // Only our backend may answer /api. Another program holding the port
          // replies with its own pages, and relaying those put raw HTML (such as
          // http.server's "501 Unsupported method ('POST')") inside the app
          // instead of a message it could act on. uvicorn always names itself in
          // the Server header, so anything else is reported as a failure.
          if (!/uvicorn/i.test(String(r.headers.server || ""))) {
            r.resume();
            res.writeHead(503, { "content-type": "application/json" });
            res.end(JSON.stringify({
              detail: "Another program is using port " + BACKEND_PORT + ", so the local engine is not "
                    + "the thing answering on it. Quit Evidence Engine and open it again to move to a free port.",
              state: "failed", port: BACKEND_PORT, log: LOG_FILE,
            }));
            return;
          }
          res.writeHead(r.statusCode || 502, r.headers);
          r.pipe(res);
        },
      );
      up.on("error", () => {
        if (res.headersSent) { res.end(); return; }
        // JSON either way: the app parses /api answers, and a plain-text body made
        // a readable "still starting" state look like a hard failure.
        if (backendFailure) {
          res.writeHead(503, { "content-type": "application/json" });
          res.end(JSON.stringify({
            detail: "The local engine stopped unexpectedly. Quit and reopen Evidence Engine; "
                  + "if it keeps happening, the log says why.",
            state: "failed", exit_code: backendFailure.code, signal: backendFailure.signal,
            log: backendFailure.log,
          }));
        } else {
          res.writeHead(503, { "content-type": "application/json", "retry-after": "2" });
          res.end(JSON.stringify({ detail: "The local engine is still starting.", state: "starting" }));
        }
      });
      req.pipe(up);
      return;
    }
    // The open window, holding the line (see index.html). This connection is
    // the only signal the launcher has that the app is still on screen, and
    // losing it is how it learns the window is gone — at once, not on a timer.
    if (req.url.startsWith("/__ee/stream")) {
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        "connection": "keep-alive",
      });
      res.write(": open\n\n");
      liveWindows += 1;
      recordBeat(true);
      // Ours, on the server side, where nothing throttles it: proxies and
      // browsers drop a stream that says nothing for long enough.
      const ping = setInterval(() => { try { res.write(": ping\n\n"); } catch { /* gone */ } }, 20000);
      let ended = false;
      const ended_ = () => {
        if (ended) return;
        ended = true;
        clearInterval(ping);
        liveWindows = Math.max(0, liveWindows - 1);
        if (liveWindows === 0) recordBeat(false);
      };
      req.on("close", ended_);
      res.on("close", ended_);
      return;
    }

    // The Dock icon was clicked again. Bring the window back, opening a fresh
    // one when the reviewer had closed it, so returning to a review costs a
    // keystroke rather than a cold start.
    if (req.url.startsWith("/__ee/open-window")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      void showWindow();
      return;
    }

    // Quit chosen from the Dock. The whole thing goes: window, engine, models.
    if (req.url.startsWith("/__ee/quit")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      setTimeout(() => cleanup(0), 300);
      return;
    }

    // The one thing the backend cannot do: quit the app and put the staged
    // update in its place. The swap runs from a script OUTSIDE the bundle,
    // after this process is gone, because a bundle cannot be replaced
    // underneath the node that is executing from it.
    if (req.url.startsWith("/__ee/apply-update")) {
      res.writeHead(200, { "content-type": "application/json" });
      const ok = applyUpdateOnExit();
      res.end(JSON.stringify(ok ? { ok: true } : { ok: false, detail: "No update is staged." }));
      if (ok) setTimeout(() => cleanup(0), 400);   // let the answer reach the page
      return;
    }

    let rel = decodeURIComponent((req.url.split("?")[0] || "/"));
    if (rel.endsWith("/")) rel += "index.html";
    let file = path.join(distDir, rel);
    if (!file.startsWith(distDir)) { res.writeHead(403); res.end(); return; }   // no traversal
    fs.stat(file, (err, st) => {
      // Missing file → SPA fallback to index.html (which we must re-stat).
      if (err || !st.isFile()) {
        file = path.join(distDir, "index.html");
        fs.stat(file, (e2, st2) => { if (e2 || !st2.isFile()) { res.writeHead(404); res.end(); return; } sendFile(file, st2, req, res); });
        return;
      }
      sendFile(file, st, req, res);
    });
  });
  server.on("error", (e) => log("Frontend server error: " + e.message));
  server.listen({ port, host: "localhost", exclusive: true }, () => {
    const bound = server.address();
    if (bound && typeof bound === "object" && bound.port !== FRONTEND_PORT) {
      FRONTEND_PORT = bound.port;   // the OS picked it; health checks and the window follow
    }
  });
  return server;
}

function killChild(child) {
  if (!child || child.killed) return;
  try {
    if (IS_WIN) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    else process.kill(-child.pid, "SIGTERM");   // whole process group (detached)
  } catch { /* already gone */ }
}

// The app window opens in the browser the reviewer already has running, not in
// a second Chrome of our own. A dedicated instance meant a second Chrome in the
// Dock beside the app's own icon, and it had to cold-start its own 300 MB
// profile on every first launch; handing the URL to the running Chrome costs
// nothing and shows one icon.
//
// That leaves no DevTools port to watch, and no process to count: the window is
// one window inside a browser full of other windows. So the window holds a
// connection open instead, and the launcher watches the connection.
//
// A connection, not a ping. The first attempt had the page send a heartbeat on
// a timer, and Chrome throttles timers in a window that is not in front to one
// tick a MINUTE: the app was declared closed, twice, while it sat there open.
// A socket is not a timer. It stays open while the window does and closes the
// instant it doesn't, however buried the window is.
//
// The state is mirrored to a file because a SECOND launcher (a Dock click while
// one is running) has to be able to read it too.
const BEAT_FILE = path.join(EE_CACHE, "window.beat");
const BEAT_GRACE_MS = 12000;   // the file is refreshed every 5s while a window is up
let liveWindows = 0;

function recordBeat(alive = true) {
  try {
    fs.mkdirSync(EE_CACHE, { recursive: true });
    fs.writeFileSync(BEAT_FILE, String(alive ? Date.now() : 0));
  } catch { /* ignore */ }
}

/** Is an Evidence Engine window open anywhere? */
function windowAlive() {
  try {
    const t = Number(fs.readFileSync(BEAT_FILE, "utf8")) || 0;
    return t > 0 && Date.now() - t < BEAT_GRACE_MS;
  } catch { return false; }
}

const IS_MAC = process.platform === "darwin";

/** Run an AppleScript, returning its output (empty on any failure). */
function osa(script) {
  if (!IS_MAC) return "";
  try {
    const r = spawnSync("osascript", ["-e", script], { encoding: "utf8" });
    return (r.stdout || "").trim();
  } catch { return ""; }
}

// `running of application` asks without launching it, which matters: the plain
// `tell application "Google Chrome"` form would START Chrome just to look.
const CHROME_TELL = (body) => `
if not (running of application "Google Chrome") then return "no"
tell application "Google Chrome"
${body}
end tell`;

/** Raise the existing app window. True if there was one. */
function raiseAppWindow() {
  const out = osa(CHROME_TELL(`
  set found to false
  repeat with w in windows
    repeat with t in tabs of w
      if (URL of t as string) starts with ${JSON.stringify(appUrl())} then
        set found to true
        set index of w to 1
        exit repeat
      end if
    end repeat
    if found then exit repeat
  end repeat
  if found then activate
  return (found as string)`));
  return out === "true";
}

/** Close the app's window, and only it: the reviewer's browser stays open. */
function closeAppWindow() {
  osa(CHROME_TELL(`
  set doomed to {}
  repeat with w in windows
    repeat with t in tabs of w
      if (URL of t as string) starts with ${JSON.stringify(appUrl())} then set end of doomed to t
    end repeat
  end repeat
  repeat with t in doomed
    try
      close t
    end try
  end repeat
  return "ok"`));
}

// Single-instance lock: the running launcher records its pid; a launch that
// finds a LIVE pid there knows an instance is already running. Filesystem-based,
// so it's reliable even for a Finder/Dock launch where localhost probes can be
// flaky — that flakiness is what let a second click fall through and open a
// stray blank window.
// Line 1 is the pid. Line 2, once the port is settled, is the URL the app is
// served at: the Dock applet reads it to reach a running launcher, since the
// port is not fixed and it has no other way to find out which one we settled on.
function lockLines() {
  try { return fs.readFileSync(LOCK_FILE, "utf8").split("\n"); } catch { return []; }
}
function launcherLockAlive() {
  try {
    const pid = parseInt((lockLines()[0] || "").trim(), 10);
    if (pid && pid !== process.pid) { process.kill(pid, 0); return true; }   // throws if dead
  } catch { /* missing, stale, or dead pid */ }
  return false;
}
function writeLock(url) {
  try {
    fs.mkdirSync(EE_CACHE, { recursive: true });
    fs.writeFileSync(LOCK_FILE, url ? `${process.pid}\n${url}\n` : String(process.pid));
  } catch { /* ignore */ }
}
function clearLock() {
  try {
    const pid = parseInt(fs.readFileSync(LOCK_FILE, "utf8").trim(), 10);
    if (pid === process.pid) fs.rmSync(LOCK_FILE, { force: true });   // only remove our own
  } catch { /* ignore */ }
}

// Bring an already-running app window to the front, via the DevTools activate
// endpoint (used when the app is launched again while it's already open).
async function focusExistingWindow() {
  if (raiseAppWindow()) return;
  // No window to raise. Opening the app SOMEWHERE ELSE here is what used to put
  // two of it on screen at once — the old fallback reached for the default
  // browser whenever this check was merely slow, so a reviewer who already had
  // the app open got a second copy of it in another window. If there is no
  // window, open one the single way the app is ever opened.
  if (!windowAlive()) await showWindow();
}

// True if an Evidence Engine window is already open. Primary signal is a live
// DevTools "page" (only true when a window is actually showing); as a fallback
// (e.g. an instance whose debug port is briefly unresponsive) treat "frontend
// server up AND an app-profile Chrome window process alive" as open too.
async function alreadyOpen() {
  if (launcherLockAlive()) return true;   // instant, filesystem — most reliable
  if (windowAlive()) return true;
  return false;
}

// Reliable shutdown for macOS/Linux: wait for the app window to appear, then for
// it to close (no DevTools "page" targets left), then tear everything down.
// Set by the Dock applet, which owns the app's lifetime when it is there: the
// tile stays, closing the window is not quitting, and Quit from the Dock is.
// Without it (a source checkout, Linux, a plain `node launch.mjs`) closing the
// window still has to stop everything, or the engine would run on with nothing
// on screen to stop it with.
const DOCK_OWNED = process.env["EE_DOCK_OWNER"] === "1";
let watchingWindow = false;

async function monitorWindow() {
  if (watchingWindow) return;      // showWindow may be called many times
  watchingWindow = true;
  let appeared = false;
  for (let i = 0; i < 120 && !appeared; i++) { appeared = windowAlive(); if (!appeared) await sleep(500); }
  if (!appeared) { watchingWindow = false; return; }   // never reported in — leave Ctrl-C in charge
  // windowAlive() already allows three missed beats, which is what rides out a
  // page too busy to beat (a long import) without shutting a live app down.
  while (windowAlive()) await sleep(1000);
  if (DOCK_OWNED) {
    // The reviewer closed the window, not the app. Everything stays warm so
    // clicking the Dock icon puts them back in their review immediately.
    log("App window closed. Still running: click the Dock icon to come back.");
    watchingWindow = false;
    return;
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
  // Nothing else to reclaim: the window belongs to the reviewer's own browser,
  // which we never start, never kill, and never hold a lock on.
}

function cleanup(code = 0) {
  if (cleaned) return;
  cleaned = true;
  shuttingDown = true;   // a child exiting now is us quitting, not a crash
  log("\nShutting down Evidence Engine...");
  for (const c of started) killChild(c);
  // Close the app's own window, and nothing else. The browser around it is the
  // reviewer's, with their other work in it: quitting Evidence Engine closes
  // one window, never their Chrome.
  closeAppWindow();
  recordBeat(false);
  clearLock();
  log("Stopped.");
  process.exit(code);
}

async function main() {
  log("=== Evidence Engine ===");
  process.on("SIGINT", () => cleanup(0));
  process.on("SIGTERM", () => cleanup(0));

  // Already running? Just bring the existing window to the front and exit —
  // don't start a second copy or open a stray browser tab. (Use process.exit,
  // NOT cleanup, which would kill the very window we're focusing.)
  if (await alreadyOpen()) {
    log("Evidence Engine is already open — bringing it to the front.");
    await focusExistingWindow();
    process.exit(0);
  }

  // Otherwise start as the single owner: reclaim (shut down) any leftover
  // instance so this run owns its servers and window, and closing the window
  // later frees all of it.
  await reclaimFromPriorInstances();
  writeLock();   // claim the single-instance lock so later launches just focus us

  const spawnOpts = { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN };

  // 2) backend — run from an isolated venv so it always uses a working Python
  //    with correctly-built deps, regardless of what `python3` resolves to when
  //    launched from Finder/Explorer (which often finds an old system Python).
  ensureBackendEnv();
  // Choose the backend port before wiring the proxy: reuse a running Evidence
  // Engine backend, otherwise the default port if free, otherwise the next free
  // one. This is verified against /api/health so a foreign server squatting :8000
  // can never masquerade as the backend (which left /api calls 404-ing — sessions
  // vanished and "Starting engine" never cleared).
  BACKEND_PORT = await resolveBackendPort();
  if (await isOurBackend(BACKEND_PORT)) {
    log(`Backend already running on :${BACKEND_PORT}`);
  } else {
    const backendPy = ensureBackendPython();
    if (!backendPy) {
      log("Could not set up the Python backend. Install Python 3.10+ from");
      log("https://www.python.org/downloads/ (or `brew install python`), then run this again.");
      openDefaultBrowser("https://www.python.org/downloads/");
      cleanup(1); return;
    }
    // Nothing here blocks the window: the models are fetched beside the app.
    warmDocling(backendPy);
    log(`Starting backend on :${BACKEND_PORT}...`);
    // --host pins the backend to the address the proxy and health probe use, and
    // its output is captured so a crash can actually be diagnosed afterwards.
    const blog = openBackendLog();
    const b = spawn(backendPy,
      ["-m", "uvicorn", "api:app", "--app-dir", BACKEND_DIR, "--host", LOCAL, "--port", String(BACKEND_PORT)],
      blog === null ? spawnOpts : { ...spawnOpts, stdio: ["ignore", blog, blog] });
    started.push(b);
    // If the backend exits on its own the app would otherwise sit on "the local
    // engine is still starting" forever. Record it so /api answers explain the
    // failure and point at the log instead of timing out silently.
    b.on("exit", (code, signal) => {
      if (shuttingDown) return;
      backendFailure = {
        code, signal,
        log: BACKEND_LOG,
        at: new Date().toISOString(),
      };
      log(`The backend stopped unexpectedly (exit ${code ?? signal}). The app window will say so.`);
    });
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
    FRONTEND_PORT = await pickFrontendPort();
    log(`Serving the app on :${FRONTEND_PORT}...`);
    serveFrontend(path.join(PROJECT, "dist"), FRONTEND_PORT);
  }

  // 4) Wait only for the FRONTEND (the cached production build) — that comes up
  //    in ~a second. Do NOT block on the backend or Ollama: the backend is
  //    already spawned and keeps importing its heavy deps (langchain/torch) in
  //    the background while the window is already on screen. The app shows a
  //    "starting the local engine" indicator until the backend answers. This is
  //    what makes Evidence Engine appear fast instead of waiting on imports.
  const ok = await waitHealthy([appUrl()], 60);
  if (!ok) { log("Frontend did not come up in time."); cleanup(1); return; }
  writeLock(appUrl());   // the port is settled, so a later launch can find this window

  // 5) open in its own window
  showWindow();

  // 6) Warm up the backend + Ollama in the BACKGROUND. The window is already
  //    open, so none of this delays first paint; the UI reflects progress via
  //    its own /api/health poll.
  warmUp();
}

/**
 * Open the app window, or bring the open one forward.
 *
 * Called at startup and again whenever the Dock icon is clicked, because a
 * reviewer who closed the window and came back wants it open, not a cold
 * start. The engine keeps running in between, which is what makes coming back
 * instant; the Dock icon's Quit is what stops it.
 */
async function showWindow() {
  if (windowAlive() && raiseAppWindow()) return;
  const browser = findChrome();
  if (browser) {
    log("Opening Evidence Engine.");
    // No --user-data-dir and no debugging port, which is the whole point:
    // Chrome's process singleton hands --app to the browser the reviewer
    // already has running, and that browser opens the window. One Chrome in
    // the Dock, and no second 300 MB profile to cold-start before anything
    // appears. The flags that used to be here (GPU, occlusion) existed to make
    // a second instance behave like a normal window; a normal window needs no
    // persuading, and a handed-off command line would ignore them anyway.
    // Detached: a launcher that is only here to re-open the window exits the
    // moment it has asked, and the asking must outlive it.
    const win = spawn(browser, [
      `--app=${appLaunchUrl()}`,
      "--no-first-run", "--no-default-browser-check",
    ], { stdio: "ignore", detached: true });
    win.unref();
    heldWindow = win;   // keep a live reference so the child isn't garbage-collected
    win.on("error", () => { openDefaultBrowser(appLaunchUrl()); keepAlive(); });
    // The child exits the moment it has handed the URL over, so its exit says
    // nothing about the window. The page reports in instead, on every platform.
    keepAlive();
    monitorWindow();
  } else {
    openDefaultBrowser(appLaunchUrl());
    keepAlive();
    monitorWindow();
  }
}

// Background warm-up (runs after the window is already open). The backend is
// already spawned; wait for it to finish importing, then install/start Ollama
// and pull the default screening model. Nothing here blocks the app window.
async function warmUp() {
  // Backend is already starting; give it time to finish importing heavy deps.
  for (let i = 0; i < 180 && !(await isOurBackend(BACKEND_PORT)); i++) await sleep(1000);

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

// ---------------------------------------------------------------------------
// Applying a staged update
// ---------------------------------------------------------------------------
//
// The reviewer's work lives in ~/.evidence-engine and the app lives somewhere
// else entirely, which is what makes this safe: only the bundle is replaced,
// and the data directory is never touched. The old copy is kept until the new
// one has been shown to start, so a bad update is recoverable rather than a
// reinstall.
const UPDATE_DIR = path.join(EE_CACHE, "updates");

function stagedUpdate() {
  try {
    const d = JSON.parse(fs.readFileSync(path.join(UPDATE_DIR, "staged.json"), "utf8"));
    return d && d.path && fs.existsSync(d.path) ? d : null;
  } catch { return null; }
}

/**
 * The .app or install folder to replace, or "" when there is nothing to.
 *
 * A source checkout returns "" on purpose. PROJECT is the repository there,
 * and an updater that treated it as an install would move a developer's work
 * aside and drop a packaged app on top of it. A packaged install is the only
 * thing this is allowed to touch, and it is recognised by the bundle layout or
 * by the VERSION stamp that packaging/build.sh writes and a clone never has.
 */
function installedRoot() {
  const app = path.resolve(PROJECT, "..", "..", "..");   // Resources/app -> .app
  if (!IS_WIN && app.endsWith(".app") && fs.existsSync(app)) return app;
  const stamped = fs.existsSync(path.join(PROJECT, "VERSION"));
  const checkout = fs.existsSync(path.join(PROJECT, ".git"));
  return stamped && !checkout ? PROJECT : "";
}

/**
 * Hand the swap to a detached script and report whether one was armed.
 *
 * Nothing is replaced in this process: the script waits for this pid to go,
 * moves the old copy aside, puts the new one in place, and starts it again.
 * If the new copy will not start it puts the old one back.
 */
function applyUpdateOnExit() {
  const staged = stagedUpdate();
  if (!staged) return false;
  const target = installedRoot();
  if (!target) {
    log("Running from a source checkout, so there is no install to replace.");
    return false;
  }
  // Like for like. A staged .app belongs over a .app and nothing else, which
  // is the check that stops a wrong path turning a swap into a loss.
  if (staged.path.endsWith(".app") !== target.endsWith(".app")) {
    log("The staged update does not match this install, so it was not applied.");
    return false;
  }
  try {
    if (IS_WIN) {
      // The Windows asset is the silent installer, which knows how to replace
      // an install that is already there.
      const cmd = path.join(UPDATE_DIR, "apply.cmd");
      fs.writeFileSync(cmd, [
        "@echo off",
        `:wait`,
        `tasklist /FI "PID eq ${process.pid}" | find "${process.pid}" >nul && (timeout /t 1 >nul & goto wait)`,
        `"${staged.path}" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`,
        `del "%~f0"`,
      ].join("\r\n"));
      spawn("cmd", ["/c", cmd], { detached: true, stdio: "ignore", windowsHide: true }).unref();
      return true;
    }
    const sh = path.join(UPDATE_DIR, "apply.sh");
    fs.writeFileSync(sh, applyScript(process.pid, staged.path, target), { mode: 0o755 });
    spawn("/bin/bash", [sh], { detached: true, stdio: "ignore" }).unref();
    return true;
  } catch (e) {
    log("Could not start the updater: " + e.message);
    return false;
  }
}

/**
 * The swap, as a shell script so it outlives the process being replaced.
 *
 * Deliberately blunt about what it will not do: it refuses to run unless the
 * staged copy is there and the target is the app, so a wrong path can never
 * turn this into a delete of something else.
 */
function applyScript(pid, stagedPath, target) {
  const open = process.platform === "darwin"
    ? `open -a "$TARGET"`
    : `nohup "$TARGET/launch.sh" >/dev/null 2>&1 &`;
  return `#!/bin/bash
# Replaces Evidence Engine with the staged copy, once the running one is gone.
# The reviewer's reviews are in ~/.evidence-engine and are not touched here.
set -u
STAGED=${JSON.stringify(stagedPath)}
TARGET=${JSON.stringify(target)}
BACKUP="$TARGET.old-$$"

[ -e "$STAGED" ] || exit 1
[ -e "$TARGET" ] || exit 1

# Wait for the app to quit, but never forever.
for _ in $(seq 1 60); do
  kill -0 ${pid} 2>/dev/null || break
  sleep 1
done

mv "$TARGET" "$BACKUP" || exit 1
if mv "$STAGED" "$TARGET"; then
  rm -rf "$BACKUP"
else
  # Could not put the new one in place: restore what was working.
  mv "$BACKUP" "$TARGET"
  exit 1
fi

rm -f ${JSON.stringify(path.join(path.dirname(stagedPath), "staged.json"))}
sleep 1
${open}
`;
}

function keepAlive() {
  log(`\nEvidence Engine is running at ${appUrl()}`);
  log(">>> Press Ctrl-C in this window to shut everything down. <<<");
  setInterval(() => {}, 1 << 30);   // stay alive until SIGINT
}

main().catch((e) => { log("Launcher error: " + e.message); cleanup(1); });
