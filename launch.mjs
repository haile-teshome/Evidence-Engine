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

// pick a python that has the backend deps; else the first python found (for install)
function pickPython() {
  const cands = IS_WIN ? ["python", "py", "python3"] : ["python3", "python"];
  let firstFound = null;
  for (const c of cands) {
    const v = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (v.status === 0 || v.status === null) {
      if (!firstFound) firstFound = c;
      const dep = spawnSync(c, ["-c", "import fastapi, uvicorn"], { stdio: "ignore" });
      if (dep.status === 0) return { cmd: c, hasDeps: true };
    }
  }
  return { cmd: firstFound, hasDeps: false };
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
  const py = pickPython();
  if (!py.cmd) { log("Python not found. Install Python 3."); process.exit(1); }
  if (!py.hasDeps && fs.existsSync(path.join(BACKEND_DIR, "requirements.txt"))) {
    log("First run: installing backend dependencies...");
    spawnSync(py.cmd, ["-m", "pip", "install", "-r", path.join(BACKEND_DIR, "requirements.txt")],
      { cwd: PROJECT, stdio: "inherit" });
  }

  const spawnOpts = { cwd: PROJECT, stdio: "ignore", detached: !IS_WIN };

  // 2) backend
  if (await httpOk(`http://localhost:${BACKEND_PORT}/docs`)) {
    log(`Backend already running on :${BACKEND_PORT}`);
  } else {
    log(`Starting backend on :${BACKEND_PORT}...`);
    const b = spawn(py.cmd, ["-m", "uvicorn", "api:app", "--app-dir", BACKEND_DIR, "--port", String(BACKEND_PORT)], spawnOpts);
    started.push(b);
  }

  // 3) frontend (run vite's JS entry with this node -> works the same on all OSes)
  if (await httpOk(APP_URL)) {
    log(`Frontend already running on :${FRONTEND_PORT}`);
  } else {
    log(`Starting frontend on :${FRONTEND_PORT}...`);
    const viteJs = path.join(PROJECT, "node_modules", "vite", "bin", "vite.js");
    const f = spawn(process.execPath, [viteJs, "--port", String(FRONTEND_PORT), "--strictPort"], spawnOpts);
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
    win.on("exit", () => cleanup(0));
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
