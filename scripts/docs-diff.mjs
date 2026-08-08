#!/usr/bin/env node
// Helper for the docs auto-update workflow.
//
//   node scripts/docs-diff.mjs            print what changed since docs were synced
//   node scripts/docs-diff.mjs --commit   advance docs/.docs-sync to current HEAD
//
// The marker file docs/.docs-sync holds the commit the documentation was last
// verified against. The pre-push hook blocks pushing when source has moved past
// it without a docs update; /sync-docs uses this helper to see the delta and,
// once edits are approved, to advance the marker.

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MARKER = path.join(ROOT, "docs", ".docs-sync");
const WATCH = /^(Backend\/|src\/app\/|launch\.mjs|README|GETTING-STARTED)/;

function git(args) {
  return execSync(`git ${args}`, { cwd: ROOT, encoding: "utf8" }).trim();
}

const head = git("rev-parse HEAD");
const marker = fs.existsSync(MARKER) ? fs.readFileSync(MARKER, "utf8").trim() : "";

if (process.argv.includes("--commit")) {
  fs.writeFileSync(MARKER, head + "\n");
  console.log(`docs marker advanced to ${head.slice(0, 9)}`);
  process.exit(0);
}

if (!marker) {
  console.log("No docs marker found. Treat every doc section as needing review.");
  console.log(`HEAD: ${head}`);
  process.exit(0);
}

let markerValid = true;
try { git(`cat-file -e ${marker}^{commit}`); } catch { markerValid = false; }
if (!markerValid) {
  console.log(`Marker ${marker.slice(0, 9)} is not a known commit (rebased history?).`);
  console.log("Review the docs against current HEAD and re-commit the marker.");
  console.log(`HEAD: ${head}`);
  process.exit(0);
}

if (marker === head) {
  console.log("docs already in sync (marker == HEAD). Nothing to update.");
  process.exit(0);
}

const changed = git(`diff --name-only ${marker} ${head}`).split("\n").filter(Boolean);
const watched = changed.filter((f) => WATCH.test(f));
const commits = git(`log --oneline ${marker}..${head}`);

console.log(`Last synced : ${marker.slice(0, 9)}`);
console.log(`Current HEAD: ${head.slice(0, 9)}`);
console.log("");
console.log(`Commits in range (${commits.split("\n").filter(Boolean).length}):`);
console.log(commits ? commits.replace(/^/gm, "  ") : "  (none)");
console.log("");
if (watched.length) {
  console.log(`Source files changed that likely affect the docs (${watched.length}):`);
  console.log(watched.map((f) => "  " + f).join("\n"));
} else {
  console.log("No watched source files changed. A refactor-only range,");
  console.log("you may advance the marker without editing prose.");
}
