#!/usr/bin/env node
// Build the multi-page documentation site from the authored single-page source.
//
//   node scripts/build-docs.mjs
//
// Source of truth: docs/_source.html (one big page, easy to edit and to diff).
// This script splits it into one page per <section>, sharing a sidebar, CSS,
// and behaviour, and emits:
//   docs/index.html            the Introduction page (entry point)
//   docs/<section-id>.html     one page per remaining section
//   docs/assets/docs.css       extracted styles + a few multi-page additions
//   docs/assets/docs.js        theme toggle + nav filter
//
// The /sync-docs workflow edits docs/_source.html, then reruns this script.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = path.join(ROOT, "docs");
const SRC = path.join(DOCS, "_source.html");

const html = fs.readFileSync(SRC, "utf8");

// ---- extract the stylesheet ---------------------------------------------
const styleMatch = html.match(/<style>([\s\S]*?)<\/style>/);
if (!styleMatch) throw new Error("no <style> block found in _source.html");
const css = styleMatch[1];

// ---- extract the shared icon sprite (injected into every page) ----------
const spriteMatch = html.match(/<!--ICON-SPRITE-START-->([\s\S]*?)<!--ICON-SPRITE-END-->/);
const sprite = spriteMatch ? spriteMatch[1].trim() : "";

// ---- extract the sidebar nav structure ----------------------------------
const navMatch = html.match(/<nav id="nav">([\s\S]*?)<\/nav>/);
if (!navMatch) throw new Error("no <nav id=\"nav\"> found in _source.html");
const navInner = navMatch[1];

// Ordered list of { kind: 'group', label } | { kind: 'link', id, label }.
const navItems = [];
const navRe = /<div class="group">([\s\S]*?)<\/div>|<a href="#([^"]+)">([\s\S]*?)<\/a>/g;
let m;
while ((m = navRe.exec(navInner)) !== null) {
  if (m[1] !== undefined) navItems.push({ kind: "group", label: m[1].trim() });
  else navItems.push({ kind: "link", id: m[2], label: m[3].trim() });
}
const links = navItems.filter((i) => i.kind === "link");
const linkLabel = Object.fromEntries(links.map((l) => [l.id, l.label]));

// ---- extract sections ----------------------------------------------------
const sections = [];
const secRe = /<section id="([^"]+)">([\s\S]*?)<\/section>/g;
while ((m = secRe.exec(html)) !== null) sections.push({ id: m[1], body: m[2] });
const sectionIds = new Set(sections.map((s) => s.id));

// Page slug: the introduction becomes the site entry (index.html).
const slug = (id) => (id === "introduction" ? "index" : id);

// Rewrite in-document anchor links (#id) to the corresponding page file.
function rewriteLinks(s) {
  return s.replace(/href="#([a-z0-9-]+)"/g, (whole, id) =>
    sectionIds.has(id) ? `href="${slug(id)}.html"` : whole,
  );
}

// ---- shared pieces -------------------------------------------------------
function sidebar(currentId) {
  const rows = navItems
    .map((i) => {
      if (i.kind === "group") return `      <div class="group">${i.label}</div>`;
      const cls = i.id === currentId ? ' class="active"' : "";
      return `      <a href="${slug(i.id)}.html"${cls}>${i.label}</a>`;
    })
    .join("\n");
  return `  <aside>
    <a class="brand" href="index.html" style="text-decoration:none">
      <div class="logo">EE</div>
      <div><b>Evidence Engine</b><span>Developer documentation</span></div>
    </a>
    <input class="search" id="search" placeholder="Filter sections…" aria-label="Filter sections" />
    <nav id="nav">
${rows}
    </nav>
  </aside>`;
}

function prevNext(id) {
  const idx = links.findIndex((l) => l.id === id);
  const prev = idx > 0 ? links[idx - 1] : null;
  const next = idx >= 0 && idx < links.length - 1 ? links[idx + 1] : null;
  if (!prev && !next) return "";
  const left = prev ? `<a class="prv" href="${slug(prev.id)}.html">← ${prev.label}</a>` : "<span></span>";
  const right = next ? `<a class="nxt" href="${slug(next.id)}.html">${next.label} →</a>` : "<span></span>";
  return `\n      <div class="prevnext">${left}${right}</div>`;
}

function page(section) {
  const title = linkLabel[section.id] || "Documentation";
  const body = rewriteLinks(section.body).trimEnd();
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${title.replace(/&amp;/g, "&")} · Evidence Engine Docs</title>
<link rel="stylesheet" href="assets/docs.css" />
</head>
<body>
${sprite}
<div class="layout">
${sidebar(section.id)}
  <main>
    <div class="topbar">
      <span class="pill" id="version-pill">v0.0.1</span>
      <button class="theme-btn" id="theme-btn">Toggle theme</button>
    </div>
    <section id="${section.id}">${body}</section>${prevNext(section.id)}
    <footer>
      Evidence Engine developer documentation.
      <a href="index.html">Home</a> · single-page version in <code>docs/_source.html</code>.
    </footer>
  </main>
</div>
<script src="assets/docs.js"></script>
</body>
</html>
`;
}

// ---- extra CSS for the multi-page layout --------------------------------
const extraCss = `
  /* multi-page additions */
  a.brand:hover { text-decoration: none; }
  .prevnext { display: flex; justify-content: space-between; gap: 14px; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }
  .prevnext a { max-width: 48%; font-size: 14px; font-weight: 600; }
  .prevnext .nxt { margin-left: auto; text-align: right; }
`;

const docsJs = `// Theme toggle + sidebar filter for the multi-page docs.
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme-btn');
  if (btn) btn.addEventListener('click', function () {
    var cur = root.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    root.setAttribute('data-theme', next);
  });
  var input = document.getElementById('search');
  if (input) {
    var links = Array.prototype.slice.call(document.querySelectorAll('#nav a'));
    input.addEventListener('input', function () {
      var q = input.value.toLowerCase().trim();
      links.forEach(function (a) {
        var hit = !q || a.textContent.toLowerCase().indexOf(q) !== -1;
        a.classList.toggle('hide', !hit);
      });
      document.querySelectorAll('#nav .group').forEach(function (g) {
        g.classList.toggle('hide', !!q);
      });
    });
  }
})();
`;

// ---- write outputs -------------------------------------------------------
fs.mkdirSync(path.join(DOCS, "assets"), { recursive: true });
fs.writeFileSync(path.join(DOCS, "assets", "docs.css"), css + extraCss);
fs.writeFileSync(path.join(DOCS, "assets", "docs.js"), docsJs);

let count = 0;
for (const s of sections) {
  fs.writeFileSync(path.join(DOCS, `${slug(s.id)}.html`), page(s));
  count++;
}

console.log(`Built ${count} pages + assets from docs/_source.html`);
console.log(`Entry: docs/index.html (${sections[0].id})`);
