// Reference-manager interop: serialize the corpus to RIS / BibTeX (which
// Zotero, EndNote, Mendeley and friends all import), and dedupe a fetched set
// against a library the researcher already has. Import parsing reuses the
// RIS/BibTeX parsers in pdfImport.ts.

import type { Paper } from "./apiClient";

// Minimal reference shape the serializers accept (a superset of Paper).
export type Ref = {
  title: string; authors?: string; year?: number;
  url?: string; source?: string; abstract?: string; doi?: string;
};

function doiOf(r: Ref): string {
  if (r.doi) return r.doi.toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, "").trim();
  const m = (r.url || "").match(/10\.\d{4,9}\/[^\s"<>]+/i);
  return m ? m[0].toLowerCase() : "";
}
function normTitle(t: string): string {
  return (t || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

/** Stable identity for dedupe: DOI when present, else normalized title. */
export function refKey(r: Ref): string {
  return doiOf(r) || normTitle(r.title);
}

export function toRis(r: Ref): string {
  const L: string[] = ["TY  - JOUR"];
  L.push(`TI  - ${r.title || ""}`);
  if (r.authors) r.authors.split(/[;]/).forEach(a => { const t = a.trim(); if (t) L.push(`AU  - ${t}`); });
  if (r.year) L.push(`PY  - ${r.year}`);
  const doi = doiOf(r);
  if (doi) L.push(`DO  - ${doi}`);
  if (r.url) L.push(`UR  - ${r.url}`);
  if (r.abstract) L.push(`AB  - ${r.abstract.slice(0, 800)}`);
  if (r.source) L.push(`N1  - Source: ${r.source}`);
  L.push("ER  - ");
  return L.join("\n");
}

export function toBibTeX(r: Ref, idx: number): string {
  const esc = (s?: string) => (s ?? "").replace(/[{}"\\]/g, "");
  const firstAuthor = (r.authors || "").split(/[;,]/)[0]?.trim().split(/\s+/).pop() || "ref";
  const titleWord = (r.title || "").split(/\s+/).find(w => w.length > 3)?.toLowerCase().replace(/\W/g, "") || "paper";
  const key = `${firstAuthor.toLowerCase().replace(/\W/g, "")}${r.year ?? "nd"}_${titleWord}`;
  const L: string[] = [`@article{${key},`];
  L.push(`  title     = {${esc(r.title)}},`);
  if (r.authors) L.push(`  author    = {${esc(r.authors)}},`);
  if (r.year) L.push(`  year      = {${r.year}},`);
  const doi = doiOf(r);
  if (doi) L.push(`  doi       = {${doi}},`);
  if (r.url) L.push(`  url       = {${r.url}},`);
  if (r.abstract) L.push(`  abstract  = {${esc(r.abstract.slice(0, 400))}},`);
  if (r.source) L.push(`  note      = {[${idx}] Source: ${r.source}},`);
  L.push("}");
  return L.join("\n");
}

export function serializeReferences(refs: Ref[], format: "RIS" | "BibTeX"): string {
  if (!refs.length) return "";
  if (format === "RIS") return refs.map(toRis).join("\n\n") + "\n";
  return refs.map((r, i) => toBibTeX(r, i + 1)).join("\n\n") + "\n";
}

/** Flag which candidate papers already exist in a reference library. */
export function dedupeAgainstLibrary<T extends Ref>(
  papers: T[], library: Ref[],
): { annotated: (T & { inLibrary: boolean })[]; matched: number } {
  const keys = new Set(library.map(refKey).filter(Boolean));
  let matched = 0;
  const annotated = papers.map(p => {
    const hit = keys.has(refKey(p));
    if (hit) matched++;
    return { ...p, inLibrary: hit };
  });
  return { annotated, matched };
}

export function paperToRef(p: Paper): Ref {
  return { title: p.title, authors: p.authors, year: p.year, url: p.url, source: p.source, abstract: p.abstract };
}
