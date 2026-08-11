// Import a user's own set of studies from common file formats into the review
// corpus. Each file yields one or more Papers (title + abstract for screening),
// and document formats also cache their full text so the full-text / extraction
// tabs work with no fetch.
//
// Supported:
//   • PDF                         → text via pdf.js            (one study)
//   • Word .docx                  → text via mammoth           (one study)
//   • .txt / .md                  → raw text                   (one study)
//   • .csv / .tsv / .xlsx / .xls  → one study PER ROW          (SheetJS)
//   • .ris / .bib / .nbib         → one study PER reference    (parsed)

import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Paper } from "./apiClient";
import type { FullTextRecord } from "./store";

// All uploaded studies share this source tag so the app can tell them apart from
// database hits (used to switch Home into "work with my studies" mode).
export const UPLOAD_SOURCE = "Local PDFs";

export type ImportedStudy = {
  paper: Paper;
  fullText?: FullTextRecord;   // present when we have the document's full text
  objectBlob?: Blob;           // the original file bytes, for the in-app viewer (the registry owns the object URL + persistence)
  objectMime?: string;         // its MIME type, so the viewer knows if it can render it inline
  html?: string;               // rendered HTML (Word docs), so the preview looks like the real document
};

const CURRENT_YEAR = 2026;

let _seq = 0;
function newId(): string {
  _seq += 1;
  return `up-${Date.now().toString(36)}-${_seq}`;
}

// ── text extraction (single-document formats) ────────────────────────────────
async function extractPdf(file: File): Promise<{ text: string; firstPage: string; metaTitle: string }> {
  const buf = await file.arrayBuffer();
  const pdfjs: any = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
  const doc = await pdfjs.getDocument({ data: buf, isEvalSupported: false }).promise;
  let metaTitle = "";
  try { metaTitle = ((await doc.getMetadata())?.info?.Title || "").toString().trim(); } catch { /* none */ }
  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    pages.push(content.items.map((it: any) => ("str" in it ? it.str : "")).join(" "));
  }
  return { text: pages.join("\n\n"), firstPage: pages[0] || "", metaTitle };
}

async function extractDocx(file: File): Promise<{ text: string; html: string }> {
  const buf = await file.arrayBuffer();
  const mod: any = await import("mammoth/mammoth.browser.js");
  const mammoth = mod.default || mod;
  // convertToHtml preserves headings, bold, lists and tables so the preview can
  // show the document as it looks, not just raw text. extractRawText feeds the
  // downstream text pipeline (screening, extraction, Q&A).
  const [raw, rich] = await Promise.all([
    mammoth.extractRawText({ arrayBuffer: buf }),
    mammoth.convertToHtml({ arrayBuffer: buf }).catch(() => null),
  ]);
  return { text: (raw?.value || "").trim(), html: (rich?.value || "").trim() };
}

// ── heuristic metadata for documents ─────────────────────────────────────────
const GENERIC_TITLE = /^(untitled|microsoft word|document\d*|paper|manuscript|pdf|\d+)\b/i;
const HEADING_STOP = /\b(introduction|background|keywords?|1\s*\.?\s*introduction|methods?|©|copyright|received:|doi:)\b/i;

function cleanFilename(name: string): string {
  return name.replace(/\.[^.]+$/, "").replace(/[_+]+/g, " ").replace(/\s{2,}/g, " ")
    .replace(/\b(final|draft|revised|v\d+|copy)\b/gi, "").trim() || name;
}

function guessTitle(metaTitle: string, firstPage: string, filename: string): string {
  const clean = (t: string) => t.replace(/\s+/g, " ").trim();
  if (metaTitle && metaTitle.length >= 8 && metaTitle.length <= 300 && !GENERIC_TITLE.test(metaTitle)) return clean(metaTitle);
  const lines = firstPage.split(/\n|(?<=\.)\s{2,}/).map(clean).filter(Boolean).slice(0, 12);
  const candidates = lines.filter(l => {
    const words = l.split(/\s+/).length;
    if (words < 4 || words > 30 || l.length < 18 || l.length > 260) return false;
    if (/^(doi|https?:|www\.|received|accepted|volume|vol\.|issn|©|copyright|abstract)\b/i.test(l)) return false;
    if (l === l.toUpperCase() && l.length > 40) return false;
    return true;
  });
  return candidates[0] || cleanFilename(filename);
}

function guessAbstract(text: string): string {
  const m = text.match(/\babstract\b\s*[:.\-\u2014]?\s*/i);
  if (m && m.index !== undefined) {
    const after = text.slice(m.index + m[0].length);
    const stop = after.search(HEADING_STOP);
    const body = (stop > 200 ? after.slice(0, stop) : after.slice(0, 1800)).replace(/\s+/g, " ").trim();
    if (body.length >= 120) return body.slice(0, 2000);
  }
  return text.replace(/\s+/g, " ").trim().slice(0, 1200);
}

function guessYear(text: string): number | undefined {
  const years = [...text.slice(0, 3000).matchAll(/\b(19|20)\d{2}\b/g)]
    .map(x => parseInt(x[0], 10)).filter(y => y >= 1950 && y <= CURRENT_YEAR);
  return years.length ? Math.max(...years) : undefined;
}

function mimeOf(file: File): string {
  if (file.type) return file.type;
  const ext = extOf(file.name);
  if (ext === "pdf") return "application/pdf";
  if (ext === "docx") return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  if (ext === "doc") return "application/msword";
  if (ext === "rtf") return "application/rtf";
  if (ext === "html" || ext === "htm") return "text/html";
  if (ext === "json") return "application/json";
  if (ext === "xml") return "application/xml";
  if (["md", "markdown", "txt", "tex"].includes(ext)) return "text/plain";
  return "application/octet-stream";
}

function studyFromDoc(file: File, text: string, firstPage: string, metaTitle: string, keepFile?: boolean, html?: string): ImportedStudy {
  const id = newId();
  const title = guessTitle(metaTitle, firstPage, file.name);
  const paper: Paper = { id, source: UPLOAD_SOURCE, title, abstract: guessAbstract(text), url: "", year: guessYear(text) };
  const fullText: FullTextRecord = { paper_id: id, title, url: "", source: UPLOAD_SOURCE, status: "found", text };
  return { paper, fullText, objectBlob: keepFile ? file : undefined, objectMime: keepFile ? mimeOf(file) : undefined, html: html || undefined };
}

// ── tabular formats (one study per row) ──────────────────────────────────────
function parseYearStr(v: string): number | undefined {
  const m = (v || "").match(/(19|20)\d{2}/);
  const y = m ? parseInt(m[0], 10) : NaN;
  return y >= 1950 && y <= CURRENT_YEAR ? y : undefined;
}

function pickCol(keys: string[], patterns: RegExp[]): string | null {
  for (const pat of patterns) { const k = keys.find(k => pat.test(k)); if (k) return k; }
  return null;
}

async function importTabular(file: File): Promise<ImportedStudy[]> {
  const XLSX: any = await import("xlsx");
  const buf = await file.arrayBuffer();
  const wb = XLSX.read(buf, { type: "array" });
  const ws = wb.Sheets[wb.SheetNames[0]];
  const rows: Record<string, any>[] = XLSX.utils.sheet_to_json(ws, { defval: "", raw: false });
  if (!rows.length) return [];

  const keys = Object.keys(rows[0]).map(k => k.trim());
  const col = {
    title: pickCol(keys, [/^title$/i, /article.*title|paper.*title|study.*title|primary.*title/i, /\btitle\b/i]),
    abstract: pickCol(keys, [/^abstract$/i, /abstract|summary/i]),
    authors: pickCol(keys, [/^authors?$/i, /author|creator/i]),
    year: pickCol(keys, [/^year$/i, /year|publication.*date|pub.*year|^date$|published/i]),
    doi: pickCol(keys, [/^doi$/i, /\bdoi\b/i]),
    url: pickCol(keys, [/^url$/i, /url|link|fulltext/i]),
    fulltext: pickCol(keys, [/full.?text|full.?text|body|content|notes/i]),
  };

  const out: ImportedStudy[] = [];
  rows.forEach((raw, i) => {
    const r: Record<string, string> = {};
    for (const k of Object.keys(raw)) r[k.trim()] = String(raw[k] ?? "").trim();
    const firstNonEmpty = keys.map(k => r[k]).find(v => v && v.length > 2) || "";
    const title = (col.title && r[col.title]) || firstNonEmpty || `Study ${i + 1}`;
    if (!title.trim()) return;
    const doi = col.doi ? r[col.doi] : "";
    const id = newId();
    const paper: Paper = {
      id, source: UPLOAD_SOURCE, title,
      abstract: (col.abstract && r[col.abstract]) || "",
      url: (col.url && r[col.url]) || (doi ? `https://doi.org/${doi.replace(/^https?:\/\/doi\.org\//i, "")}` : ""),
      year: col.year ? parseYearStr(r[col.year]) : undefined,
      authors: col.authors ? r[col.authors] : undefined,
    };
    const ftText = col.fulltext ? r[col.fulltext] : "";
    const study: ImportedStudy = { paper };
    if (ftText && ftText.length > 200) {
      study.fullText = { paper_id: id, title, url: paper.url, source: UPLOAD_SOURCE, status: "found", text: ftText };
    }
    out.push(study);
  });
  return out;
}

// ── reference-manager exports (one study per record) ─────────────────────────
function studyFromRef(f: { title: string; abstract?: string; authors?: string; year?: number; doi?: string; url?: string }): ImportedStudy | null {
  const title = (f.title || "").trim();
  if (!title) return null;
  const id = newId();
  const paper: Paper = {
    id, source: UPLOAD_SOURCE, title,
    abstract: (f.abstract || "").trim(),
    url: f.url || (f.doi ? `https://doi.org/${f.doi.replace(/^https?:\/\/doi\.org\//i, "")}` : ""),
    year: f.year, authors: f.authors,
  };
  return { paper };
}

function parseRIS(text: string): ImportedStudy[] {   // .ris and .nbib (PubMed) share the TAG  - value form
  const out: ImportedStudy[] = [];
  let cur: any = null;
  const flush = () => { if (cur) { const st = studyFromRef(cur); if (st) out.push(st); } cur = null; };
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^([A-Z0-9]{2,4})\s*-\s?(.*)$/);
    if (!m) { if (cur && cur._last) cur[cur._last] = (cur[cur._last] || "") + " " + line.trim(); continue; }
    const tag = m[1], val = m[2].trim();
    if (tag === "TY" || (tag === "PMID" && cur)) { if (cur) flush(); cur = {}; }
    if (!cur) cur = {};
    const set = (field: string) => { cur[field] = cur[field] ? cur[field] + (field === "authors" ? "; " : " ") + val : val; cur._last = field; };
    switch (tag) {
      case "TI": case "T1": case "CT": set("title"); break;
      case "AB": set("abstract"); break;
      case "AU": case "A1": case "FAU": set("authors"); break;
      case "PY": case "Y1": case "DP": case "DA": if (!cur.year) cur.year = parseYearStr(val); break;
      case "DO": case "LID": case "AID": if (!cur.doi && /10\.\d{4}/.test(val)) cur.doi = (val.match(/10\.\S+/) || [])[0]; break;
      case "UR": case "L1": if (!cur.url) cur.url = val; break;
      case "ER": flush(); break;
    }
  }
  flush();
  return out;
}

function parseBibTeX(text: string): ImportedStudy[] {
  const out: ImportedStudy[] = [];
  const entries = text.split(/@\w+\s*\{/).slice(1);
  const field = (body: string, name: string) => {
    const re = new RegExp(name + "\\s*=\\s*[{\"]([\\s\\S]*?)[}\"]\\s*,?\\s*(?:\\n|$)", "i");
    const m = body.match(re);
    return m ? m[1].replace(/[{}]/g, "").replace(/\s+/g, " ").trim() : "";
  };
  for (const e of entries) {
    const st = studyFromRef({
      title: field(e, "title"), abstract: field(e, "abstract"), authors: field(e, "author"),
      year: parseYearStr(field(e, "year") || field(e, "date")), doi: field(e, "doi"), url: field(e, "url"),
    });
    if (st) out.push(st);
  }
  return out;
}

// ── plain-text extraction for lighter formats ────────────────────────────────
function stripHtml(html: string): string {
  const noBlocks = html.replace(/<(script|style|head)[\s\S]*?<\/\1>/gi, " ");
  const withBreaks = noBlocks
    .replace(/<\/(p|div|h[1-6]|li|tr|br|section|article)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n");
  const text = withBreaks.replace(/<[^>]+>/g, " ");
  const decoded = text
    .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">").replace(/&quot;/gi, '"').replace(/&#39;/gi, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
  return decoded.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

// Minimal RTF → text: drop control words, hex escapes, and group braces.
function stripRtf(rtf: string): string {
  return rtf
    .replace(/\\'[0-9a-fA-F]{2}/g, "")
    .replace(/\\par[d]?\b/g, "\n")
    .replace(/\\tab\b/g, "\t")
    .replace(/\\[a-zA-Z]+-?\d* ?/g, "")
    .replace(/[{}]/g, "")
    .replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

// EndNote .enw tagged export (%T title, %A author, %D year, %X abstract, %U url).
function parseEnw(text: string): ImportedStudy[] {
  const out: ImportedStudy[] = [];
  let cur: any = null;
  const flush = () => { if (cur) { const st = studyFromRef(cur); if (st) out.push(st); } cur = null; };
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^%([A-Z0-9])\s(.*)$/);
    if (!m) continue;
    if (m[1] === "0" || (m[1] === "T" && cur?.title)) flush();
    if (!cur) cur = {};
    const val = m[2].trim();
    switch (m[1]) {
      case "T": cur.title = cur.title ? cur.title + " " + val : val; break;
      case "A": cur.authors = cur.authors ? cur.authors + "; " + val : val; break;
      case "X": case "N": cur.abstract = cur.abstract ? cur.abstract + " " + val : val; break;
      case "D": if (!cur.year) cur.year = parseYearStr(val); break;
      case "R": if (!cur.doi && /10\.\d{4}/.test(val)) cur.doi = (val.match(/10\.\S+/) || [])[0]; break;
      case "U": if (!cur.url) cur.url = val; break;
    }
  }
  flush();
  return out;
}

// ── dispatch ─────────────────────────────────────────────────────────────────
function extOf(name: string): string {
  const m = name.toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : "";
}

// Formats we can extract usable text from. Spreadsheets (incl. OpenDocument .ods)
// go through the xlsx reader; references through their taggers; everything else
// is read as text, with HTML/RTF cleaned up. Any file whose MIME is text/* is
// also accepted, so uncommon plain-text extensions still work.
export const ACCEPTED_EXTS = [
  "pdf", "docx", "doc", "rtf",
  "txt", "md", "markdown", "tex", "json", "xml", "html", "htm",
  "csv", "tsv", "xlsx", "xls", "xlsm", "ods",
  "ris", "bib", "nbib", "enw",
];

export function isAccepted(file: File): boolean {
  return ACCEPTED_EXTS.includes(extOf(file.name)) || file.type === "application/pdf" || file.type.startsWith("text/");
}

async function importOne(file: File): Promise<ImportedStudy[]> {
  const ext = extOf(file.name);
  // For file formats we keep the original of (PDF, Word, RTF, HTML, text), text
  // extraction is best-effort: a scanned/image-only PDF, or any file we can't pull
  // text from, is still imported so it can be viewed as the actual document. Text
  // powers Q&A / screening when available; its absence no longer rejects the file.
  if (ext === "pdf" || file.type === "application/pdf") {
    let text = "", firstPage = "", metaTitle = "";
    try { ({ text, firstPage, metaTitle } = await extractPdf(file)); } catch { /* image-only PDF → keep it anyway */ }
    return [studyFromDoc(file, text, firstPage, metaTitle, true)];
  }
  if (ext === "docx" || ext === "doc") {
    // mammoth handles .docx; .doc (legacy binary) throws and is reported as failed.
    const { text, html } = await extractDocx(file);
    return [studyFromDoc(file, text, text.slice(0, 3000), "", true, html)];
  }
  if (["csv", "tsv", "xlsx", "xls", "xlsm", "ods"].includes(ext)) return importTabular(file);
  if (ext === "ris" || ext === "nbib") return parseRIS(await file.text());
  if (ext === "bib") return parseBibTeX(await file.text());
  if (ext === "enw") return parseEnw(await file.text());
  if (ext === "rtf") {
    const text = stripRtf(await file.text());
    return [studyFromDoc(file, text, text.slice(0, 3000), "", true)];
  }
  if (ext === "html" || ext === "htm" || file.type === "text/html") {
    const text = stripHtml(await file.text());
    return [studyFromDoc(file, text, text.slice(0, 3000), "", true)];
  }
  // txt / md / tex / json / xml / other text
  const text = await file.text();
  return [studyFromDoc(file, text, text.slice(0, 3000), "", true)];
}

// Process a batch with progress. Per-file failures are collected, not thrown, so
// one bad file doesn't abort the whole upload.
export async function importStudies(
  files: File[],
  onProgress?: (done: number, total: number, name: string) => void,
): Promise<{ studies: ImportedStudy[]; failed: { name: string; error: string }[] }> {
  const usable = files.filter(isAccepted);
  const studies: ImportedStudy[] = [];
  const failed: { name: string; error: string }[] = [];
  for (let i = 0; i < usable.length; i++) {
    onProgress?.(i, usable.length, usable[i].name);
    try {
      const got = await importOne(usable[i]);
      if (got.length) studies.push(...got);
      else failed.push({ name: usable[i].name, error: "no studies found" });
    } catch (e: any) {
      failed.push({ name: usable[i].name, error: e?.message || "could not parse" });
    }
  }
  onProgress?.(usable.length, usable.length, "");
  return { studies, failed };
}
