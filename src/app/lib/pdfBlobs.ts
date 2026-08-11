// Registry of uploaded-document previews, keyed by paper id.
//   - url:  object URL of the original file (PDF renders inline; others download).
//   - mime: its MIME type, so the viewer knows whether it can render it inline.
//   - html: rendered HTML for formats the browser can't show inline (Word), so
//           the preview looks like the actual document, not raw text.
//
// The live registry holds session object URLs (which can't be serialized). The
// underlying bytes/HTML are ALSO persisted to IndexedDB, keyed per document with
// a small index list, so a page reload can rebuild the previews. The study corpus
// itself (rawPapers / fullTexts) is persisted by the store under the same paper
// ids, so the keys line up after a refresh.

import { idbGet, idbSet, idbDel } from "./idb";

const INDEX_KEY = "doc-files-index";
const fileKey = (id: string) => `docfile:${id}`;

type BlobEntry = { url?: string; mime?: string; html?: string };
type StoredEntry = { blob?: Blob; mime?: string; html?: string };

const blobs: Record<string, BlobEntry> = {};

// Serialize IndexedDB writes so concurrent uploads don't clobber the index.
let writeChain: Promise<void> = Promise.resolve();
function enqueue(op: () => Promise<void>) {
  writeChain = writeChain.then(op).catch(() => { /* persistence is best-effort */ });
  return writeChain;
}

async function addToIndex(id: string) {
  const idx = (await idbGet<string[]>(INDEX_KEY).catch(() => undefined)) || [];
  if (!idx.includes(id)) { idx.push(id); await idbSet(INDEX_KEY, idx); }
}
async function removeFromIndex(id: string) {
  const idx = (await idbGet<string[]>(INDEX_KEY).catch(() => undefined)) || [];
  const next = idx.filter(x => x !== id);
  if (next.length !== idx.length) await idbSet(INDEX_KEY, next);
}

export function registerPdfBlobs(entries: { id: string; blob?: Blob; mime?: string; html?: string }[]) {
  for (const e of entries) {
    if (!e.blob && !e.html) continue;
    const prev = blobs[e.id];
    let url = prev?.url;
    if (e.blob) {
      if (url) { try { URL.revokeObjectURL(url); } catch { /* ignore */ } }
      url = URL.createObjectURL(e.blob);
    }
    blobs[e.id] = { url, mime: e.mime ?? prev?.mime, html: e.html ?? prev?.html };
    // Persist the bytes/HTML (not the object URL) so the preview survives reloads.
    const stored: StoredEntry = { blob: e.blob, mime: e.mime, html: e.html };
    enqueue(async () => { await idbSet(fileKey(e.id), stored); await addToIndex(e.id); });
  }
}

export function removePdfBlob(id: string) {
  const prev = blobs[id];
  if (prev?.url) { try { URL.revokeObjectURL(prev.url); } catch { /* ignore */ } }
  delete blobs[id];
  enqueue(async () => { await idbDel(fileKey(id)); await removeFromIndex(id); });
}

export function getPdfBlob(id: string): string | undefined {
  return blobs[id]?.url;
}

export function getPdfBlobMime(id: string): string | undefined {
  return blobs[id]?.mime;
}

export function getDocHtml(id: string): string | undefined {
  return blobs[id]?.html;
}

export function hasPdfBlob(id: string): boolean {
  return Boolean(blobs[id]);
}

// Rebuild the in-memory registry from IndexedDB after a reload, recreating a
// fresh object URL for each stored file. Runs once at module load; consumers can
// await `docFilesReady` if they need the previews before first render.
async function hydrateDocFiles(): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  try {
    const idx = (await idbGet<string[]>(INDEX_KEY).catch(() => undefined)) || [];
    await Promise.all(idx.map(async id => {
      if (blobs[id]) return;                                   // a fresh upload already registered it
      const stored = await idbGet<StoredEntry>(fileKey(id)).catch(() => undefined);
      if (!stored) return;
      const url = stored.blob ? URL.createObjectURL(stored.blob) : undefined;
      blobs[id] = { url, mime: stored.mime, html: stored.html };
    }));
  } catch { /* persistence unavailable → previews just won't survive reload */ }
}

export const docFilesReady: Promise<void> = hydrateDocFiles();
