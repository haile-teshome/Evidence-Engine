// In-memory registry of uploaded-PDF blob URLs, keyed by paper id. Object URLs
// can't be serialized/persisted, so they live only for the current session —
// enough for the in-app PDF viewer to render a study the user just uploaded.
// Downstream tabs read text from `fullTexts`; this is only for the PDF preview.

const blobs: Record<string, string> = {};

export function registerPdfBlobs(entries: { id: string; url: string }[]) {
  for (const e of entries) {
    if (!e.url) continue;
    if (blobs[e.id] && blobs[e.id] !== e.url) {
      try { URL.revokeObjectURL(blobs[e.id]); } catch { /* ignore */ }
    }
    blobs[e.id] = e.url;
  }
}

export function getPdfBlob(id: string): string | undefined {
  return blobs[id];
}

export function hasPdfBlob(id: string): boolean {
  return Boolean(blobs[id]);
}
