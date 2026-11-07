// Hand a reviewer's own PDF to the backend.
//
// An imported document is parsed in the browser: the text goes into the store
// and the bytes into IndexedDB, and nothing ever reaches the server. That is
// fine until a stage needs the FILE rather than the text. Table extraction is
// the clearest case: its strongest readers (the publisher's structured XML
// aside) run over the PDF itself, and they were being skipped for exactly the
// papers a reviewer supplied, because the server had no file to open.
//
// So the bytes are handed over on import, and again on demand for papers
// imported before this existed. The server keeps them in the same cache that
// full-text acquisition writes, keyed by paper id.
import { getPdfFile } from "./pdfBlobs";
import { AIService } from "./apiClient";

// One handover per paper per session. The cache on the other end is keyed by
// paper id, so a second upload would only rewrite the same bytes.
const sent = new Set<string>();

/** True if the file is now on the server (or already was). Never throws. */
export async function sendPdfToServer(paperId: string): Promise<boolean> {
  if (!paperId) return false;
  if (sent.has(paperId)) return true;
  try {
    const file = await getPdfFile(paperId);
    if (!file) return false;
    await AIService.uploadPdf(paperId, file);
    sent.add(paperId);
    return true;
  } catch {
    // Not fatal: the ladder has other rungs, and this one simply does not run.
    return false;
  }
}

/** Hand over several, one at a time so a folder of PDFs cannot flood the API. */
export async function sendPdfsToServer(paperIds: string[]): Promise<number> {
  let done = 0;
  for (const id of paperIds) {
    if (await sendPdfToServer(id)) done++;
  }
  return done;
}

/**
 * Read a document the browser could not: a scanned PDF has no text layer, and
 * the server can render and OCR it. Returns the text, or "" when that failed.
 */
export async function readTextOnServer(
  paperId: string,
): Promise<{ text: string; method: string }> {
  // "unsent" and "none" are different failures: one never reached the reader,
  // the other reached it and came back empty, and only the second one says
  // anything about the document.
  if (!(await sendPdfToServer(paperId))) return { text: "", method: "unsent" };
  try {
    const { text, method } = await AIService.readPdfText(paperId);
    return { text: text || "", method: method || "none" };
  } catch {
    return { text: "", method: "unreachable" };
  }
}

/** Test seam: forget what has been sent. */
export function _resetPdfHandoff() {
  sent.clear();
}
