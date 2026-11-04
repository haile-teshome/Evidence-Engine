/** What the saved session leaves out, and how it is put back on load.
 *
 *  A session is persisted as one JSON document, twice: into IndexedDB after every
 *  change and to the backend a moment later. Measured on a real review of ~3,900
 *  records it had grown to 67 MB, and about two thirds of that was either a second
 *  copy of something already in the document or article text the backend already
 *  holds. Each helper below removes one such redundancy on the way out and its
 *  partner restores it on the way in, so nothing the reviewer can see changes. */
import type { FullTextRecord, ScreeningPass } from "./store";
import type { Paper, ScreenResult } from "./mockServices";

/** Bodies with a backend cache key are dropped; the key is enough to get them back.
 *  Records that predate the cache (no key) keep their text so nothing is lost. */
export function compactFullTexts(ft: Record<string, FullTextRecord>): Record<string, FullTextRecord> {
  const out: Record<string, FullTextRecord> = {};
  for (const [id, rec] of Object.entries(ft || {})) {
    if (rec && rec.text_key && rec.text !== undefined) {
      const { text: _omit, ...rest } = rec;
      out[id] = rest;
    } else {
      out[id] = rec;
    }
  }
  return out;
}

export const CURRENT_RESULTS = "current" as const;

/** The newest screening pass keeps a copy of the results table for restore, and
 *  that copy IS the live table (same array). Store a marker instead of the copy. */
export function compactArchive(archive: ScreeningPass[], results: ScreenResult[] | null): ScreeningPass[] {
  return (archive || []).map(p =>
    results && p.results === results ? { ...p, results: [], resultsRef: CURRENT_RESULTS } : p,
  );
}

/** Reinstate the marker as the very array `results` hydrates to, so the run-history
 *  panel's identity check (pass.results === s.results) works after a reload too. */
export function expandArchive(
  archive: ScreeningPass[] | null | undefined,
  results: ScreenResult[] | null | undefined,
): ScreeningPass[] | null | undefined {
  if (!Array.isArray(archive)) return archive;
  return archive.map(p => {
    if (p.resultsRef !== CURRENT_RESULTS) return p;
    const { resultsRef: _omit, ...rest } = p;
    return { ...rest, results: results ?? [] };
  });
}

const ABS_REF = "__absRef";

/** rawPapers (pre-dedup) and uniquePapers (post-dedup) share almost every record,
 *  and the abstract is nearly all of a record's bytes. Where a raw record's abstract
 *  is byte-identical to its unique twin's, drop it and leave a marker. */
export function compactRawPapers(raw: Paper[] | null, unique: Paper[] | null): any[] | null {
  if (!raw || !unique) return raw;
  const abs = new Map<string, string>();
  for (const u of unique) if (u && u.id != null && typeof u.abstract === "string") abs.set(String(u.id), u.abstract);
  return raw.map(p => {
    if (!p || typeof p.abstract !== "string" || !p.abstract) return p;
    if (abs.get(String(p.id)) !== p.abstract) return p;
    const { abstract: _omit, ...rest } = p as any;
    return { ...rest, [ABS_REF]: 1 };
  });
}

export function expandRawPapers(raw: any[] | null | undefined, unique: Paper[] | null | undefined): Paper[] | null | undefined {
  if (!Array.isArray(raw)) return raw as any;
  if (!raw.some(p => p && p[ABS_REF])) return raw as Paper[];
  const abs = new Map<string, string>();
  for (const u of unique || []) if (u && u.id != null && typeof u.abstract === "string") abs.set(String(u.id), u.abstract);
  return raw.map(p => {
    if (!p || !p[ABS_REF]) return p;
    const { [ABS_REF]: _omit, ...rest } = p;
    return { ...rest, abstract: abs.get(String(p.id)) ?? "" };
  });
}
