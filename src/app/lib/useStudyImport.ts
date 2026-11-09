import { useState } from "react";
import { useStore } from "./store";
import { importStudies, isAccepted, UPLOAD_SOURCE, ImportedStudy, guessTitle } from "./pdfImport";
import { registerPdfBlobs, removePdfBlob } from "./pdfBlobs";
import { sendPdfsToServer, readTextOnServer } from "./pdfHandoff";
import { AIService, apiConfig } from "./apiClient";
import { apiFetch } from "./backendClient";
import type { Paper, ScreenResult, FullTextResult } from "./apiClient";
import { toast } from "sonner";
import {
  ImportStage, importReason, needsAbstractRow, needsFullTextRow, readsFullText,
} from "./importedAt";

// Shared logic for importing a user's own studies as the corpus. Used by the
// chat bar's "+" attach button and by the import control on every stage page.
// Keeps the store wiring in one place.
//
// `stage` is where the reviewer is importing from. Anything past abstract
// screening also seeds the earlier rows, because each stage reads the previous
// stage's output and would otherwise show nothing. Those seeded rows are marked
// as imported so PRISMA and the AI decision log stay truthful: see importedAt.ts.
/** A document after it has been filed: where it landed, and what is known. */
type LoadedDoc = {
  id: string;
  title: string;
  hasText: boolean;
  titleFromFilename?: boolean;
  parseError?: string;
};

/**
 * The queries in a pasted list.
 *
 * One per line is what the box asks for, and a line that reads as a title is
 * taken whole, because a title has commas in it. But people paste a row of
 * DOIs separated by commas, and that went to the server as a single long
 * query that matched nothing: five real DOIs came back as one "not found".
 * So a line carrying identifiers is read as the list of identifiers it is.
 */
const DOI_IN_LINE = /10\.\d{4,9}\/[^\s,;"'<>]+/g;
const IDS_ONLY = /^[\d\s,;]+$/;

export function listQueries(text: string): string[] {
  const out: string[] = [];
  for (const raw of (text || "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const dois = line.match(DOI_IN_LINE);
    // Trailing punctuation belongs to the sentence, not to the DOI.
    if (dois) { out.push(...dois.map(d => d.replace(/[.,;]+$/, ""))); continue; }
    if (IDS_ONLY.test(line)) { out.push(...line.split(/[\s,;]+/).filter(Boolean)); continue; }
    out.push(line);
  }
  // The same identifier twice is one lookup, not two.
  return [...new Set(out)];
}

export function useStudyImport(stage: ImportStage = "home") {
  const s = useStore();
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number; name: string } | null>(null);

  /** Load documents into the corpus and say, per document, which paper id they
   *  actually landed under and whether anything can be read from them. The
   *  caller needs both: a re-upload lands under the id the corpus already has,
   *  so anything keyed on the fresh parse's id addresses a paper that does not
   *  exist. */
  function loadIntoCorpus(studies: ImportedStudy[]): LoadedDoc[] {
    if (!studies.length) return [];
    // A re-upload of a document already in the corpus (matched by title) must map
    // onto the EXISTING paper id, so its file, rendered HTML, and refreshed text
    // attach to the id the rest of the app references. Otherwise the corpus keeps
    // the old id while the preview data lands under a throwaway new id.
    const existing = (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE);
    const idByTitle = new Map(existing.map(p => [p.title.toLowerCase().trim(), p.id]));

    const freshPapers: Paper[] = [];
    const ftMap: Record<string, NonNullable<ImportedStudy["fullText"]>> = {};
    const blobEntries: { id: string; blob?: Blob; mime?: string; html?: string }[] = [];

    const loaded: LoadedDoc[] = [];

    for (const x of studies) {
      const key = x.paper.title.toLowerCase().trim();
      const existingId = idByTitle.get(key);
      const id = existingId || x.paper.id;
      if (!existingId) { freshPapers.push(x.paper); idByTitle.set(key, id); }
      const fresh = (x.fullText?.text || "").trim();
      const held = (s.fullTexts[id]?.text || "").trim();
      // Never replace text we already have with none: the same paper can arrive
      // a second time as a scan, and that is not a reason to forget its words.
      if (x.fullText && (fresh || !held)) ftMap[id] = { ...x.fullText, paper_id: id };
      if (x.objectBlob || x.html) blobEntries.push({ id, blob: x.objectBlob, mime: x.objectMime, html: x.html });
      loaded.push({
        id, title: x.paper.title, hasText: Boolean(fresh || held),
        titleFromFilename: Boolean(x.titleFromFilename && !existingId),
        parseError: x.parseError || "",
      });
    }

    registerPdfBlobs(blobEntries);
    // Hand the files to the backend in the background: the table readers work
    // on the PDF, not on the text the browser parsed out of it.
    void sendPdfsToServer(blobEntries.filter(b => b.blob).map(b => b.id));
    seedEarlierStages(freshPapers);
    // Preserve papers pulled from database searches; only the UPLOAD_SOURCE slice is
    // rebuilt here. Attaching a PDF must ADD to the corpus, never replace the searched
    // papers (replacing silently dropped every pulled result from the library).
    const nonUploads = (s.rawPapers || []).filter(p => p.source !== UPLOAD_SOURCE);
    s.setRawPapers([...nonUploads, ...existing, ...freshPapers]);
    s.setUniquePapers([...(s.uniquePapers || []), ...freshPapers]);
    s.setFullTexts(prev => ({ ...prev, ...ftMap }));
    if (!s.sources.includes("Local PDFs")) s.setSources([...s.sources, "Local PDFs"]);
    return loaded;
  }

  // Make imported records visible at the stage they were dropped into, without
  // claiming they were screened here.
  function seedEarlierStages(papers: Paper[]) {
    if (!papers.length || !needsAbstractRow(stage)) return;
    const reason = importReason(stage);

    const already = new Set((s.results || []).map(r => r.paper_id));
    const rows: ScreenResult[] = papers
      .filter(p => !already.has(p.id))
      .map(p => ({
        paper_id: p.id,
        Source: p.source, Title: p.title, URL: p.url || "", Abstract: p.abstract || "",
        Decision: "INCLUDE" as const,
        Reason: reason,
        Agent_Trace: {},   // no votes: nothing judged this record here
        Bucket: "Imported",
        Imported: stage,
        doi: p.doi,
      }));
    if (rows.length) s.setResults([...(s.results || []), ...rows]);

    if (!needsFullTextRow(stage)) return;
    const haveFt = new Set((s.fullTextResults || []).map(r => r.paper_id));
    const ftRows: FullTextResult[] = papers
      .filter(p => !haveFt.has(p.id))
      .map(p => ({
        paper_id: p.id,
        Title: p.title, URL: p.url || "", Source: p.source, Abstract: p.abstract || "",
        Decision: "Include" as const,
        Reason: reason,
        criteriaEval: {}, criteriaEvidence: {},
        inclusion_score: 0, exclusion_violations: 0,
        Imported: stage,
        doi: p.doi,
      }));
    if (ftRows.length) s.setFullTextResults([...(s.fullTextResults || []), ...ftRows]);
  }

  /**
   * Fetch the article itself for records that arrived as records.
   *
   * A DOI lookup returns a title and an abstract. The extraction, appraisal and
   * writing stages read the full text, so a study imported there used to land
   * in the table with nothing in it and a message pointing at another stage.
   * Importing at a stage says the reviewer wants to work on it there, so the
   * same fetch Full-Text Acquisition runs happens here, in the background.
   *
   * One at a time, because this reaches publishers and OA indexes that rate
   * limit, and the paper the reviewer is looking at arrives first either way.
   */
  async function acquireFullTexts(papers: Paper[]) {
    const queue = papers.filter(p => !(s.fullTexts[p.id]?.text || "").trim());
    if (!queue.length) return;
    const what = `${queue.length} stud${queue.length === 1 ? "y" : "ies"}`;
    const tid = toast.loading(`Fetching the full text for ${what}\u2026`);
    let found = 0;
    for (let i = 0; i < queue.length; i++) {
      const paper = queue[i];
      try {
        const res = await AIService.fetchFullText({
          Title: paper.title, URL: paper.url || "", Source: paper.source, paper_id: paper.id,
          doi: paper.doi,
        });
        s.setFullTexts(prev => ({
          ...prev,
          [paper.id]: {
            paper_id: paper.id, title: paper.title, url: paper.url || "", source: paper.source,
            status: res.status, text: res.text, text_key: res.text_key, reason: res.reason,
            retrieved_via: res.source,
            reason_code: res.reason_code, oa_status: res.oa_status,
            doi: res.doi, pmid: res.pmid, links: res.links,
            pdf_url: res.pdf_key
              ? `${apiConfig.baseUrl}/fulltext/pdf/${encodeURIComponent(res.pdf_key)}`
              : undefined,
          },
        }));
        if (res.status === "found") found++;
      } catch (e: any) {
        // One paper behind a publisher wall is not a reason to stop fetching
        // the rest, and the miss is already recorded against that paper.
        console.error(`fulltext-fetch ${paper.id} failed:`, e?.message);
      }
      toast.loading(`Fetching the full text for ${what}\u2026 ${i + 1} of ${queue.length}`, { id: tid });
    }
    // Named honestly: a paywalled article is a miss the reviewer has to resolve
    // with their library, not a failure of the import.
    if (found === queue.length) {
      toast.success(`Got the full text for all ${queue.length}`, { id: tid });
    } else if (found) {
      toast.warning(`Got ${found} of ${queue.length} full texts. The rest need your library, `
                  + `on Full-Text Acquisition.`, { id: tid, duration: 9000 });
    } else {
      toast.warning(`No full text was openly available. Try Full-Text Acquisition, `
                  + `which can use your library's access.`, { id: tid, duration: 9000 });
    }
  }

  /** Put a better title on a study once the document has actually been read.
   *  A file whose text the browser could not reach is named after its file, and
   *  that name then follows it through every stage. */
  function renameStudy(id: string, title: string) {
    // Functional updates throughout: this runs after a round trip to the
    // reader, by which time the captured state is several imports out of date,
    // and writing a stale array back would erase the papers it is renaming.
    const rename = <T extends { id: string }>(rows: T[] | null) =>
      (rows ? rows.map(p => (p.id === id ? { ...p, title } : p)) : rows);
    const retitle = <T extends { paper_id: string; Title: string }>(rows: T[] | null) =>
      (rows ? rows.map(r => (r.paper_id === id ? { ...r, Title: title } : r)) : rows);

    s.setRawPapers(rename);
    s.setUniquePapers(rename);
    s.setResults(retitle);
    s.setFullTextResults(retitle);
    s.setFullTexts(prev => (prev[id] ? { ...prev, [id]: { ...prev[id], title } } : prev));
  }

  /** Why a read came back empty, in the reviewer's terms rather than ours. */
  function reasonFor(methods: Set<string>, errors: Set<string> = new Set()): string {
    const why = [...errors].filter(Boolean)[0];
    if (why) return `The browser could not read the file: ${why}`;
    if (methods.has("unsent")) return "The file never reached the reader on this machine.";
    if (methods.has("unreachable")) return "The reader is not responding; is the backend running?";
    return "The pages were read as images too and no text came back.";
  }

  /** Read scanned documents on the server, one at a time, and file the text. */
  async function recoverText(docs: LoadedDoc[]) {
    const what = `${docs.length} document${docs.length === 1 ? "" : "s"}`;
    const tid = toast.loading(`Reading ${what}\u2026`);
    let done = 0;
    const why = new Set<string>();
    const errors = new Set<string>(docs.map(d => d.parseError || "").filter(Boolean));
    for (const d of docs) {
      const { text, method } = await readTextOnServer(d.id);
      if (!text.trim()) why.add(method);
      if (text.trim()) {
        done++;
        // The document has been read, so a title taken from the filename can be
        // replaced by one taken from the paper.
        if (d.titleFromFilename) {
          const better = guessTitle("", text.slice(0, 3000).replace(/^#+\s*/gm, ""), d.title);
          if (better && better !== d.title) renameStudy(d.id, better);
        }
        s.setFullTexts(prev => ({
          ...prev,
          [d.id]: {
            ...(prev[d.id] || {}),
            paper_id: d.id, title: d.title, url: "", source: UPLOAD_SOURCE,
            status: "found", text,
          },
        }));
      }
      toast.loading(`Reading ${what}\u2026 ${done} of ${docs.length}`, { id: tid });
    }
    if (done === docs.length) {
      toast.success(`Read ${done} of ${docs.length}`, { id: tid });
    } else if (done) {
      toast.warning(`Read ${done} of ${docs.length}. ${reasonFor(why, errors)}`, { id: tid, duration: 9000 });
    } else {
      toast.error(`Could not read ${what}. ${reasonFor(why, errors)}`, { id: tid, duration: 9000 });
    }
  }

  async function importFiles(files: File[]) {
    const usable = files.filter(isAccepted);
    if (!usable.length) { toast.error("Attach PDF, Word, spreadsheet (CSV/Excel), or reference (RIS/BibTeX) files."); return; }
    // No cap: a reviewer arriving mid-review may legitimately have thousands of
    // PDFs. Parsing is per-file though, so say what it will cost rather than
    // either blocking them or letting the tab look frozen. The real limits are
    // time and the browser's storage quota, not a number we picked.
    const pdfCount = usable.filter(f => /\.pdf$/i.test(f.name) || f.type === "application/pdf").length;
    if (pdfCount > 500) {
      toast.warning(
        `Reading ${pdfCount.toLocaleString()} PDFs. This runs one file at a time and will take a while; `
        + `you can keep working and it will keep going.`,
        { duration: 9000 },
      );
    }
    setBusy(true);
    setProgress({ done: 0, total: usable.length, name: "" });
    try {
      const { studies, failed } = await importStudies(usable, (done, total, name) => setProgress({ done, total, name }));
      const loaded = loadIntoCorpus(studies);
      if (studies.length) {
        toast.success(`Attached ${studies.length} stud${studies.length === 1 ? "y" : "ies"}`
          + (failed.length ? `. ${failed.length} could not be read (unsupported or corrupt file?)` : ""));
        // A scanned PDF imports fine and carries no words, so it would appear
        // at every stage that works from a record and at none of the stages
        // that read text. The server has the file and can render and read it,
        // so that happens here rather than being handed back as a chore.
        //
        // Keyed on what loadIntoCorpus reports, not on the parse: a document
        // that was already in the corpus belongs to the id the corpus gave it.
        const unreadable = loaded.filter(d => !d.hasText);
        if (unreadable.length) void recoverText(unreadable);
      } else {
        toast.error("Could not read those files (unsupported or corrupt?).");
      }
    } catch (e: any) {
      toast.error(e?.message || "Attach failed");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  // A pasted list of names, DOIs or PubMed ids. Each is resolved to a real
  // record first: a bare title has no abstract to screen and no DOI to acquire
  // a full text with, so importing it unresolved would look like progress and
  // then fail silently two stages later.
  async function importTitles(queries: string[]) {
    const list = queries.map(q => q.trim()).filter(Boolean);
    if (!list.length) return;
    setBusy(true);
    setProgress({ done: 0, total: list.length, name: "Looking up" });
    try {
      const out = await apiFetch("/papers/lookup", {
        method: "POST",
        body: JSON.stringify({ queries: list, cap: list.length }),
      });
      const papers: Paper[] = (out?.papers || []).map((p: any) => ({
        id: String(p.id || p.doi || p.title),
        source: p.source || "Imported list",
        title: p.title || "Untitled",
        abstract: p.abstract || "",
        url: p.url || "",
        year: p.year ?? undefined,
        authors: p.authors || undefined,
        // Kept because the full-text fetch needs it. A PMC or publisher URL
        // has no DOI in it to mine, so dropping it here made every study
        // looked up BY its DOI come back "no_doi" when its text was fetched.
        doi: p.doi || undefined,
      }));
      const missing: string[] = out?.missing || [];

      if (papers.length) {
        loadIntoCorpus(papers.map(paper => ({ paper })) as any);
        // Looked-up records carry no article. The stages that read one fetch it
        // here rather than leaving the reviewer on an empty page.
        if (readsFullText(stage)) void acquireFullTexts(papers);
        toast.success(
          `Added ${papers.length.toLocaleString()} of ${list.length.toLocaleString()}`
          + (missing.length ? `. ${missing.length} not found.` : ""),
        );
      } else {
        toast.error("None of those could be found. Check the spelling, or paste DOIs.");
      }
      if (missing.length) {
        // Name them rather than only counting: the reviewer has to go and find
        // these by hand, and a count alone does not say which.
        console.warn("[import] not found:", missing);
      }
    } catch (e: any) {
      toast.error(e?.message || "Lookup failed");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  async function enhanceWithAI() {
    const uploaded = (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE);
    if (!uploaded.length) return;
    setBusy(true);
    const updates = new Map<string, Partial<Paper>>();
    for (let i = 0; i < uploaded.length; i++) {
      const p = uploaded[i];
      setProgress({ done: i, total: uploaded.length, name: p.title.slice(0, 48) });
      const text = s.fullTexts[p.id]?.text || p.abstract || "";
      try {
        const m = await AIService.extractPdfMetadata(text, p.title);
        updates.set(p.id, {
          title: m.title || p.title,
          abstract: m.abstract || p.abstract,
          year: m.year ?? p.year,
          authors: m.authors || p.authors,
          url: m.doi ? `https://doi.org/${m.doi}` : p.url,
        });
      } catch { /* keep heuristic values on failure */ }
    }
    const apply = (list: Paper[] | null) => (list || []).map(p => (updates.has(p.id) ? { ...p, ...updates.get(p.id) } : p));
    s.setRawPapers(apply(s.rawPapers));
    s.setUniquePapers(apply(s.uniquePapers));
    s.setFullTexts(prev => {
      const next = { ...prev };
      updates.forEach((u, id) => { if (next[id] && u.title) next[id] = { ...next[id], title: u.title }; });
      return next;
    });
    setBusy(false);
    setProgress(null);
    toast.success(`Enhanced ${uploaded.length} stud${uploaded.length === 1 ? "y" : "ies"} with AI`);
  }

  function clearStudies() {
    (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE).forEach(p => removePdfBlob(p.id));
    const keep = (s.rawPapers || []).filter(p => p.source !== UPLOAD_SOURCE);
    s.setRawPapers(keep.length ? keep : null);
    s.setUniquePapers(keep.length ? keep : null);
    toast.success("Removed attached studies");
  }

  function removeStudy(id: string) {
    const drop = (list: Paper[] | null) => {
      const next = (list || []).filter(p => p.id !== id);
      return next.length ? next : null;
    };
    removePdfBlob(id);
    s.setRawPapers(drop(s.rawPapers));
    s.setUniquePapers(drop(s.uniquePapers));
    s.setFullTexts(prev => { const n = { ...prev }; delete n[id]; return n; });
  }

  const uploaded = (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE);
  return { importFiles, importTitles, enhanceWithAI, clearStudies, removeStudy, uploaded, uploadedCount: uploaded.length, busy, progress };
}
