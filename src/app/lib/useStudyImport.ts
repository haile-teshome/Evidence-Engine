import { useState } from "react";
import { useStore } from "./store";
import { importStudies, isAccepted, UPLOAD_SOURCE, ImportedStudy } from "./pdfImport";
import { registerPdfBlobs } from "./pdfBlobs";
import { AIService } from "./apiClient";
import type { Paper } from "./apiClient";
import { toast } from "sonner";

// Shared logic for importing a user's own PDFs as the study corpus. Used by the
// chat bar's "+" attach button. Keeps the store wiring in one place.
export function useStudyImport() {
  const s = useStore();
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number; name: string } | null>(null);

  function loadIntoCorpus(studies: ImportedStudy[]) {
    if (!studies.length) return;
    const papers = studies.map(x => x.paper);
    const ftMap = Object.fromEntries(studies.filter(x => x.fullText).map(x => [x.paper.id, x.fullText!]));
    registerPdfBlobs(studies.filter(x => x.objectUrl).map(x => ({ id: x.paper.id, url: x.objectUrl! })));

    const existing = (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE);
    const seen = new Set(existing.map(p => p.title.toLowerCase().trim()));
    const fresh = papers.filter(p => !seen.has(p.title.toLowerCase().trim()));
    const all = [...existing, ...fresh];

    s.setRawPapers(all);
    s.setUniquePapers(all);
    s.setDuplicatesCount(0);
    s.setFullTexts(prev => ({ ...prev, ...ftMap }));
    if (!s.sources.includes("Local PDFs")) s.setSources([...s.sources, "Local PDFs"]);
  }

  async function importFiles(files: File[]) {
    const usable = files.filter(isAccepted);
    if (!usable.length) { toast.error("Attach PDF, Word, spreadsheet (CSV/Excel), or reference (RIS/BibTeX) files."); return; }
    // Soft warning for very large PDF sets: parsing + screening is per-file, so
    // hundreds of PDFs take a while. References/spreadsheets are light, so we only
    // warn on the heavy (full-text PDF) count.
    const pdfCount = usable.filter(f => /\.pdf$/i.test(f.name) || f.type === "application/pdf").length;
    if (pdfCount > 500) {
      toast.warning(`Attaching ${pdfCount.toLocaleString()} PDFs — parsing and screening this many takes a while. It still works; consider smaller batches if it feels slow.`, { duration: 9000 });
    }
    setBusy(true);
    setProgress({ done: 0, total: usable.length, name: "" });
    try {
      const { studies, failed } = await importStudies(usable, (done, total, name) => setProgress({ done, total, name }));
      loadIntoCorpus(studies);
      if (studies.length) {
        toast.success(`Attached ${studies.length} stud${studies.length === 1 ? "y" : "ies"}`
          + (failed.length ? ` — ${failed.length} could not be read (scanned/image-only?)` : ""));
      } else {
        toast.error("None of the files yielded extractable text (scanned/image-only PDFs?).");
      }
    } catch (e: any) {
      toast.error(e?.message || "Attach failed");
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
    s.setRawPapers(drop(s.rawPapers));
    s.setUniquePapers(drop(s.uniquePapers));
    s.setFullTexts(prev => { const n = { ...prev }; delete n[id]; return n; });
  }

  const uploaded = (s.rawPapers || []).filter(p => p.source === UPLOAD_SOURCE);
  return { importFiles, enhanceWithAI, clearStudies, removeStudy, uploaded, uploadedCount: uploaded.length, busy, progress };
}
