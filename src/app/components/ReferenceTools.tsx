import { useRef, useState } from "react";
import { useStore } from "../lib/store";
import { serializeReferences, dedupeAgainstLibrary, paperToRef, refKey, Ref } from "../lib/references";
import { importStudies } from "../lib/pdfImport";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Download, Upload, Library, Loader2, Trash2 } from "lucide-react";
import { toast } from "sonner";

function download(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// Reference-manager interop: export the corpus to RIS / BibTeX (importable by
// Zotero, EndNote, Mendeley), and dedupe it against a library file the
// researcher already has so previously-seen records can be flagged or removed.
export function ReferenceTools() {
  const s = useStore();
  const papers = s.uniquePapers || s.rawPapers || [];
  const [busy, setBusy] = useState(false);
  const [dup, setDup] = useState<{ matched: number; total: number; keys: Set<string> } | null>(null);
  const libRef = useRef<HTMLInputElement>(null);

  function exportRefs(fmt: "RIS" | "BibTeX") {
    if (!papers.length) { toast.error("No papers to export yet."); return; }
    const ext = fmt === "RIS" ? "ris" : "bib";
    const mime = fmt === "RIS" ? "application/x-research-info-systems" : "application/x-bibtex";
    download(`corpus.${ext}`, serializeReferences(papers.map(paperToRef), fmt), mime);
    toast.success(`Exported ${papers.length} references as ${fmt}`);
  }

  async function onLibrary(file: File) {
    setBusy(true);
    try {
      const { studies } = await importStudies([file]);
      if (!studies.length) { toast.error("No references found in that file. Use RIS or BibTeX."); return; }
      const lib: Ref[] = studies.map(st => paperToRef(st.paper));
      const { matched } = dedupeAgainstLibrary(papers.map(paperToRef), lib);
      setDup({ matched, total: papers.length, keys: new Set(lib.map(refKey).filter(Boolean)) });
      toast.success(`${matched} of ${papers.length} already in your library`);
    } catch (e: any) {
      toast.error(e?.message || "Could not read library file");
    } finally {
      setBusy(false);
      if (libRef.current) libRef.current.value = "";
    }
  }

  function removeDuplicates() {
    if (!dup?.keys.size) return;
    const keep = papers.filter(p => !dup.keys.has(refKey(paperToRef(p))));
    s.setUniquePapers(keep.length ? keep : null);
    setDup(null);
    toast.success(`Removed ${papers.length - keep.length} library duplicates from the corpus`);
  }

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-center gap-1.5 text-sm font-medium"><Library className="size-4 text-primary" />Reference library</div>
      <p className="text-xs text-muted-foreground">
        Export the {papers.length} record{papers.length === 1 ? "" : "s"} for Zotero, EndNote, or Mendeley, or upload your existing library (RIS / BibTeX) to flag records you already have.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => exportRefs("RIS")} disabled={!papers.length}><Download className="size-3.5 mr-1.5" />Export RIS</Button>
        <Button size="sm" variant="outline" onClick={() => exportRefs("BibTeX")} disabled={!papers.length}><Download className="size-3.5 mr-1.5" />Export BibTeX</Button>
        <Button size="sm" variant="outline" onClick={() => libRef.current?.click()} disabled={busy || !papers.length}>
          {busy ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Upload className="size-3.5 mr-1.5" />}Dedupe against library
        </Button>
        <input ref={libRef} type="file" accept=".ris,.bib,.nbib,.txt" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) onLibrary(f); }} />
        {dup && (
          <>
            <Badge variant="secondary">{dup.matched} of {dup.total} already in library</Badge>
            {dup.matched > 0 && (
              <Button size="sm" variant="ghost" className="text-destructive" onClick={removeDuplicates}>
                <Trash2 className="size-3.5 mr-1.5" />Remove {dup.matched} from corpus
              </Button>
            )}
          </>
        )}
      </div>
    </Card>
  );
}
