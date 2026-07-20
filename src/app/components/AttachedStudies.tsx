import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import { useStudyImport } from "../lib/useStudyImport";
import { getPdfBlob } from "../lib/pdfBlobs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Wand2, Trash2, X, FileText, Loader2, ExternalLink } from "lucide-react";

// Panel to review the studies the user attached: a list on the left, a preview
// (PDF or extracted text + metadata) on the right, with Enhance / remove actions.
export function AttachedStudies({
  open,
  onOpenChange,
  studyImport,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  studyImport: ReturnType<typeof useStudyImport>;
}) {
  const s = useStore();
  const { uploaded, enhanceWithAI, clearStudies, removeStudy, busy, progress } = studyImport;
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Keep a valid selection as the list changes.
  useEffect(() => {
    if (!uploaded.length) { setSelectedId(null); return; }
    if (!selectedId || !uploaded.some(p => p.id === selectedId)) setSelectedId(uploaded[0].id);
  }, [uploaded, selectedId]);

  const selected = uploaded.find(p => p.id === selectedId) || null;
  const ft = selected ? s.fullTexts[selected.id] : undefined;
  const blob = selected ? getPdfBlob(selected.id) : undefined;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[96vw] max-w-[1700px] sm:max-w-[1700px] h-[92vh] flex flex-col p-0 gap-0">
        <DialogHeader className="px-5 py-3 border-b flex-row items-center justify-between space-y-0">
          <DialogTitle className="text-base">Attached studies ({uploaded.length})</DialogTitle>
          <div className="flex items-center gap-2 pr-8">
            {busy && (
              <span className="text-xs text-muted-foreground inline-flex items-center gap-1.5">
                <Loader2 className="size-3.5 animate-spin" />
                {progress ? `${progress.done}/${progress.total}` : "Working…"}
              </span>
            )}
            <Button size="sm" variant="outline" onClick={enhanceWithAI} disabled={busy || !uploaded.length}
              title="Fix stray characters and broken words from PDF extraction, and pull clean titles, authors, year, and abstracts">
              <Wand2 className="size-3.5 mr-1.5" />Clean up text
            </Button>
            <Button size="sm" variant="ghost" className="text-muted-foreground" onClick={() => { clearStudies(); onOpenChange(false); }} disabled={!uploaded.length}>
              <Trash2 className="size-3.5 mr-1.5" />Clear all
            </Button>
          </div>
        </DialogHeader>

        {uploaded.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
            No studies attached. Use the + on the chat bar to attach files.
          </div>
        ) : (
          <div className="flex-1 min-h-0 flex">
            {/* List */}
            <div className="w-72 shrink-0 border-r overflow-y-auto">
              {uploaded.map(p => (
                <button
                  key={p.id}
                  onClick={() => setSelectedId(p.id)}
                  className={`group w-full text-left px-3 py-2.5 border-b flex items-start gap-2 hover:bg-muted/60 ${p.id === selectedId ? "bg-primary/10" : ""}`}
                >
                  <FileText className="size-3.5 mt-0.5 shrink-0 text-muted-foreground" />
                  <span className="flex-1 min-w-0">
                    <span className="block text-xs font-medium line-clamp-2">{p.title}</span>
                    <span className="block text-[11px] text-muted-foreground mt-0.5">
                      {p.authors ? p.authors.split(/[;,]/)[0] + " et al." : "—"}{p.year ? ` · ${p.year}` : ""}
                    </span>
                  </span>
                  <span
                    role="button"
                    onClick={e => { e.stopPropagation(); removeStudy(p.id); }}
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0"
                    title="Remove"
                  >
                    <X className="size-3.5" />
                  </span>
                </button>
              ))}
            </div>

            {/* Preview */}
            <div className="flex-1 min-w-0 flex flex-col">
              {selected && (
                <>
                  <div className="px-5 py-3 border-b">
                    <h3 className="font-medium text-sm leading-snug">{selected.title}</h3>
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1.5 text-xs text-muted-foreground">
                      {selected.authors && <span className="truncate max-w-md">{selected.authors}</span>}
                      {selected.year && <span>{selected.year}</span>}
                      <Badge variant="secondary" className="text-[10px]">{blob ? "PDF" : ft?.text ? "Full text" : "Reference"}</Badge>
                      {selected.url && (
                        <a href={selected.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary hover:underline">
                          <ExternalLink className="size-3" />source
                        </a>
                      )}
                    </div>
                  </div>

                  <div className="flex-1 min-h-0 overflow-hidden">
                    {blob ? (
                      <iframe title="PDF preview" src={blob} className="w-full h-full border-0" />
                    ) : (
                      <div className="h-full overflow-y-auto px-5 py-4 space-y-3">
                        {selected.abstract && (
                          <div>
                            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Abstract</div>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{selected.abstract}</p>
                          </div>
                        )}
                        {ft?.text && (
                          <div>
                            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Full text</div>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">{ft.text}</p>
                          </div>
                        )}
                        {!selected.abstract && !ft?.text && (
                          <p className="text-sm text-muted-foreground">No abstract or full text for this reference. Attach the PDF or acquire full text to preview it.</p>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
