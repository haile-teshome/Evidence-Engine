// Bring your own studies in, at whatever stage you are actually at.
//
// A reviewer may have done the earlier steps elsewhere and arrive with a folder
// of PDFs ready for full-text screening, or with nothing but a list of titles.
// One button on every stage page opens one dialog that takes all of it: files,
// a folder, a drag of either, or a pasted list of names, DOIs or PubMed ids.
//
// Dropping is handled inside the dialog rather than on the page, because
// Acquisition already has its own page-level drop target for bulk PDFs and two
// overlapping drop zones on one page is a coin toss about which one catches it.
//
// What happens after the import depends on the stage: see lib/importedAt.ts for
// why records arriving past abstract screening are marked rather than quietly
// written in as though this app had screened them.
import { useRef, useState } from "react";
import { Upload, Loader2, Plus } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "./ui/dialog";
import { useStudyImport } from "../lib/useStudyImport";
import { ImportStage, STAGE_LABEL } from "../lib/importedAt";

/** Walk a dragged folder. The entry API is the only way to see inside one. */
async function filesFromDataTransfer(dt: DataTransfer): Promise<File[]> {
  const entries: any[] = [];
  for (const item of Array.from(dt.items || [])) {
    const entry = (item as any).webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  if (!entries.length) return Array.from(dt.files || []);

  const out: File[] = [];
  async function walk(entry: any): Promise<void> {
    if (!entry) return;
    if (entry.isFile) {
      const f: File = await new Promise((res, rej) => entry.file(res, rej));
      out.push(f);
      return;
    }
    if (entry.isDirectory) {
      const reader = entry.createReader();
      // readEntries returns at most 100 per call, so keep going until it is
      // empty. Reading once silently truncates any folder bigger than that.
      for (;;) {
        const batch: any[] = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e);
      }
    }
  }
  for (const e of entries) await walk(e);
  return out;
}

export function ImportStudies({
  stage,
  label = "Add studies",
  onFiles,
}: {
  stage: ImportStage;
  label?: string;
  // When a page can do something smarter with the files than create records
  // from all of them, it takes them here instead. Acquisition does this: a file
  // whose name matches a study already in the review should attach to it as
  // that study's full text, not arrive as a second copy of the same paper.
  onFiles?: (files: File[]) => void;
}) {
  const { importFiles, importTitles, busy, progress } = useStudyImport(stage);
  const fileRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [over, setOver] = useState(false);
  const [listText, setListText] = useState("");

  const titles = listText.split("\n").map(t => t.trim()).filter(Boolean);

  function take(files: File[]) {
    if (!files.length) return;
    setOpen(false);
    if (onFiles) onFiles(files);
    else importFiles(files);
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setOver(false);
    take(await filesFromDataTransfer(e.dataTransfer));
  }

  return (
    <>
      {/* Same size, variant and shadow as the other actions in the pane it sits in. */}
      <Button size="sm" variant="outline" className="h-8 shadow-sm" disabled={busy}
              onClick={() => setOpen(true)}>
        {busy
          ? <Loader2 className="size-3.5 mr-1.5 animate-spin" />
          : <Plus className="size-3.5 mr-1.5" />}
        {busy && progress
          ? `${progress.done}/${progress.total}`
          : label}
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[min(1100px,92vw)] max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add your own studies</DialogTitle>
            <DialogDescription>
              New studies, added at {STAGE_LABEL[stage]}.
            </DialogDescription>
          </DialogHeader>

          <div
            role="button"
            tabIndex={0}
            onClick={() => fileRef.current?.click()}
            onKeyDown={e => { if (e.key === "Enter" || e.key === " ") fileRef.current?.click(); }}
            onDragOver={e => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={onDrop}
            className={`rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
              over ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:bg-muted/30"
            }`}>
            <Upload className="size-8 mx-auto mb-3 text-muted-foreground" />
            <div className="text-base font-medium">Drop files or a folder here</div>
            <div className="text-sm text-muted-foreground mt-1">or click to choose files</div>
          </div>

          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <div className="h-px flex-1 bg-border" />
            or paste a list
            <div className="h-px flex-1 bg-border" />
          </div>

          <Textarea
            value={listText}
            onChange={e => setListText(e.target.value)}
            className="font-mono text-xs min-h-[260px] resize-y"
            placeholder={"Effect of X on Y: a randomised trial\n10.1001/jama.2020.1234\n32150360"}
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {titles.length
                ? `${titles.length.toLocaleString()} to look up, to fill in abstracts and identifiers`
                : "One per line: a title, a DOI, or a PubMed ID"}
            </span>
            <Button
              size="sm"
              disabled={!titles.length || busy}
              onClick={async () => { setOpen(false); const q = titles; setListText(""); await importTitles(q); }}>
              Look up and add
            </Button>
          </div>

          <input
            ref={fileRef} type="file" multiple hidden
            onChange={e => { const f = Array.from(e.target.files || []); e.target.value = ""; take(f); }}
          />
        </DialogContent>
      </Dialog>
    </>
  );
}
