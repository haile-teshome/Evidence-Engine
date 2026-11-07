// The page of the PDF a value was read from, with the sentence marked on it.
//
// A quote in a box proves the sentence exists in the document. It does not show
// the table it sits under or the figure beside it, and it does not tell the
// reviewer which page to turn to in the paper they are about to cite. So the
// page itself is rendered here and the sentence is marked on it, which is what
// anyone checking an extracted number actually wants to look at.
//
// Everything is best-effort: a review must never stop because a PDF would not
// render. Each failure says which one it is, because "no file here", "this page
// does not contain that sentence" and "this file cannot be read" send the
// reviewer in three different directions.
import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { getPdfFile } from "../lib/pdfBlobs";
import { ensurePdfSupport } from "../lib/pdfImport";
import { matchItems, itemBox, mergeBoxes, Box, PdfTextItem } from "../lib/pdfHighlight";

type State =
  | { kind: "loading" }
  | { kind: "nofile" }
  | { kind: "error" }
  | { kind: "ready"; page: number; pages: number; found: boolean };

export function PdfQuotePage({ paperId, quote }: { paperId: string; quote: string }) {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [boxes, setBoxes] = useState<Box[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let dropped = false;
    setState({ kind: "loading" });
    setBoxes([]);

    (async () => {
      try {
        const file = await getPdfFile(paperId);
        if (dropped) return;
        if (!file) { setState({ kind: "nofile" }); return; }

        ensurePdfSupport();
        const pdfjs: any = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
        const doc = await pdfjs.getDocument({
          data: await file.arrayBuffer(), isEvalSupported: false,
        }).promise;
        if (dropped) return;

        // Which page is it on? Reading order, first match wins: a sentence
        // repeated in an abstract and a results section is cited from the
        // first place it appears, which is where a reader would look.
        let target = 1;
        let hits: { item: PdfTextItem }[] = [];
        for (let n = 1; n <= doc.numPages; n++) {
          const page = await doc.getPage(n);
          const content = await page.getTextContent();
          if (dropped) return;
          const items = (content.items || []).filter((i: any) => "str" in i) as PdfTextItem[];
          const found = matchItems(items, quote);
          if (found.length) { target = n; hits = found; break; }
        }

        const page = await doc.getPage(target);
        if (dropped) return;
        const unscaled = page.getViewport({ scale: 1 });
        const width = wrapRef.current?.clientWidth || 760;
        const scale = Math.min(2, Math.max(0.5, width / unscaled.width));
        const viewport = page.getViewport({ scale });

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        await page.render({ canvasContext: canvas.getContext("2d"), viewport, canvas }).promise;
        if (dropped) return;

        setBoxes(mergeBoxes(hits.map(h => itemBox(h.item, viewport.transform, scale))));
        setState({ kind: "ready", page: target, pages: doc.numPages, found: hits.length > 0 });
      } catch (e: any) {
        // The reviewer gets a plain sentence and the passage below, which is
        // always there. The detail belongs in the console, where it is of use
        // to someone fixing it rather than to someone reading a paper.
        console.error("Could not render the PDF page for the quote:", e,
                      typeof navigator === "undefined" ? "" : navigator.userAgent);
        if (!dropped) setState({ kind: "error" });
      }
    })();

    return () => { dropped = true; };
  }, [paperId, quote]);

  // Nothing at all when the page cannot be shown. The quote and its passage are
  // directly below and answer the question on their own; a line explaining that
  // a viewer failed is the app talking about itself in the reviewer's way. The
  // reason is in the console for whoever is fixing it.
  if (state.kind === "error" || state.kind === "nofile") return null;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-xs text-muted-foreground min-h-5">
        {state.kind === "loading" && (
          <><Loader2 className="size-3.5 animate-spin" />Opening the document…</>
        )}
        {state.kind === "ready" && (
          <span>
            {state.found
              ? `Page ${state.page} of ${state.pages}, with the sentence marked`
              : `Page ${state.page} of ${state.pages}. The sentence was not found on any page of this file.`}
          </span>
        )}
      </div>

      <div ref={wrapRef}
           className={`relative overflow-auto rounded-lg border bg-muted/20 max-h-[52vh]
                       ${state.kind === "ready" ? "" : "hidden"}`}>
        <div className="relative inline-block">
          <canvas ref={canvasRef} className="block" />
          {boxes.map((b, i) => (
            <div key={i} aria-hidden
                 className="absolute rounded-sm bg-amber-300/45 ring-1 ring-amber-500/60
                            pointer-events-none"
                 style={{ left: b.left, top: b.top, width: b.width, height: b.height }} />
          ))}
        </div>
      </div>
    </div>
  );
}
