// The app's PDF viewer.
//
// Every PDF surface in the app used to be a bare <iframe>, which hands the page
// to the browser's own viewer: its toolbar, its typography, its zoom controls,
// none of them the app's, and no way to mark anything on the page. That last
// part is the one that matters here. A reviewer checking an extracted number
// wants the sentence it was read from marked on the page it was printed on, and
// an iframe cannot do that at all.
//
// So the document is rendered here with pdf.js: all pages in one scroll, the
// quote marked where it falls, and a toolbar that belongs to this app.
//
// Three things worth knowing about the implementation:
//
//   - Pages rasterise at devicePixelRatio. Rendering at CSS size and letting
//     the browser upscale is what made the old quote view look soft on a
//     retina screen, which reads as a bad render rather than a small one.
//   - Highlight geometry is computed once at scale 1 and multiplied by the
//     current scale. The viewport transform is linear in scale, so this is
//     exact, and it means zooming does not re-run the text matching and a jump
//     to a quote does not have to wait for a page to rasterise.
//   - Only the pages near the viewport hold a canvas. A 40-page PDF at retina
//     density is hundreds of megabytes of bitmap if you keep them all.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft, ChevronRight, ExternalLink, FileType2, Loader2, Minus, Plus, Scan,
} from "lucide-react";

import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { getPdfFile } from "../lib/pdfBlobs";
import { ensurePdfSupport } from "../lib/pdfImport";
import { itemBox, matchItems, mergeBoxes, type Box, type PdfTextItem } from "../lib/pdfHighlight";
import { Button } from "./ui/button";

/** A quote found in the document, with its geometry at scale 1. */
type Hit = { quoteIndex: number; page: number; boxes1: Box[] };

type Doc = {
  /** The pdf.js document. */
  api: any;
  /** Each page's size at scale 1, for sizing a placeholder before it renders. */
  sizes: { width: number; height: number }[];
};

type Status =
  | { kind: "loading" }
  | { kind: "nofile" }
  | { kind: "error" }
  | { kind: "ready" };

const MIN_SCALE = 0.35;
const MAX_SCALE = 4;
/** Pages to keep rasterised. Beyond this the furthest from view is released. */
const MAX_CANVASES = 6;
/** A ceiling on one page's bitmap. At high zoom a letter page at device density
 *  runs to tens of millions of pixels, and several of those at once is hundreds
 *  of megabytes of canvas for no visible gain. */
const MAX_PIXELS = 9e6;
/** How far outside the viewport a page starts rendering. */
const RENDER_MARGIN = "700px 0px";
/** Padding inside the scroll area, both sides, in CSS pixels. */
const GUTTER = 32;

export type PdfViewerProps = {
  /** A document in the upload registry. Takes precedence over `url`. */
  paperId?: string;
  /** Or anything fetchable: a backend route, or an object URL. */
  url?: string;
  /** Sentences to mark. The first one found is scrolled to on open. */
  quotes?: string[];
  /** Page to open on, when no quote is given. */
  initialPage?: number;
  /** Shown at the left of the toolbar. */
  title?: string;
  className?: string;
  /** Offered when the document will not render here. */
  externalUrl?: string;
  /** Hide the toolbar's document title, for a pane that has its own header. */
  hideTitle?: boolean;
  /** Which page each quote was found on, once the search has run. Lets the
   *  caller name the page beside its own copy of the quote, so the two agree. */
  onQuotesLocated?: (found: { quoteIndex: number; page: number }[]) => void;
};

export function PdfViewer({
  paperId, url, quotes, initialPage, title, className = "", externalUrl, hideTitle,
  onQuotesLocated,
}: PdfViewerProps) {
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [doc, setDoc] = useState<Doc | null>(null);
  const [hits, setHits] = useState<Hit[]>([]);
  const [page, setPage] = useState(initialPage ?? 1);
  const [scale, setScale] = useState(1);
  /** Until the reader zooms, the page tracks the width of the pane. */
  const [autoFit, setAutoFit] = useState(true);
  const [near, setNear] = useState<Set<number>>(() => new Set([initialPage ?? 1]));
  const [fileUrl, setFileUrl] = useState("");

  const scrollRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef(new Map<number, HTMLDivElement>());
  /** Pages that hold a rasterised canvas, newest last. */
  const rendered = useRef<number[]>([]);
  const renderTasks = useRef(new Map<number, any>());
  /** The latest render asked for per page. Bumped synchronously, so a render
   *  that loses the race can tell and stand down. */
  const renderSeq = useRef(new Map<number, number>());
  /** A jump requested before the geometry was known. */
  const pendingJump = useRef<number | null>(null);
  // Held in a ref so a caller passing an inline function cannot restart the
  // document load, which is the expensive part of this component.
  const report = useRef(onQuotesLocated);
  report.current = onQuotesLocated;
  // Read inside the load, but not a reason to run it again: moving to another
  // page must not re-download and re-parse the document.
  const openAt = useRef(initialPage);
  openAt.current = initialPage;

  // The quote list is an array literal at most call sites, so a new identity
  // every render. Key the load on its content instead.
  const quoteKey = (quotes ?? []).join("\u0000");

  // ── Load ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    let dropped = false;
    setStatus({ kind: "loading" });
    setDoc(null);
    setHits([]);
    rendered.current = [];

    let localUrl = "";
    let api: any = null;

    (async () => {
      try {
        // The stored bytes first: they outlive an object URL and survive a
        // reload. A url is the fallback, so a caller may pass both without
        // having to work out which one it has.
        let bytes: ArrayBuffer | null = null;

        if (paperId) {
          const file = await getPdfFile(paperId);
          if (dropped) return;
          if (file) {
            bytes = await file.arrayBuffer();
            localUrl = URL.createObjectURL(file);
          }
        }
        if (!bytes && url) {
          // The hash is the browser viewer's page anchor and means nothing to a
          // fetch, but it does change the cache key, so drop it.
          const res = await fetch(url.replace(/#.*$/, ""));
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const blob = await res.blob();
          if (dropped) return;
          bytes = await blob.arrayBuffer();
          localUrl = URL.createObjectURL(blob);
        }
        if (dropped) return;
        if (!bytes) { setStatus({ kind: "nofile" }); return; }
        setFileUrl(localUrl);

        ensurePdfSupport();
        const pdfjs: any = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
        api = await pdfjs.getDocument({ data: bytes, isEvalSupported: false }).promise;
        if (dropped) return;

        // Page sizes first, so the scrollbar is the right length from the start
        // and a jump can be measured before anything has rasterised.
        const sizes: { width: number; height: number }[] = [];
        for (let n = 1; n <= api.numPages; n++) {
          const vp = (await api.getPage(n)).getViewport({ scale: 1 });
          sizes.push({ width: vp.width, height: vp.height });
        }
        if (dropped) return;
        setDoc({ api, sizes });
        setStatus({ kind: "ready" });

        // Then find the quotes. A separate pass, after the document is on
        // screen: the reader should not wait on a text search to see page one.
        const wanted = (quotes ?? []).map(q => (q || "").trim());
        if (wanted.some(Boolean)) {
          const found: Hit[] = [];
          const outstanding = new Set(wanted.map((q, i) => (q ? i : -1)).filter(i => i >= 0));
          for (let n = 1; n <= api.numPages && outstanding.size; n++) {
            const p = await api.getPage(n);
            const content = await p.getTextContent();
            if (dropped) return;
            const items = (content.items || []).filter((i: any) => "str" in i) as PdfTextItem[];
            if (!items.length) continue;
            const vp1 = p.getViewport({ scale: 1 });
            for (const qi of [...outstanding]) {
              const match = matchItems(items, wanted[qi]);
              if (!match.length) continue;
              found.push({
                quoteIndex: qi,
                page: n,
                boxes1: mergeBoxes(match.map(m => itemBox(m.item, vp1.transform, 1))),
              });
              outstanding.delete(qi);
            }
          }
          if (dropped) return;
          found.sort((a, b) => a.quoteIndex - b.quoteIndex);
          setHits(found);
          report.current?.(found.map(h => ({ quoteIndex: h.quoteIndex, page: h.page })));
          // Open on the first quote rather than on page one.
          if (found.length && openAt.current == null) pendingJump.current = 0;
        }
      } catch (e) {
        // The reader gets a plain sentence and a way out. The detail belongs in
        // the console, for whoever is fixing it rather than reading a paper.
        console.error("Could not render this PDF:", e);
        if (!dropped) setStatus({ kind: "error" });
      }
    })();

    return () => {
      dropped = true;
      for (const task of renderTasks.current.values()) {
        try { task.cancel(); } catch { /* already settled */ }
      }
      renderTasks.current.clear();
      try { api?.destroy?.(); } catch { /* nothing to destroy */ }
      if (localUrl) { try { URL.revokeObjectURL(localUrl); } catch { /* ignore */ } }
    };
  }, [paperId, url, quoteKey]);

  // ── Fit to the pane ─────────────────────────────────────────────────────────
  const widest = useMemo(
    () => (doc ? Math.max(...doc.sizes.map(s => s.width), 1) : 1),
    [doc],
  );

  const fit = useCallback(() => {
    const el = scrollRef.current;
    if (!el || !doc) return;
    const room = el.clientWidth - GUTTER;
    if (room <= 0) return;
    // Quantised, because a resize fires continuously and setting a new scale
    // clears every canvas before it redraws. Rounded to the nearest percent,
    // a drag across the width of the screen costs a handful of redraws rather
    // than one per pixel, and the page does not flicker while it is dragged.
    const next = Math.round(clamp(room / widest) * 100) / 100;
    setScale(prev => (Math.abs(prev - next) < 0.005 ? prev : next));
  }, [doc, widest]);

  useLayoutEffect(() => {
    if (!autoFit || !doc) return;
    fit();
    const el = scrollRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => fit());
    ro.observe(el);
    return () => ro.disconnect();
  }, [autoFit, doc, fit]);

  // ── Which pages are worth rasterising ───────────────────────────────────────
  useEffect(() => {
    const root = scrollRef.current;
    if (!root || !doc || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      entries => {
        setNear(prev => {
          const next = new Set(prev);
          let changed = false;
          for (const e of entries) {
            const n = Number((e.target as HTMLElement).dataset.page);
            if (!n) continue;
            if (e.isIntersecting && !next.has(n)) { next.add(n); changed = true; }
          }
          return changed ? next : prev;
        });
      },
      { root, rootMargin: RENDER_MARGIN },
    );
    for (const el of pageRefs.current.values()) io.observe(el);
    return () => io.disconnect();
  }, [doc]);

  // ── Rasterise ───────────────────────────────────────────────────────────────
  const renderPage = useCallback(async (n: number) => {
    const d = doc;
    const host = pageRefs.current.get(n);
    if (!d || !host) return;
    const canvas = host.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return;
    if (canvas.dataset.scale === String(scale)) return;      // already current

    // One render at a time per canvas, and the last one asked for wins.
    //
    // Claiming the canvas has to happen before the first await. Registering the
    // task afterwards left a window in which a resize could start a second
    // render with nothing to cancel, and pdf.js will not draw two things into
    // one canvas: the page came out half-drawn at the wrong scale, with the
    // quote highlights, which are plain DOM, correctly placed for the new one.
    const seq = (renderSeq.current.get(n) ?? 0) + 1;
    renderSeq.current.set(n, seq);
    const current = () => renderSeq.current.get(n) === seq;

    // Cancelling is not instant: pdf.js holds the canvas until the cancelled
    // render actually settles, and starting the replacement before that throws
    // "Cannot use the same canvas during multiple render operations". The page
    // then stayed blank, because nothing re-triggers a render for a scale that
    // has already been applied. So wait for the canvas to come free.
    const running = renderTasks.current.get(n);
    if (running) {
      try { running.cancel(); } catch { /* already settled */ }
      await Promise.resolve(running.promise).catch(() => { /* cancelled, as asked */ });
      if (!current()) return;
      if (renderTasks.current.get(n) === running) renderTasks.current.delete(n);
    }

    try {
      const p = await d.api.getPage(n);
      if (!current()) return;                                // superseded while waiting
      const ctx = canvas.getContext("2d");
      if (!ctx) return;                                      // no 2D canvas here
      const css = p.getViewport({ scale });
      const screen = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
      const area = Math.max(1, css.width * css.height);
      const dpr = Math.max(1, Math.min(screen, 2, Math.sqrt(MAX_PIXELS / area)));
      const raster = p.getViewport({ scale: scale * dpr });
      canvas.width = Math.floor(raster.width);
      canvas.height = Math.floor(raster.height);
      canvas.style.width = `${Math.floor(css.width)}px`;
      canvas.style.height = `${Math.floor(css.height)}px`;
      const task = p.render({ canvasContext: ctx, viewport: raster, canvas });
      renderTasks.current.set(n, task);
      await task.promise;
      if (renderTasks.current.get(n) === task) renderTasks.current.delete(n);
      if (!current()) return;                                // a newer scale won
      canvas.dataset.scale = String(scale);

      // Release the canvases furthest from what is being read.
      rendered.current = [...rendered.current.filter(x => x !== n), n];
      while (rendered.current.length > MAX_CANVASES) {
        const victim = rendered.current
          .slice(0, -1)
          .reduce((far, x) => (Math.abs(x - n) > Math.abs(far - n) ? x : far));
        rendered.current = rendered.current.filter(x => x !== victim);
        const old = pageRefs.current.get(victim)?.querySelector("canvas");
        if (old instanceof HTMLCanvasElement) {
          old.width = 0;
          old.height = 0;
          delete old.dataset.scale;
        }
      }
    } catch (e: any) {
      // A cancelled render is the expected outcome of scrolling or zooming.
      if (e?.name !== "RenderingCancelledException") {
        console.error(`Could not render page ${n}:`, e);
      }
      // Only if it is still ours: a newer render may already have claimed it.
      if (current()) renderTasks.current.delete(n);
    }
  }, [doc, scale]);

  useEffect(() => {
    if (!doc) return;
    for (const n of near) renderPage(n);
  }, [doc, near, scale, renderPage]);

  // ── Jump to a quote ─────────────────────────────────────────────────────────
  const jumpTo = useCallback((hitIndex: number) => {
    const hit = hits[hitIndex];
    const host = hit && pageRefs.current.get(hit.page);
    if (!hit || !host) { pendingJump.current = hitIndex; return; }
    setPage(hit.page);
    const top = hit.boxes1.length ? Math.min(...hit.boxes1.map(b => b.top)) * scale : 0;
    const scroller = scrollRef.current;
    if (!scroller) return;
    // Put the marked sentence a third of the way down, so what comes before it
    // is on screen too: a number means little without the row above it.
    const target = host.offsetTop + top - scroller.clientHeight / 3;
    scrollTop(scroller, Math.max(0, target));
  }, [hits, scale]);

  useEffect(() => {
    if (pendingJump.current == null || !hits.length) return;
    const i = pendingJump.current;
    pendingJump.current = null;
    // After layout, so offsetTop is the laid-out one.
    const id = requestAnimationFrame(() => jumpTo(i));
    return () => cancelAnimationFrame(id);
  }, [hits, jumpTo]);

  // ── Which page is being read ────────────────────────────────────────────────
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !doc) return;
    let frame = 0;
    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const mid = el.scrollTop + el.clientHeight / 2;
        let best = 1;
        for (const [n, host] of pageRefs.current) {
          if (host.offsetTop <= mid) best = Math.max(best, n);
        }
        setPage(best);
      });
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [doc]);

  useEffect(() => {
    if (initialPage == null || !doc) return;
    const host = pageRefs.current.get(Math.min(doc.sizes.length, Math.max(1, initialPage)));
    const scroller = scrollRef.current;
    if (!host || !scroller) return;
    setPage(initialPage);
    setNear(prev => (prev.has(initialPage) ? prev : new Set(prev).add(initialPage)));
    const id = requestAnimationFrame(() => scrollTop(scroller, Math.max(0, host.offsetTop - 8)));
    return () => cancelAnimationFrame(id);
  }, [initialPage, doc]);

  const goToPage = (n: number) => {
    if (!doc) return;
    const clamped = Math.min(doc.sizes.length, Math.max(1, n));
    const host = pageRefs.current.get(clamped);
    const scroller = scrollRef.current;
    if (!host || !scroller) return;
    setNear(prev => (prev.has(clamped) ? prev : new Set(prev).add(clamped)));
    scrollTop(scroller, Math.max(0, host.offsetTop - 8));
  };

  const zoom = (by: number) => {
    setAutoFit(false);
    setScale(s => clamp(s + by));
  };

  const open = externalUrl || fileUrl;
  const byPage = useMemo(() => {
    const m = new Map<number, Hit[]>();
    for (const h of hits) m.set(h.page, [...(m.get(h.page) ?? []), h]);
    return m;
  }, [hits]);

  const asked = (quotes ?? []).filter(q => (q || "").trim()).length;

  // ── Nothing to show ─────────────────────────────────────────────────────────
  if (status.kind === "nofile" || status.kind === "error") {
    return (
      <div className={`flex flex-col items-center justify-center gap-2 rounded-lg border
                       bg-muted/20 p-6 text-center text-sm text-muted-foreground ${className}`}>
        <FileType2 className="size-7 opacity-40" />
        <div>
          {status.kind === "nofile"
            ? "No document file for this study."
            : "This file could not be displayed here."}
        </div>
        {open && (
          <a href={open} target="_blank" rel="noreferrer"
             className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
            Open it in a new tab <ExternalLink className="size-3" />
          </a>
        )}
      </div>
    );
  }

  return (
    <div className={`flex min-h-0 flex-col overflow-hidden rounded-lg border bg-card ${className}`}>
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-1 border-b bg-muted/30 px-2 py-1.5 text-xs">
        {!hideTitle && (
          <>
            <FileType2 className="size-3.5 shrink-0 text-muted-foreground" />
            <span className="truncate font-medium" title={title}>{title || "Document"}</span>
          </>
        )}

        {/* Jump to a marked sentence. One quote needs no chip: it is where the
            view already opened. */}
        {hits.length > 1 && (
          <div className="ml-2 flex shrink-0 items-center gap-1">
            {hits.map((h, i) => (
              <button
                key={h.quoteIndex}
                type="button"
                onClick={() => jumpTo(i)}
                title={`Go to quote ${h.quoteIndex + 1}, on page ${h.page}`}
                className="rounded border bg-background px-1.5 py-0.5 font-medium tabular-nums
                           text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
              >
                {h.quoteIndex + 1}
              </button>
            ))}
          </div>
        )}

        {/* An unverified value looks exactly like a verified one, so a sentence
            that is not in the file has to be said out loud. */}
        {asked === 1 && status.kind === "ready" && !hits.length && doc && (
          <span className="ml-2 shrink-0 text-rose-600 dark:text-rose-400">
            Not found in this file
          </span>
        )}

        <div className="ml-auto flex shrink-0 items-center gap-0.5">
          <Button size="icon" variant="ghost" className="size-6" title="Previous page"
                  disabled={page <= 1} onClick={() => goToPage(page - 1)}>
            <ChevronLeft className="size-3.5" />
          </Button>
          <span className="min-w-14 text-center tabular-nums text-muted-foreground">
            {doc ? `${page} / ${doc.sizes.length}` : "…"}
          </span>
          <Button size="icon" variant="ghost" className="size-6" title="Next page"
                  disabled={!doc || page >= doc.sizes.length} onClick={() => goToPage(page + 1)}>
            <ChevronRight className="size-3.5" />
          </Button>

          <span className="mx-1 h-4 w-px bg-border" />

          <Button size="icon" variant="ghost" className="size-6" title="Zoom out"
                  disabled={scale <= MIN_SCALE} onClick={() => zoom(-0.2)}>
            <Minus className="size-3.5" />
          </Button>
          <span className="min-w-10 text-center tabular-nums text-muted-foreground">
            {Math.round(scale * 100)}%
          </span>
          <Button size="icon" variant="ghost" className="size-6" title="Zoom in"
                  disabled={scale >= MAX_SCALE} onClick={() => zoom(0.2)}>
            <Plus className="size-3.5" />
          </Button>
          <Button size="icon" variant="ghost" className="size-6"
                  title="Fit the page to the width of this pane"
                  onClick={() => { setAutoFit(true); fit(); }}>
            <Scan className="size-3.5" />
          </Button>

          {open && (
            <a href={open} target="_blank" rel="noreferrer"
               title="Open this document in a new tab"
               className="ml-0.5 inline-flex size-6 items-center justify-center rounded
                          text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
              <ExternalLink className="size-3.5" />
            </a>
          )}
        </div>
      </div>

      {/* Pages, on a mat. A white sheet with a shadow is what makes this read as
          a document rather than as an embedded frame. */}
      <div ref={scrollRef}
           className="relative min-h-0 flex-1 overflow-auto bg-neutral-200/80 px-4 py-4 dark:bg-neutral-900">
        {status.kind === "loading" && (
          <div className="flex h-full items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" />Opening the document…
          </div>
        )}

        {doc && (
          <div className="flex flex-col items-center gap-4">
            {doc.sizes.map((size, i) => {
              const n = i + 1;
              const w = Math.floor(size.width * scale);
              const h = Math.floor(size.height * scale);
              return (
                <div
                  key={n}
                  data-page={n}
                  ref={el => {
                    if (el) pageRefs.current.set(n, el);
                    else pageRefs.current.delete(n);
                  }}
                  style={{ width: w, height: h }}
                  className="relative shrink-0 overflow-hidden rounded-sm bg-white shadow-[0_1px_3px_rgba(0,0,0,0.18),0_8px_24px_-8px_rgba(0,0,0,0.25)]"
                >
                  <canvas className="block" />
                  {(byPage.get(n) ?? []).map(hit =>
                    hit.boxes1.map((b, k) => (
                      <div
                        key={`${hit.quoteIndex}-${k}`}
                        aria-hidden
                        className="pointer-events-none absolute rounded-sm bg-amber-300/40 ring-1 ring-amber-500/60 mix-blend-multiply"
                        style={{
                          left: b.left * scale,
                          top: b.top * scale,
                          width: b.width * scale,
                          height: b.height * scale,
                        }}
                      />
                    )),
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function clamp(s: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));
}

/** Scroll smoothly where the browser allows it. `scrollTo` is missing in some
 *  environments and throws rather than doing nothing, and a viewer that cannot
 *  animate a jump should still make the jump. */
function scrollTop(el: HTMLElement, top: number) {
  try {
    if (typeof el.scrollTo === "function") {
      el.scrollTo({ top, behavior: "smooth" });
      return;
    }
  } catch { /* fall through to the plain assignment */ }
  el.scrollTop = top;
}
