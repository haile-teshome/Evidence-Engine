import { useState, useMemo, useCallback } from "react";
import { Download, ChevronRight, ChevronDown, ExternalLink, ArrowDown, ArrowRight, FileType2, Code2 } from "lucide-react";
import { Button } from "./ui/button";
import { toast } from "sonner";
import type { ScreenResult, FullTextResult } from "../lib/mockServices";
import type { AbstractDecision, FullTextDecision } from "../lib/exclusionBucketing";
import { bucketFullTextExclusionsByPaper } from "../lib/exclusionBucketing";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PrismaStep = {
  title: string;
  n: number;
  details?: Record<string, number>;
  aside?: { title: string; items: Record<string, number> };
  caption?: string;
};

type PaperRef = { paper_id: string; title: string; url?: string; source?: string };

type Counts = {
  identified: number;
  source_counts?: Record<string, number>;
  duplicates_removed: number;
  after_duplicates?: number;
  rerank_dropped?: number;
  after_rerank?: number;
  rerank_floor?: number;
  quality_excluded?: number;
  after_quality?: number;
  screened: number;
  excluded_total: number;
  exclusion_breakdown: Record<string, number>;
  ft_exclusion_breakdown?: Record<string, number>;
  included_final?: number;
};

// ---------------------------------------------------------------------------
// Editable primitives
// ---------------------------------------------------------------------------

function EditableText({ value, onSave, className = "" }: { value: string; onSave: (v: string) => void; className?: string }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  const commit = () => {
    const t = draft.trim();
    if (t) onSave(t); else setDraft(value);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        autoFocus
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => {
          if (e.key === "Enter") { e.preventDefault(); commit(); }
          if (e.key === "Escape") { setDraft(value); setEditing(false); }
        }}
        className={`bg-transparent border-b border-[#0d6b66] outline-none w-full ${className}`}
        style={{ font: "inherit" }}
      />
    );
  }
  return (
    <span onClick={() => { setDraft(value); setEditing(true); }} className={`cursor-text hover:bg-teal-50 rounded-sm px-0.5 ${className}`} title="Click to edit">
      {value}
    </span>
  );
}

function EditableNumber({ value, onSave, className = "" }: { value: number; onSave: (v: number) => void; className?: string }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value));

  const commit = () => {
    const n = parseInt(draft, 10);
    if (!isNaN(n) && n >= 0) onSave(n);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        autoFocus
        type="number"
        min="0"
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => {
          if (e.key === "Enter") { e.preventDefault(); commit(); }
          if (e.key === "Escape") { setDraft(String(value)); setEditing(false); }
        }}
        className={`bg-transparent border-b border-[#166534] outline-none text-center w-16 ${className}`}
        style={{ font: "inherit" }}
      />
    );
  }
  return (
    <span onClick={() => { setDraft(String(value)); setEditing(true); }} className={`cursor-text hover:bg-teal-50 rounded-sm px-0.5 ${className}`} title="Click to edit">
      {value.toLocaleString()}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Paper list (expandable)
// ---------------------------------------------------------------------------

function PaperList({ papers }: { papers: PaperRef[] }) {
  return (
    <div className="mt-1 ml-3 border-l border-gray-300 pl-2 space-y-0.5">
      {papers.slice(0, 50).map(p => (
        <div key={p.paper_id} className="flex items-start gap-1 text-xs text-gray-500">
          <span className="shrink-0 opacity-40">·</span>
          {p.url ? (
            <a href={p.url} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 flex items-center gap-0.5 min-w-0 group">
              <span className="truncate leading-snug">{p.title || p.paper_id}</span>
              <ExternalLink className="size-2.5 shrink-0 opacity-0 group-hover:opacity-60" />
            </a>
          ) : (
            <span className="truncate leading-snug">{p.title || p.paper_id}</span>
          )}
        </div>
      ))}
      {papers.length > 50 && <div className="text-xs text-gray-400">+{papers.length - 50} more</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// PRISMA 2020 box styles
// ---------------------------------------------------------------------------

const BOX_STYLE = "border border-[#a3c4c2] bg-white rounded text-xs leading-snug p-2.5 text-[#0f172a] w-full";
const EXCLUDED_BOX_STYLE = "border border-[#a3c4c2] bg-white rounded text-xs leading-snug p-2.5 text-[#0f172a] w-full";
const PHASE_BAR = "text-white text-[10px] font-bold tracking-widest uppercase writing-vertical flex items-center justify-center bg-[#0d6b66] w-7 shrink-0 select-none";
// Light teal panel behind each phase group, in our colour.
const SECTION_PANEL = "flex items-stretch bg-[#eef6f5] rounded-xl overflow-hidden border border-[#cfe3e1] shadow-sm";

// ---------------------------------------------------------------------------
// Exclusion box (right side, with expandable paper list)
// ---------------------------------------------------------------------------

function ExclusionBox({
  title, items, papersByReason,
  onTitleSave, onItemLabelSave, onItemCountSave,
}: {
  title: string;
  items: { key: string; label: string; count: number }[];
  papersByReason?: Record<string, PaperRef[]>;
  onTitleSave: (v: string) => void;
  onItemLabelSave: (key: string, v: string) => void;
  onItemCountSave: (key: string, v: number) => void;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (k: string) => setExpanded(p => { const n = new Set(p); n.has(k) ? n.delete(k) : n.add(k); return n; });
  const total = items.reduce((s, it) => s + it.count, 0);

  return (
    <div className={EXCLUDED_BOX_STYLE}>
      <div className="font-semibold mb-1">
        <EditableText value={title} onSave={onTitleSave} />
        {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={total} onSave={() => {}} /></span>)
      </div>
      {items.map(it => {
        const papers = papersByReason?.[it.key] ?? [];
        const isExpanded = expanded.has(it.key);
        return (
          <div key={it.key}>
            <div className="flex items-start gap-1 py-0.5">
              <button onClick={() => papers.length > 0 && toggle(it.key)} className={`shrink-0 mt-0.5 ${papers.length > 0 ? "text-gray-400 hover:text-gray-700 cursor-pointer" : "invisible"}`}>
                {isExpanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
              </button>
              <span className="flex-1 min-w-0">
                <EditableText value={it.label} onSave={v => onItemLabelSave(it.key, v)} />
              </span>
              <span className="shrink-0 font-semibold tabular-nums ml-1">(n = <EditableNumber value={it.count} onSave={v => onItemCountSave(it.key, v)} />)</span>
            </div>
            {isExpanded && papers.length > 0 && <PaperList papers={papers} />}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main PRISMA 2020 component
// ---------------------------------------------------------------------------

// ---- Export rasterisation helpers (module-level, browser-robust) ------------

/** Pull width/height off the generated SVG root so the raster canvas is sized
 *  explicitly, never relying on an SVG <img>'s intrinsic size, which WebKit
 *  reports as 0. */
function svgDimensions(svg: string): { w: number; h: number } {
  const wm = svg.match(/\bwidth="(\d+(?:\.\d+)?)"/);
  const hm = svg.match(/\bheight="(\d+(?:\.\d+)?)"/);
  return { w: wm ? parseFloat(wm[1]) : 900, h: hm ? parseFloat(hm[1]) : 700 };
}

/** UTF-8-safe data URL. Safari rasterises data: SVGs far more reliably than
 *  blob: URLs. */
function svgToDataUrl(svg: string): string {
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

/** Render an SVG string to a PNG blob on a white background at the given scale. */
function rasterizeSvg(svg: string, scale: number): Promise<{ blob: Blob; w: number; h: number }> {
  const { w, h } = svgDimensions(svg);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) { reject(new Error("no 2d context")); return; }
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(b => b ? resolve({ blob: b, w, h }) : reject(new Error("toBlob returned null")), "image/png");
      } catch (e) { reject(e as Error); }
    };
    img.onerror = () => reject(new Error("SVG image failed to load"));
    img.src = svgToDataUrl(svg);
  });
}

/** Trigger a file download; the anchor is attached to the DOM so the click
 *  fires in every browser (a detached anchor is a no-op in Firefox/Safari). */
function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function PrismaFlow({
  counts,
  abstractResults: _abstractResults,
  abstractOverrides: _abstractOverrides,
  fullTextResults,
  fullTextOverrides = {},
  inclusion = [],
  exclusion = [],
}: {
  counts: Counts;
  abstractResults?: ScreenResult[] | null;   // reserved for future use
  abstractOverrides?: Record<string, AbstractDecision>; // reserved
  fullTextResults?: FullTextResult[] | null;
  fullTextOverrides?: Record<string, FullTextDecision>;
  inclusion?: string[];
  exclusion?: string[];
}) {
  // ---- Derived counts -------------------------------------------------------
  const afterDuplicates = counts.after_duplicates ?? Math.max(0, counts.identified - counts.duplicates_removed);
  const rerankDropped = counts.rerank_dropped ?? 0;
  const screened = counts.screened;
  const abstractExcluded = counts.excluded_total;
  const assessed = Math.max(0, screened - abstractExcluded);
  const ftExcluded = counts.ft_exclusion_breakdown
    ? Object.values(counts.ft_exclusion_breakdown).reduce((s, v) => s + v, 0)
    : 0;
  const included = counts.included_final ?? 0;
  const otherSources = 0;

  // ---- Editable overrides ---------------------------------------------------
  type N = Record<string, number>;
  type S = Record<string, string>;
  const [nEdits, setNEdits] = useState<N>({});
  const [labelEdits, setLabelEdits] = useState<S>({});
  const [exportOpen, setExportOpen] = useState(false);
  const n = (key: string, fallback: number) => nEdits[key] ?? fallback;
  const lbl = (key: string, fallback: string) => labelEdits[key] ?? fallback;
  const setN = (key: string, v: number) => setNEdits(p => ({ ...p, [key]: v }));
  const setL = (key: string, v: string) => setLabelEdits(p => ({ ...p, [key]: v }));

  // ---- Exclusion breakdowns -------------------------------------------------
  const ftByReason = useMemo((): Record<string, PaperRef[]> => {
    if (!fullTextResults) return {};
    const groups = bucketFullTextExclusionsByPaper(fullTextResults, fullTextOverrides, inclusion, exclusion);
    return Object.fromEntries(Object.entries(groups).map(([reason, papers]) => [
      reason, papers.map(r => ({ paper_id: r.paper_id, title: r.Title, url: r.URL || undefined, source: r.Source })),
    ]));
  }, [fullTextResults, fullTextOverrides, inclusion, exclusion]);

  // Map exclusion breakdown keys → { key, label, count }
  const abstractExcItems = useMemo(() =>
    Object.entries(counts.exclusion_breakdown).map(([k]) => ({
      key: k, label: lbl(`abs|${k}`, k), count: nEdits[`abs|${k}`] ?? counts.exclusion_breakdown[k],
    })), [counts.exclusion_breakdown, labelEdits, nEdits]);

  const ftExcItems = useMemo(() => {
    // Prefer the live bucketing (reflects the current categoriser + overrides)
    // so reasons stay brief and aggregated without re-running screening; fall
    // back to the stored breakdown when full-text results aren't loaded.
    const live: Record<string, number> = Object.keys(ftByReason).length > 0
      ? Object.fromEntries(Object.entries(ftByReason).map(([k, ps]) => [k, ps.length]))
      : (counts.ft_exclusion_breakdown ?? {});
    return Object.entries(live).map(([k]) => ({
      key: k, label: lbl(`ft|${k}`, k), count: nEdits[`ft|${k}`] ?? live[k],
    }));
  }, [ftByReason, counts.ft_exclusion_breakdown, labelEdits, nEdits]);

  const sourceCounts = counts.source_counts ?? {};

  // ---- Export helpers -------------------------------------------------------
  const prismaData = useCallback((): SvgData => ({
    identified: n("identified", counts.identified),
    sourceCounts,
    otherSources: n("otherSources", otherSources),
    duplicatesRemoved: n("duplicatesRemoved", counts.duplicates_removed),
    afterDuplicates: n("afterDuplicates", afterDuplicates),
    screened: n("screened", screened),
    abstractExcluded: n("abstractExcluded", abstractExcluded),
    abstractExcItems,
    soughtRetrieval: n("soughtRetrieval", assessed),
    notRetrieved: n("notRetrieved", 0),
    assessed: n("assessed", assessed),
    ftExcItems,
    included: n("included", included),
    labels: labelEdits,
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [n, counts, sourceCounts, afterDuplicates, screened, abstractExcluded, abstractExcItems, assessed, ftExcItems, included, labelEdits, nEdits]);

  const buildSvgString = useCallback((): string => buildPrisma2020Svg(prismaData()), [prismaData]);

  function exportSvg() {
    try {
      const blob = new Blob([buildSvgString()], { type: "image/svg+xml;charset=utf-8" });
      triggerDownload(blob, `prisma-${new Date().toISOString().slice(0, 10)}.svg`);
      toast.success("Exported as SVG");
    } catch (e) {
      console.error("PRISMA SVG export failed:", e);
      toast.error("SVG export failed");
    }
  }

  // Word (.docx) export: embeds the polished vector diagram as a high-resolution
  // image, so the document is publication quality and converts cleanly to PDF.
  // Numbers/labels are edited inline in the app before exporting, so the figure
  // always reflects the current data.
  async function exportDocx() {
    const docx = await import("docx");
    const { Document, Packer } = docx;
    // Build the figure as native DrawingML shapes (rounded boxes, teal phase
    // pills, real connector arrows) with editable text — no "Convert to Shape".
    const doc = new Document({
      sections: [{
        properties: { page: {
          margin: { top: 900, bottom: 720, left: 720, right: 720 },  // 0.5in sides so the wide figure fits
        } },
        children: buildPrismaDocx(prismaData(), docx),
      }],
    });
    const out = await Packer.toBlob(doc);
    triggerDownload(out, `prisma-${new Date().toISOString().slice(0, 10)}.docx`);
    toast.success("Exported as an editable Word figure");
  }

  // ---- Render ---------------------------------------------------------------

  // Clean flow connectors: a solid teal line with a triangular arrowhead.
  const DOWN = "#0d6b66";
  const DownArrow = () => (
    <div className="flex flex-col items-center justify-center self-stretch min-h-full">
      <div className="w-[3px] flex-1 min-h-[2rem] rounded-full" style={{ background: DOWN }} />
      <div className="size-0 -mt-px" style={{ borderLeft: "7px solid transparent", borderRight: "7px solid transparent", borderTop: `12px solid ${DOWN}` }} />
    </div>
  );
  const RightArrow = () => (
    <div className="flex items-center self-center w-full">
      <div className="h-[3px] flex-1 rounded-full" style={{ background: DOWN }} />
      <div className="size-0 -ml-px" style={{ borderTop: "7px solid transparent", borderBottom: "7px solid transparent", borderLeft: `12px solid ${DOWN}` }} />
    </div>
  );

  return (
    <div className="py-2 space-y-3 font-sans text-xs">
      <div className="flex items-center justify-between gap-4">
        <p className="text-xs text-muted-foreground">Click any label or number to edit inline.</p>
        {/* Single Export button + a plain inline menu (no portal). Radix
            portaled menus dropped the click in Safari, so keep it inline. */}
        <div className="relative shrink-0">
          <Button size="sm" variant="outline" className="h-8" onClick={() => setExportOpen(o => !o)}>
            <Download className="size-3.5 mr-1.5" />Export<ChevronDown className={`size-3.5 ml-1.5 opacity-60 transition-transform ${exportOpen ? "rotate-180" : ""}`} />
          </Button>
          {exportOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setExportOpen(false)} aria-hidden="true" />
              <div className="absolute right-0 mt-1 z-50 w-44 rounded-lg border bg-popover text-popover-foreground shadow-md p-1">
                <div className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Choose a format</div>
                {[
                  { label: "Word (.docx)", icon: FileType2, run: () => { void exportDocx(); } },
                  { label: "SVG (vector)", icon: Code2, run: () => exportSvg() },
                ].map(opt => (
                  <button key={opt.label} onClick={() => { setExportOpen(false); opt.run(); }}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm hover:bg-muted transition-colors text-left">
                    <opt.icon className="size-4 text-muted-foreground" />{opt.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* PRISMA 2020 diagram */}
      <div className="overflow-x-auto">
        <div className="min-w-[700px] flex flex-col">

          {/* ── IDENTIFICATION ─────────────────────────────────────────── */}
          <div className={SECTION_PANEL}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Identification</div>
            <div className="flex-1 p-2.5 space-y-2">
              {/* Row 1: two source boxes side by side */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2">
                <div className={BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("dbTitle", "Studies from databases/registers")} onSave={v => setL("dbTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("identified", counts.identified)} onSave={v => setN("identified", v)} /></span>)
                  </div>
                  {Object.entries(sourceCounts).map(([src, cnt]) => (
                    <div key={src} className="ml-3 text-gray-500">
                      <EditableText value={lbl(`src|${src}`, src)} onSave={v => setL(`src|${src}`, v)} /> (n = <EditableNumber value={n(`src|${src}`, cnt)} onSave={v => setN(`src|${src}`, v)} />)
                    </div>
                  ))}
                </div>
                <div />
                <div className={BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("otherTitle", "References from other sources")} onSave={v => setL("otherTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("otherSources", otherSources)} onSave={v => setN("otherSources", v)} /></span>)
                  </div>
                  <div className="ml-3 text-gray-500">Citation searching (n = <EditableNumber value={n("citationSearch", 0)} onSave={v => setN("citationSearch", v)} />)</div>
                  <div className="ml-3 text-gray-500">Grey literature (n = <EditableNumber value={n("greyLit", 0)} onSave={v => setN("greyLit", v)} />)</div>
                </div>
              </div>

              {/* Row 2: "removed before screening" box branches off on the right.
                  The main flow arrow lives between this phase and Screening. */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 items-start">
                <div />
                <div />
                <div className={BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("removedTitle", "References removed before screening")} onSave={v => setL("removedTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("duplicatesRemoved", counts.duplicates_removed)} onSave={v => setN("duplicatesRemoved", v)} /></span>)
                  </div>
                  <div className="ml-3 text-gray-500">Duplicates identified (n = <EditableNumber value={n("dupManual", counts.duplicates_removed)} onSave={v => setN("dupManual", v)} />)</div>
                  <div className="ml-3 text-gray-500">Marked ineligible by automation (n = <EditableNumber value={n("autoIneligible", rerankDropped)} onSave={v => setN("autoIneligible", v)} />)</div>
                </div>
              </div>
            </div>
          </div>

          {/* Flow arrow between Identification and Screening. */}
          <div className="flex py-3">
            <div className="w-7 shrink-0" />
            <div className="flex-1 px-2.5">
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2"><div className="flex justify-center"><DownArrow /></div><div /><div /></div>
            </div>
          </div>

          {/* ── SCREENING ──────────────────────────────────────────────── */}
          <div className={SECTION_PANEL}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Screening</div>
            <div className="flex-1 p-2.5 space-y-0">
              {/* Row: screened → excluded */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("screenedTitle", "Studies screened")} onSave={v => setL("screenedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("screened", screened)} onSave={v => setN("screened", v)} /></span>)
                </div>
                {/* Excluded box is multi-line/tall; pin the arrow to the top and
                    align it with the single-line "Studies screened" box center
                    instead of self-centering it in the whole (tall) row. */}
                <div className="self-start w-full mt-[13px]"><RightArrow /></div>
                <div className={EXCLUDED_BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("absExcTitle", "Studies excluded")} onSave={v => setL("absExcTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("abstractExcluded", abstractExcluded)} onSave={v => setN("abstractExcluded", v)} /></span>)
                  </div>
                  {abstractExcItems.map(it => (
                    <div key={it.key} className="flex items-start gap-1 py-0.5">
                      <span className="flex-1"><EditableText value={it.label} onSave={v => setL(`abs|${it.key}`, v)} /></span>
                      <span className="shrink-0 font-semibold ml-1">(n = <EditableNumber value={it.count} onSave={v => setN(`abs|${it.key}`, v)} />)</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Down arrow, centered under the left (flow) column */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 py-2.5"><div className="flex justify-center"><DownArrow /></div><div /><div /></div>

              {/* Row: sought for retrieval → not retrieved */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("soughtTitle", "Studies sought for retrieval")} onSave={v => setL("soughtTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("soughtRetrieval", assessed)} onSave={v => setN("soughtRetrieval", v)} /></span>)
                </div>
                <RightArrow />
                <div className={EXCLUDED_BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("notRetrievedTitle", "Studies not retrieved")} onSave={v => setL("notRetrievedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("notRetrieved", 0)} onSave={v => setN("notRetrieved", v)} /></span>)
                </div>
              </div>

              {/* Down arrow, centered under the left (flow) column */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 py-2.5"><div className="flex justify-center"><DownArrow /></div><div /><div /></div>

              {/* Row: assessed for eligibility → excluded at full text.
                  Reasons + counts only, no per-study disclosure. */}
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("assessedTitle", "Studies assessed for eligibility")} onSave={v => setL("assessedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("assessed", assessed)} onSave={v => setN("assessed", v)} /></span>)
                </div>
                <RightArrow />
                <div className={EXCLUDED_BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("ftExcTitle", "Studies excluded")} onSave={v => setL("ftExcTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber
                      value={n("ftExcluded", ftExcItems.length > 0
                        ? ftExcItems.reduce((s, it) => s + it.count, 0)
                        : ftExcluded)}
                      onSave={v => setN("ftExcluded", v)} /></span>)
                  </div>
                  {ftExcItems.map(it => (
                    <div key={it.key} className="flex items-start gap-1 py-0.5">
                      <span className="flex-1"><EditableText value={it.label} onSave={v => setL(`ft|${it.key}`, v)} /></span>
                      <span className="shrink-0 font-semibold ml-1">(n = <EditableNumber value={it.count} onSave={v => setN(`ft|${it.key}`, v)} />)</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ── INCLUDED ───────────────────────────────────────────────── */}
          {/* Between-phase arrow: mirror the panel layout (invisible phase-bar
              spacer + p-2.5 + the same gap-2 grid) so it lines up exactly under
              the flow column, instead of an approximate padding. */}
          <div className="flex py-3">
            <div className="w-7 shrink-0" />
            <div className="flex-1 px-2.5">
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2"><div className="flex justify-center"><DownArrow /></div><div /><div /></div>
            </div>
          </div>
          <div className={SECTION_PANEL}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Included</div>
            <div className="flex-1 p-2.5 space-y-2">
              <div className="grid grid-cols-[5fr_48px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("includedTitle", "Studies included in review")} onSave={v => setL("includedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("included", included)} onSave={v => setN("included", v)} /></span>)
                </div>
                <div />
                <div className={`${BOX_STYLE} border-dashed`}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("ongoingTitle", "Included studies ongoing")} onSave={v => setL("ongoingTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("ongoing", 0)} onSave={v => setN("ongoing", v)} /></span>)
                  </div>
                  <div>
                    <EditableText value={lbl("awaitingTitle", "Studies awaiting classification")} onSave={v => setL("awaitingTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("awaiting", 0)} onSave={v => setN("awaiting", v)} /></span>)
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SVG export: PRISMA 2020 layout
// ---------------------------------------------------------------------------

function esc(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function _wrap(text: string, maxCh: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if (!cur) cur = w;
    else if ((cur + " " + w).length <= maxCh) cur += " " + w;
    else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  return lines;
}

type SvgData = {
  identified: number; sourceCounts: Record<string, number>;
  otherSources: number; duplicatesRemoved: number; afterDuplicates: number;
  screened: number; abstractExcluded: number;
  abstractExcItems: { key: string; label: string; count: number }[];
  soughtRetrieval: number; notRetrieved: number;
  assessed: number; ftExcItems: { key: string; label: string; count: number }[];
  included: number; labels: Record<string, string>;
};

// Text run inside a box line. Runs flow inline as <tspan>s so a label and its
// green "(n = …)" count share one editable <text> element in the exported SVG.
type Run = { text: string; bold?: boolean; color?: string };
type BoxLine = { runs: Run[]; indent?: boolean; right?: Run; size?: number };

function buildPrisma2020Svg(d: SvgData): string {
  // Geometry mirrors the on-screen card layout (SECTION_PANEL + grid-cols
  // [5fr 48px 6fr]) so the export is a 1:1 match that stays fully editable:
  // every label and number is real <text>, not baked into a raster.
  const W = 920;
  const TEAL = "#0d6b66";        // phase bar
  const PANEL = "#eef6f5";       // card background
  const CARD_BORDER = "#cfe3e1"; // card outline
  const BORDER = "#a3c4c2";      // inner box outline
  const ARROW = "#0d6b66";       // connector arrows
  const TEXT = "#0f172a";        // primary ink
  const GREEN = "#166534";       // counts
  const GRAY = "#6b7280";        // sub-source ink
  const FONT = `font-family="Calibri, Arial, sans-serif"`;

  const CARD_X = 6, CARD_W = W - 12;      // 6 … 914
  const BAR_W = 28;
  const PAD = 10;                          // card content padding (p-2.5)
  const contentX = CARD_X + BAR_W + PAD;   // 44
  const contentR = CARD_X + CARD_W - PAD;  // 904
  const contentW = contentR - contentX;    // 860
  const GAP = 8, MID = 48;
  const flex = contentW - MID - GAP * 2;   // 796
  const leftColW = Math.round((5 / 11) * flex);   // 362
  const rightColW = flex - leftColW;              // 434
  const leftColX = contentX;               // 44
  const rightColX = leftColX + leftColW + GAP + MID + GAP; // 470
  const flowCenterX = leftColX + leftColW / 2;    // centre of the flow column
  const leftColR = leftColX + leftColW;    // right edge of the flow-column box

  const LH = 17, PAD_BOX = 10;
  const bh = (nLines: number) => PAD_BOX * 2 + nLines * LH;

  const L = (k: string, def: string) => (d.labels && d.labels[k]) || def;
  const num = (v: number) => v.toLocaleString();

  const title = (label: string, n: number, size = 11.5): BoxLine =>
    ({ runs: [{ text: label + " ", bold: true, color: TEXT }, { text: `(n = ${num(n)})`, bold: true, color: GREEN }], size });
  const sub = (label: string, n: number): BoxLine =>
    ({ runs: [{ text: `${label} (n = ${num(n)})`, color: GRAY }], indent: true, size: 10.5 });
  const reason = (label: string, n: number): BoxLine =>
    ({ runs: [{ text: label, color: TEXT }], right: { text: `(n = ${num(n)})`, bold: true, color: TEXT }, size: 10.5 });

  const parts: string[] = [];

  // A white content box with inline, editable text runs.
  function box(x: number, y: number, w: number, lines: BoxLine[], dashed = false): string {
    const h = bh(lines.length);
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="#ffffff" stroke="${BORDER}" stroke-width="1.25"${dashed ? ' stroke-dasharray="5,3"' : ""}/>`;
    lines.forEach((ln, i) => {
      const baseY = y + PAD_BOX + LH * i + 12;
      const tx = x + PAD_BOX + (ln.indent ? 12 : 0);
      const size = ln.size ?? 11;
      const runs = ln.runs.map(r => `<tspan fill="${r.color ?? TEXT}" font-weight="${r.bold ? "bold" : "normal"}">${esc(r.text)}</tspan>`).join("");
      s += `<text x="${tx}" y="${baseY}" ${FONT} font-size="${size}">${runs}</text>`;
      if (ln.right) s += `<text x="${x + w - PAD_BOX}" y="${baseY}" ${FONT} font-size="${size}" text-anchor="end" font-weight="${ln.right.bold ? "bold" : "normal"}" fill="${ln.right.color ?? TEXT}">${esc(ln.right.text)}</text>`;
    });
    return s;
  }

  // Card shell: rounded panel + border + left teal phase bar with a rotated label.
  function card(y: number, h: number, label: string): string {
    const r = 14;
    let s = `<rect x="${CARD_X}" y="${y}" width="${CARD_W}" height="${h}" rx="${r}" fill="${PANEL}" stroke="${CARD_BORDER}" stroke-width="1.25"/>`;
    const bx = CARD_X + 1, by = y + 1, bhgt = h - 2, bw = BAR_W - 1, br = r - 1;
    s += `<path d="M${bx + bw},${by} L${bx + br},${by} Q${bx},${by} ${bx},${by + br} L${bx},${by + bhgt - br} Q${bx},${by + bhgt} ${bx + br},${by + bhgt} L${bx + bw},${by + bhgt} Z" fill="${TEAL}"/>`;
    const tcx = CARD_X + BAR_W / 2, tcy = y + h / 2;
    s += `<text x="${tcx}" y="${tcy}" ${FONT} font-size="10.5" font-weight="bold" letter-spacing="1.5" fill="#ffffff" text-anchor="middle" dominant-baseline="central" transform="rotate(-90 ${tcx} ${tcy})">${esc(label.toUpperCase())}</text>`;
    return s;
  }

  const arrowDown = (cx: number, y1: number, y2: number) =>
    `<line x1="${cx}" y1="${y1}" x2="${cx}" y2="${y2}" stroke="${ARROW}" stroke-width="3" stroke-linecap="round" marker-end="url(#arr)"/>`;
  const arrowRight = (yc: number, x1: number, x2: number) =>
    `<line x1="${x1}" y1="${yc}" x2="${x2}" y2="${yc}" stroke="${ARROW}" stroke-width="3" stroke-linecap="round" marker-end="url(#arr)"/>`;

  const GAP_CARD = 40;  // vertical band + arrow between phase cards
  const GAP_ROW = 34;   // vertical band + arrow between rows inside a card

  let y = 20;

  // Title, centered above the flow.
  parts.push(`<text x="${W / 2}" y="${y + 16}" ${FONT} font-size="17" font-weight="bold" fill="${GREEN}" text-anchor="middle">PRISMA 2020 Flow Diagram</text>`);
  y += 46;

  // ── IDENTIFICATION ──────────────────────────────────────────────────────
  const dbLines: BoxLine[] = [
    title(L("dbTitle", "Studies from databases/registers"), d.identified),
    ...Object.entries(d.sourceCounts).map(([k, v]) => sub(L(`src|${k}`, k), v)),
  ];
  const otherLines: BoxLine[] = [
    title(L("otherTitle", "References from other sources"), d.otherSources),
    sub("Citation searching", 0),
    sub("Grey literature", 0),
  ];
  const removedLines: BoxLine[] = [
    title(L("removedTitle", "References removed before screening"), d.duplicatesRemoved),
    sub("Duplicate records", d.duplicatesRemoved),
    sub("Marked ineligible by automation", 0),
  ];
  const idRow1H = Math.max(bh(dbLines.length), bh(otherLines.length));
  const idRow2H = bh(removedLines.length);
  const idCardH = PAD + idRow1H + 8 + idRow2H + PAD;
  const idTop = y;
  parts.push(card(idTop, idCardH, "Identification"));
  parts.push(box(leftColX, idTop + PAD, leftColW, dbLines));
  parts.push(box(rightColX, idTop + PAD, rightColW, otherLines));
  parts.push(box(rightColX, idTop + PAD + idRow1H + 8, rightColW, removedLines));
  y = idTop + idCardH;

  // Arrow band between Identification and Screening.
  parts.push(arrowDown(flowCenterX, y + 8, y + GAP_CARD - 2));
  y += GAP_CARD;

  // ── SCREENING ───────────────────────────────────────────────────────────
  const screenedLines: BoxLine[] = [title(L("screenedTitle", "Studies screened"), d.screened)];
  const absExcLines: BoxLine[] = [
    title(L("absExcTitle", "Studies excluded"), d.abstractExcluded),
    ...d.abstractExcItems.map(it => reason(it.label, it.count)),
  ];
  const soughtLines: BoxLine[] = [title(L("soughtTitle", "Studies sought for retrieval"), d.soughtRetrieval)];
  const notRetLines: BoxLine[] = [title(L("notRetrievedTitle", "Studies not retrieved"), d.notRetrieved)];
  const ftTotal = d.ftExcItems.reduce((s, it) => s + it.count, 0);
  const assessedLines: BoxLine[] = [title(L("assessedTitle", "Studies assessed for eligibility"), d.assessed)];
  const ftExcLines: BoxLine[] = [
    title(L("ftExcTitle", "Studies excluded"), ftTotal),
    ...d.ftExcItems.map(it => reason(it.label, it.count)),
  ];

  const scRowAH = Math.max(bh(screenedLines.length), bh(absExcLines.length));
  const scRowBH = Math.max(bh(soughtLines.length), bh(notRetLines.length));
  const scRowCH = Math.max(bh(assessedLines.length), bh(ftExcLines.length));
  const scCardH = PAD + scRowAH + GAP_ROW + scRowBH + GAP_ROW + scRowCH + PAD;
  const scTop = y;
  parts.push(card(scTop, scCardH, "Screening"));

  let ry = scTop + PAD;
  // Row A: screened → excluded
  parts.push(box(leftColX, ry, leftColW, screenedLines));
  parts.push(box(rightColX, ry, rightColW, absExcLines));
  parts.push(arrowRight(ry + bh(1) / 2, leftColR, rightColX));
  ry += scRowAH;
  parts.push(arrowDown(flowCenterX, ry + 4, ry + GAP_ROW - 2));
  ry += GAP_ROW;
  // Row B: sought → not retrieved
  parts.push(box(leftColX, ry, leftColW, soughtLines));
  parts.push(box(rightColX, ry, rightColW, notRetLines));
  parts.push(arrowRight(ry + bh(1) / 2, leftColR, rightColX));
  ry += scRowBH;
  parts.push(arrowDown(flowCenterX, ry + 4, ry + GAP_ROW - 2));
  ry += GAP_ROW;
  // Row C: assessed → excluded at full text
  parts.push(box(leftColX, ry, leftColW, assessedLines));
  parts.push(box(rightColX, ry, rightColW, ftExcLines));
  parts.push(arrowRight(ry + bh(1) / 2, leftColR, rightColX));
  y = scTop + scCardH;

  // Arrow band between Screening and Included.
  parts.push(arrowDown(flowCenterX, y + 8, y + GAP_CARD - 2));
  y += GAP_CARD;

  // ── INCLUDED ──────────────────────────────────────────────────────────
  const includedLines: BoxLine[] = [title(L("includedTitle", "Studies included in review"), d.included)];
  const ongoingLines: BoxLine[] = [
    title(L("ongoingTitle", "Included studies ongoing"), 0),
    title(L("awaitingTitle", "Studies awaiting classification"), 0),
  ];
  const inRowH = Math.max(bh(includedLines.length), bh(ongoingLines.length));
  const inCardH = PAD + inRowH + PAD;
  const inTop = y;
  parts.push(card(inTop, inCardH, "Included"));
  parts.push(box(leftColX, inTop + PAD, leftColW, includedLines));
  parts.push(box(rightColX, inTop + PAD, rightColW, ongoingLines, true));
  y = inTop + inCardH + 12;

  // Assemble with the final height baked into the root <svg> (Safari needs an
  // explicit height or it reports naturalHeight=0 when rasterizing for PNG/Word).
  const H = y;
  const head =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" ${FONT}>` +
    `<defs><marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${ARROW}"/></marker></defs>` +
    `<rect width="${W}" height="${H}" fill="#ffffff"/>`;
  return head + parts.join("") + `</svg>`;
}

// ---------------------------------------------------------------------------
// Word export: a NATIVE, directly-editable PRISMA table (Evidence Engine style)
// ---------------------------------------------------------------------------
// Rendered as a borderless Word table so every label and number is real,
// directly-editable text (no "Convert to Shape"): teal phase bars with vertical
// labels, light panels behind each phase, white bordered boxes, green counts,
// and thin connector arrows. Reliable in Word (crisp text, no shape artifacts).
function buildPrismaDocx(d: SvgData, docx: typeof import("docx")): (InstanceType<typeof docx.Paragraph> | InstanceType<typeof docx.Table>)[] {
  const {
    Paragraph, TextRun, Table, TableRow, TableCell, WidthType, TableLayoutType,
    BorderStyle, ShadingType, TextDirection, VerticalAlign, AlignmentType,
    HeightRule, TabStopType,
  } = docx;

  const TEAL = "0D6B66", TEALDK = "0A544F", PANEL = "F0F7F6", BORDER = "CFE3E1";
  const TITLE = "0F172A", COUNT = "166534", SUB = "64748B", ARROW = "5F8A86";
  const FONT = "Calibri";
  const NBSP = String.fromCharCode(160);
  const NN = (n: number) => `(n${NBSP}=${NBSP}${n.toLocaleString()})`;
  const L = (k: string, def: string) => (d.labels && d.labels[k]) || def;

  const COL = { bar: 560, padL: 250, left: 4450, mid: 830, right: 4450, padR: 260 };
  const COLS = [COL.bar, COL.padL, COL.left, COL.mid, COL.right, COL.padR];
  const TOTAL = COLS.reduce((a, b) => a + b, 0);
  const rightInner = COL.right - 360;

  const NB = { style: BorderStyle.NONE, size: 0, color: "auto" };
  const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
  const boxBorder = (dashed = false) => {
    const b = { style: dashed ? BorderStyle.DASHED : BorderStyle.SINGLE, size: 4, color: BORDER };
    return { top: b, bottom: b, left: b, right: b };
  };

  type Line = InstanceType<typeof Paragraph>;
  const titleLine = (label: string, n: number): Line => new Paragraph({
    spacing: { after: 24, line: 264, lineRule: "auto" },
    children: [
      new TextRun({ text: label + " ", bold: true, color: TITLE, size: 23, font: FONT }),
      new TextRun({ text: NN(n), bold: true, color: COUNT, size: 23, font: FONT }),
    ],
  });
  const subLine = (label: string, n: number): Line => new Paragraph({
    spacing: { after: 0, line: 250, lineRule: "auto" },
    indent: { left: 230 },
    children: [
      new TextRun({ text: "–  ", color: ARROW, size: 20, font: FONT }),
      new TextRun({ text: `${label} ${NN(n)}`, color: SUB, size: 20, font: FONT }),
    ],
  });
  const reasonLine = (label: string, n: number): Line => new Paragraph({
    spacing: { after: 0, line: 250, lineRule: "auto" },
    tabStops: [{ type: TabStopType.RIGHT, position: rightInner }],
    children: [
      new TextRun({ text: "–  ", color: ARROW, size: 20, font: FONT }),
      new TextRun({ text: label, color: SUB, size: 20, font: FONT }),
      new TextRun({ text: `\t${NN(n)}`, bold: true, color: SUB, size: 20, font: FONT }),
    ],
  });

  const boxCell = (width: number, paras: Line[], dashed = false) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: boxBorder(dashed),
    shading: { fill: "FFFFFF", type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 170, bottom: 170, left: 210, right: 210 },
    verticalAlign: VerticalAlign.CENTER, children: paras,
  });
  const panelCell = (width: number) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: noBorders,
    shading: { fill: PANEL, type: ShadingType.CLEAR, color: "auto" },
    children: [new Paragraph({ children: [] })],
  });
  const whiteCell = (width: number) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: noBorders,
    children: [new Paragraph({ children: [] })],
  });
  const arrowCell = (width: number, glyph: string, size: number, onPanel: boolean) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: noBorders,
    ...(onPanel ? { shading: { fill: PANEL, type: ShadingType.CLEAR, color: "auto" } } : {}),
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 },
      children: [new TextRun({ text: glyph, color: ARROW, size, font: FONT })] })],
  });
  const phaseBarCell = (label: string, rowSpan: number) => new TableCell({
    width: { size: COL.bar, type: WidthType.DXA }, rowSpan, borders: noBorders,
    shading: { fill: TEAL, type: ShadingType.CLEAR, color: "auto" },
    textDirection: TextDirection.BOTTOM_TO_TOP_LEFT_TO_RIGHT, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: label.toUpperCase(), bold: true, color: "FFFFFF", size: 19, font: FONT, characterSpacing: 40 })] })],
  });
  const H = (v: number) => ({ value: v, rule: HeightRule.ATLEAST });
  const row = (height: number, children: InstanceType<typeof TableCell>[]) => new TableRow({ height: H(height), children });

  const ftTotal = d.ftExcItems.reduce((s, it) => s + it.count, 0);
  const spacerH = 150;
  const rows: InstanceType<typeof TableRow>[] = [];

  // IDENTIFICATION (rowSpan 5)
  rows.push(row(spacerH, [phaseBarCell("Identification", 5),
    panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(1240, [panelCell(COL.padL),
    boxCell(COL.left, [titleLine(L("dbTitle", "Studies from databases/registers"), d.identified),
      ...Object.entries(d.sourceCounts).map(([k, v]) => subLine(L(`src|${k}`, k), v))]),
    panelCell(COL.mid),
    boxCell(COL.right, [titleLine(L("otherTitle", "References from other sources"), d.otherSources),
      subLine("Citation searching", 0), subLine("Grey literature", 0)]),
    panelCell(COL.padR)]));
  rows.push(row(170, [panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(720, [panelCell(COL.padL), panelCell(COL.left),
    arrowCell(COL.mid, "→", 34, true),
    boxCell(COL.right, [titleLine(L("removedTitle", "References removed before screening"), d.duplicatesRemoved),
      subLine("Duplicate records", d.duplicatesRemoved), subLine("Marked ineligible by automation", 0)]),
    panelCell(COL.padR)]));
  rows.push(row(spacerH, [panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));

  // between-phase arrow (white)
  rows.push(row(560, [whiteCell(COL.bar), whiteCell(COL.padL), arrowCell(COL.left, "↓", 44, false), whiteCell(COL.mid), whiteCell(COL.right), whiteCell(COL.padR)]));

  // SCREENING (rowSpan 7)
  rows.push(row(spacerH, [phaseBarCell("Screening", 7),
    panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(800, [panelCell(COL.padL),
    boxCell(COL.left, [titleLine(L("screenedTitle", "Studies screened"), d.screened)]),
    arrowCell(COL.mid, "→", 34, true),
    boxCell(COL.right, [titleLine(L("absExcTitle", "Studies excluded"), d.abstractExcluded),
      ...d.abstractExcItems.map(it => reasonLine(it.label, it.count))]),
    panelCell(COL.padR)]));
  rows.push(row(420, [panelCell(COL.padL), arrowCell(COL.left, "↓", 44, true), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(680, [panelCell(COL.padL),
    boxCell(COL.left, [titleLine(L("soughtTitle", "Studies sought for retrieval"), d.soughtRetrieval)]),
    arrowCell(COL.mid, "→", 34, true),
    boxCell(COL.right, [titleLine(L("notRetrievedTitle", "Studies not retrieved"), d.notRetrieved)]),
    panelCell(COL.padR)]));
  rows.push(row(420, [panelCell(COL.padL), arrowCell(COL.left, "↓", 44, true), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(800, [panelCell(COL.padL),
    boxCell(COL.left, [titleLine(L("assessedTitle", "Studies assessed for eligibility"), d.assessed)]),
    arrowCell(COL.mid, "→", 34, true),
    boxCell(COL.right, [titleLine(L("ftExcTitle", "Studies excluded"), ftTotal),
      ...d.ftExcItems.map(it => reasonLine(it.label, it.count))]),
    panelCell(COL.padR)]));
  rows.push(row(spacerH, [panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));

  // between-phase arrow (white)
  rows.push(row(560, [whiteCell(COL.bar), whiteCell(COL.padL), arrowCell(COL.left, "↓", 44, false), whiteCell(COL.mid), whiteCell(COL.right), whiteCell(COL.padR)]));

  // INCLUDED (rowSpan 3)
  rows.push(row(spacerH, [phaseBarCell("Included", 3),
    panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));
  rows.push(row(800, [panelCell(COL.padL),
    boxCell(COL.left, [titleLine(L("includedTitle", "Studies included in review"), d.included)]),
    panelCell(COL.mid),
    boxCell(COL.right, [titleLine(L("ongoingTitle", "Included studies ongoing"), 0),
      titleLine(L("awaitingTitle", "Studies awaiting classification"), 0)], true),
    panelCell(COL.padR)]));
  rows.push(row(spacerH, [panelCell(COL.padL), panelCell(COL.left), panelCell(COL.mid), panelCell(COL.right), panelCell(COL.padR)]));

  const table = new Table({
    columnWidths: COLS,
    width: { size: TOTAL, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    borders: { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
    rows,
  });

  const eyebrow = new Paragraph({
    spacing: { after: 20 },
    children: [new TextRun({ text: "EVIDENCE ENGINE", bold: true, color: TEALDK, size: 15, font: FONT, characterSpacing: 60 })],
  });
  const heading = new Paragraph({
    spacing: { after: 90 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: TEAL, space: 8 } },
    children: [new TextRun({ text: "PRISMA 2020 Flow Diagram", bold: true, color: TITLE, size: 30, font: FONT })],
  });
  const gap = new Paragraph({ spacing: { after: 120 }, children: [] });
  return [eyebrow, heading, gap, table];
}
