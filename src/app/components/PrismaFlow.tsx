import { useState, useMemo, useCallback } from "react";
import { Download, ChevronRight, ChevronDown, ExternalLink } from "lucide-react";
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
const PHASE_BAR = "text-white text-[10px] font-bold tracking-widest uppercase writing-vertical flex items-center justify-center bg-[#0d6b66] px-2 select-none";
// Light teal panel behind each phase group — the Covidence look, in our colour.
const SECTION_PANEL = "flex items-stretch rounded-lg bg-[#eef6f5] overflow-hidden";

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
 *  explicitly — never relying on an SVG <img>'s intrinsic size, which WebKit
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
  const buildSvgString = useCallback((): string => {
    return buildPrisma2020Svg({
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
    });
  }, [n, counts, sourceCounts, afterDuplicates, screened, abstractExcluded, abstractExcItems, assessed, ftExcItems, included, labelEdits, nEdits]);

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

  async function exportPng() {
    try {
      const { blob } = await rasterizeSvg(buildSvgString(), 2);
      triggerDownload(blob, `prisma-${new Date().toISOString().slice(0, 10)}.png`);
      toast.success("Exported as PNG (2×)");
    } catch (e) {
      console.error("PRISMA PNG export failed:", e);
      toast.error("PNG export failed");
    }
  }

  // Build a NATIVE, editable Word document that mirrors the on-screen Covidence
  // layout: teal vertical phase bands, a [band | main | gap | excluded] grid,
  // bordered white boxes, and right-aligned reason counts — every value real,
  // editable text rather than a flat image.
  async function exportDocx() {
    const {
      Document, Packer, Paragraph, TextRun, AlignmentType,
      Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
      VerticalAlign, TextDirection, TabStopType, HeightRule, TableLayoutType,
    } = await import("docx");

    const FONT = "Calibri";
    const TEAL = "0D6B66", GREEN = "166534", SUB = "64748B", BORD = "A3C4C2", GREY = "94A3B8";
    const W_BAND = 420, W_MAIN = 4200, W_GAP = 300, W_EXC = 4080;   // twips
    const INNER = W_BAND + W_MAIN + W_GAP + W_EXC;                  // 9000 — fits inside the panel
    const PAGE = 9360;                                             // Letter content width

    const edge = { style: BorderStyle.SINGLE, size: 6, color: BORD };
    const dash = { style: BorderStyle.DASHED, size: 6, color: BORD };
    const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
    const boxB = { top: edge, bottom: edge, left: edge, right: edge };
    const dashB = { top: dash, bottom: dash, left: dash, right: dash };
    const noB = { top: none, bottom: none, left: none, right: none };
    const margins = { top: 80, bottom: 80, left: 140, right: 140 };

    const run = (t: string, o: { bold?: boolean; italics?: boolean; color?: string; size?: number } = {}) =>
      new TextRun({ text: t, bold: o.bold, italics: o.italics, color: o.color, size: o.size ?? 20, font: FONT });
    const para = (children: any[], o: { align?: any; indentLeft?: number; tabStops?: any; after?: number } = {}) =>
      new Paragraph({ alignment: o.align, indent: o.indentLeft ? { left: o.indentLeft } : undefined, tabStops: o.tabStops, spacing: { after: o.after ?? 0 }, children });

    // Bold black title with a green "(n = X)".
    const titlePara = (title: string, count: number) =>
      para([run(title + " ", { bold: true }), run(`(n = ${count})`, { bold: true, color: GREEN })]);
    const subPara = (text: string) => para([run(text, { color: SUB, size: 18 })], { indentLeft: 200 });
    // Exclusion reason with the count right-aligned (tab stop), like the screen.
    const reasonPara = (label: string, count: number) =>
      para([run(label, { size: 18 }), run(`\t(n = ${count})`, { size: 18, bold: true })],
        { tabStops: [{ type: TabStopType.RIGHT, position: W_EXC - 360 }] });

    const boxCell = (children: any[], o: { w: number; fill?: string; dashed?: boolean; rowSpan?: number } ) => new TableCell({
      width: { size: o.w, type: WidthType.DXA },
      rowSpan: o.rowSpan, verticalAlign: VerticalAlign.CENTER, margins,
      borders: o.dashed ? dashB : boxB,
      shading: { type: ShadingType.CLEAR, color: "auto", fill: o.fill ?? "FFFFFF" },
      children,
    });
    const bandCell = (label: string, rowSpan: number) => new TableCell({
      width: { size: W_BAND, type: WidthType.DXA }, rowSpan,
      verticalAlign: VerticalAlign.CENTER, margins: { top: 40, bottom: 40, left: 20, right: 20 },
      borders: noB, shading: { type: ShadingType.CLEAR, color: "auto", fill: TEAL },
      textDirection: TextDirection.BOTTOM_TO_TOP_LEFT_TO_RIGHT,
      children: [para([run(label.toUpperCase(), { bold: true, color: "FFFFFF", size: 16 })], { align: AlignmentType.CENTER })],
    });
    const plain = (children: any[], w: number) => new TableCell({
      width: { size: w, type: WidthType.DXA }, borders: noB, verticalAlign: VerticalAlign.CENTER, margins, children,
    });
    const gapDown = (w: number) => plain([para([run("↓", { color: GREY, size: 28 })], { align: AlignmentType.CENTER })], w);
    const gapRight = () => plain([para([run("→", { color: GREY, size: 24 })], { align: AlignmentType.CENTER })], W_GAP);
    const gapEmpty = (w: number) => plain([para([])], w);

    const noTableB = { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none };
    const grid = (rows: any[]) => new Table({
      width: { size: INNER, type: WidthType.DXA },
      layout: TableLayoutType.FIXED,                 // respect columnWidths, no autofit
      indent: { size: 0, type: WidthType.DXA },
      columnWidths: [W_BAND, W_MAIN, W_GAP, W_EXC],
      borders: noTableB,
      rows,
    });
    // Wrap a phase grid in a light-teal panel (Covidence look): the inner grid is
    // left-aligned (band flush left) and narrower than the panel, so the tint
    // shows above/below/right of the white boxes. `clear` = transparent (used to
    // align the between-phase arrow with the panels).
    const panel = (inner: any, clear = false) => new Table({
      width: { size: PAGE, type: WidthType.DXA },
      layout: TableLayoutType.FIXED,
      indent: { size: 0, type: WidthType.DXA },
      columnWidths: [PAGE],
      borders: noTableB,
      rows: [new TableRow({ children: [new TableCell({
        borders: noB,
        shading: clear ? undefined : { type: ShadingType.CLEAR, color: "auto", fill: "EEF6F5" },
        margins: { top: 120, bottom: 120, left: 0, right: 0 },
        children: [inner],
      })] })],
    });
    const spacer = () => new Paragraph({ spacing: { after: 100 }, children: [] });

    // ── Values (honour inline edits via n()/lbl()) ───────────────────────────
    const identified = n("identified", counts.identified);
    const srcSubs = Object.entries(sourceCounts).map(([src, cnt]) =>
      subPara(`${lbl(`src|${src}`, src)} (n = ${n(`src|${src}`, cnt)})`));
    const ftExclTotal = ftExcItems.length > 0 ? ftExcItems.reduce((a, it) => a + it.count, 0) : n("ftExcluded", ftExcluded);

    const children: any[] = [
      para([run("PRISMA 2020 Flow Diagram", { bold: true, color: GREEN, size: 30 })], { align: AlignmentType.CENTER, after: 200 }),

      // ── Identification ──────────────────────────────────────────────────
      panel(grid([
        new TableRow({ children: [
          bandCell("Identification", 2),
          boxCell([titlePara(lbl("dbTitle", "Studies from databases/registers"), identified), ...srcSubs], { w: W_MAIN }),
          gapEmpty(W_GAP),
          boxCell([
            titlePara(lbl("otherTitle", "References from other sources"), n("otherSources", otherSources)),
            subPara(`Citation searching (n = ${n("citationSearch", 0)})`),
            subPara(`Grey literature (n = ${n("greyLit", 0)})`),
          ], { w: W_EXC }),
        ] }),
        new TableRow({ children: [
          gapDown(W_MAIN),
          gapEmpty(W_GAP),
          boxCell([
            titlePara(lbl("removedTitle", "References removed before screening"), n("duplicatesRemoved", counts.duplicates_removed)),
            subPara(`Duplicates identified (n = ${n("dupManual", counts.duplicates_removed)})`),
            subPara(`Marked ineligible by automation (n = ${n("autoIneligible", rerankDropped)})`),
          ], { w: W_EXC }),
        ] }),
      ])),
      spacer(),

      // ── Screening ───────────────────────────────────────────────────────
      panel(grid([
        new TableRow({ children: [
          bandCell("Screening", 5),
          boxCell([titlePara(lbl("screenedTitle", "Studies screened"), n("screened", screened))], { w: W_MAIN }),
          gapRight(),
          boxCell([
            titlePara(lbl("absExcTitle", "Studies excluded"), n("abstractExcluded", abstractExcluded)),
            ...abstractExcItems.map(it => reasonPara(it.label, it.count)),
          ], { w: W_EXC }),
        ] }),
        new TableRow({ children: [gapDown(W_MAIN), gapEmpty(W_GAP), gapEmpty(W_EXC)] }),
        new TableRow({ children: [
          boxCell([titlePara(lbl("soughtTitle", "Studies sought for retrieval"), n("soughtRetrieval", assessed))], { w: W_MAIN }),
          gapRight(),
          boxCell([titlePara(lbl("notRetrievedTitle", "Studies not retrieved"), n("notRetrieved", 0))], { w: W_EXC }),
        ] }),
        new TableRow({ children: [gapDown(W_MAIN), gapEmpty(W_GAP), gapEmpty(W_EXC)] }),
        new TableRow({ children: [
          boxCell([titlePara(lbl("assessedTitle", "Studies assessed for eligibility"), n("assessed", assessed))], { w: W_MAIN }),
          gapRight(),
          boxCell([
            titlePara(lbl("ftExcTitle", "Studies excluded"), ftExclTotal),
            ...ftExcItems.map(it => reasonPara(it.label, it.count)),
          ], { w: W_EXC }),
        ] }),
      ])),

      // arrow between Screening and Included, aligned under the main column
      panel(grid([new TableRow({ children: [gapEmpty(W_BAND), gapDown(W_MAIN), gapEmpty(W_GAP), gapEmpty(W_EXC)] })]), true),

      // ── Included ────────────────────────────────────────────────────────
      panel(grid([
        // Min height so the vertical "INCLUDED" band fits on one line.
        new TableRow({ height: { value: 1150, rule: HeightRule.ATLEAST }, children: [
          bandCell("Included", 1),
          boxCell([titlePara(lbl("includedTitle", "Studies included in review"), n("included", included))], { w: W_MAIN }),
          gapEmpty(W_GAP),
          boxCell([
            titlePara(lbl("ongoingTitle", "Included studies ongoing"), n("ongoing", 0)),
            titlePara(lbl("awaitingTitle", "Studies awaiting classification"), n("awaiting", 0)),
          ], { w: W_EXC, dashed: true }),
        ] }),
      ])),

      spacer(),
      para([run(`Generated by Evidence Engine · ${new Date().toISOString().slice(0, 10)}`, { italics: true, color: GREY, size: 16 })], { align: AlignmentType.RIGHT }),
    ];

    const doc = new Document({ sections: [{ properties: {}, children }] });
    const out = await Packer.toBlob(doc);
    triggerDownload(out, `prisma-${new Date().toISOString().slice(0, 10)}.docx`);
    toast.success("Exported as editable Word document");
  }

  // ---- Render ---------------------------------------------------------------

  const ARROW = <div className="flex justify-center text-gray-400 text-lg leading-none">↓</div>;
  const HARROW = <div className="flex items-center justify-center text-gray-400 text-sm">→</div>;

  return (
    <div className="py-2 space-y-3 font-sans text-xs">
      <div className="flex items-center justify-between gap-4">
        <p className="text-xs text-muted-foreground">Click any label or number to edit inline.</p>
        {/* Plain buttons — Radix dropdown items didn't deliver the click in Safari. */}
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <Download className="size-3" />Export
          </span>
          <Button size="sm" variant="outline" className="h-7 px-2.5" onClick={() => exportSvg()}>SVG</Button>
          <Button size="sm" variant="outline" className="h-7 px-2.5" onClick={() => { void exportPng(); }}>PNG</Button>
          <Button size="sm" variant="outline" className="h-7 px-2.5" onClick={() => { void exportDocx(); }}>Word</Button>
        </div>
      </div>

      {/* PRISMA 2020 diagram */}
      <div className="overflow-x-auto">
        <div className="min-w-[700px]">

          {/* ── IDENTIFICATION ─────────────────────────────────────────── */}
          <div className={SECTION_PANEL}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Identification</div>
            <div className="flex-1 p-2.5 space-y-2">
              {/* Row 1: two source boxes side by side */}
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2">
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

              {/* Row 2: down arrow under left, removed box on right */}
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2 items-start">
                <div className="flex justify-center py-1 text-gray-400 text-base">↓</div>
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

          {/* ── SCREENING ──────────────────────────────────────────────── */}
          <div className={`mt-2 ${SECTION_PANEL}`}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Screening</div>
            <div className="flex-1 p-2.5 space-y-0">
              {/* Row: screened → excluded */}
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("screenedTitle", "Studies screened")} onSave={v => setL("screenedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("screened", screened)} onSave={v => setN("screened", v)} /></span>)
                </div>
                <div className="flex items-start justify-center pt-2.5 text-gray-400 text-sm">→</div>
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
              <div className="grid grid-cols-[5fr_16px_6fr] py-0.5">{ARROW}</div>

              {/* Row: sought for retrieval → not retrieved */}
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("soughtTitle", "Studies sought for retrieval")} onSave={v => setL("soughtTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("soughtRetrieval", assessed)} onSave={v => setN("soughtRetrieval", v)} /></span>)
                </div>
                {HARROW}
                <div className={EXCLUDED_BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("notRetrievedTitle", "Studies not retrieved")} onSave={v => setL("notRetrievedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("notRetrieved", 0)} onSave={v => setN("notRetrieved", v)} /></span>)
                </div>
              </div>

              {/* Down arrow, centered under the left (flow) column */}
              <div className="grid grid-cols-[5fr_16px_6fr] py-0.5">{ARROW}</div>

              {/* Row: assessed for eligibility → excluded at full text.
                  Reasons + counts only — no per-study disclosure. */}
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2 items-start">
                <div className={BOX_STYLE}>
                  <span className="font-semibold">
                    <EditableText value={lbl("assessedTitle", "Studies assessed for eligibility")} onSave={v => setL("assessedTitle", v)} />
                  </span>
                  {" "}(<span className="font-bold text-[#166534]">n = <EditableNumber value={n("assessed", assessed)} onSave={v => setN("assessed", v)} /></span>)
                </div>
                <div className="flex items-start justify-center pt-2.5 text-gray-400 text-sm">→</div>
                <div className={EXCLUDED_BOX_STYLE}>
                  <div className="font-semibold mb-0.5">
                    <EditableText value={lbl("ftExcTitle", "Studies excluded")} onSave={v => setL("ftExcTitle", v)} />
                    {" "}(<span className="font-bold text-[#166534]">n = {ftExcItems.length > 0
                      ? ftExcItems.reduce((s, it) => s + it.count, 0)
                      : <EditableNumber value={n("ftExcluded", ftExcluded)} onSave={v => setN("ftExcluded", v)} />}</span>)
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
          <div className="pl-[2.6rem] pr-2.5"><div className="grid grid-cols-[5fr_16px_6fr]">{ARROW}<div /><div /></div></div>
          <div className={`mt-1 ${SECTION_PANEL}`}>
            <div className={`${PHASE_BAR} self-stretch`} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Included</div>
            <div className="flex-1 p-2.5 space-y-2">
              <div className="grid grid-cols-[5fr_16px_6fr] gap-2 items-start">
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
// SVG export — PRISMA 2020 layout
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

function buildPrisma2020Svg(d: SvgData): string {
  const W = 900;
  const NAVY = "#166534";
  const LIGHT = "#f0fdf4";
  const BORDER = "#86efac";
  const TEXT = "#0f172a";
  const GRAY = "#475569";
  const FONT = `font-family="Calibri, Arial, sans-serif"`;

  const BAR_W = 26;
  const COL_GAP = 14;
  const LEFT = BAR_W + COL_GAP + 10;
  const RIGHT_COL = (W - LEFT) / 2 + LEFT;
  const COL_W = (W - LEFT - COL_GAP * 3) / 2;
  const BOX_PAD = 8;
  const LH = 16;

  let y = 20;
  const parts: string[] = [];
  const phaseBars: { x: number; y: number; h: number; label: string }[] = [];

  const boxRect = (x: number, bY: number, w: number, h: number, fill = "#fff", dashed = false) =>
    `<rect x="${x}" y="${bY}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${BORDER}" stroke-width="1.5"${dashed ? ' stroke-dasharray="5,3"' : ""}/>`;

  const txt = (x: number, tY: number, text: string, opts: { bold?: boolean; size?: number; color?: string; anchor?: string } = {}) =>
    `<text x="${x}" y="${tY}" ${FONT} font-size="${opts.size ?? 12}" font-weight="${opts.bold ? "bold" : "normal"}" fill="${opts.color ?? TEXT}" text-anchor="${opts.anchor ?? "start"}">${esc(text)}</text>`;

  function textBox(x: number, bY: number, w: number, lines: { text: string; bold?: boolean; indent?: boolean; color?: string }[], fill = "#fff", dashed = false) {
    let lineY = bY + BOX_PAD + LH;
    const totalH = BOX_PAD * 2 + lines.length * LH;
    const rects = boxRect(x, bY, w, totalH, fill, dashed);
    const texts = lines.map(l => {
      const ix = x + BOX_PAD + (l.indent ? 12 : 0);
      const t = txt(ix, lineY, l.text, { bold: l.bold, color: l.color, size: l.bold ? 12 : 11 });
      lineY += LH;
      return t;
    }).join("");
    return { svg: rects + texts, h: totalH };
  }

  function arrow(x1: number, y1: number, x2: number, y2: number, horizontal = false) {
    if (horizontal) {
      return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${BORDER}" stroke-width="1.5" marker-end="url(#arr)"/>`;
    }
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${BORDER}" stroke-width="1.5" marker-end="url(#arr)"/>`;
  }

  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} 900" width="${W}" height="900" ${FONT}>`,
    `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${BORDER}"/></marker></defs>`,
    `<rect width="${W}" height="900" fill="#f8fafc"/>`,
  );

  // ── IDENTIFICATION ──────────────────────────────────────────────────────
  const identStart = y;

  // Two top boxes
  const srcLines: { text: string; bold?: boolean; indent?: boolean; color?: string }[] = [
    { text: `Studies from databases/registers (n = ${d.identified.toLocaleString()})`, bold: true, color: NAVY },
    ...Object.entries(d.sourceCounts).map(([k, v]) => ({ text: `${k} (n = ${v.toLocaleString()})`, indent: true, color: GRAY })),
  ];
  const otherLines = [
    { text: `References from other sources (n = ${d.otherSources.toLocaleString()})`, bold: true, color: NAVY },
    { text: `Citation searching (n = 0)`, indent: true, color: GRAY },
    { text: `Grey literature (n = 0)`, indent: true, color: GRAY },
  ];
  const maxTopH = Math.max(srcLines.length, otherLines.length) * LH + BOX_PAD * 2;

  const srcBox = textBox(LEFT, y, COL_W, srcLines);
  const othBox = textBox(RIGHT_COL, y, COL_W, otherLines);
  parts.push(srcBox.svg, othBox.svg);

  y += maxTopH + 10;

  // Arrow from left box down
  parts.push(arrow(LEFT + COL_W / 2, identStart + maxTopH, LEFT + COL_W / 2, y));

  // Removed box (right side)
  const removedLines = [
    { text: `References removed before screening (n = ${d.duplicatesRemoved.toLocaleString()})`, bold: true, color: NAVY },
    { text: `Duplicate records (n = ${d.duplicatesRemoved.toLocaleString()})`, indent: true, color: GRAY },
  ];
  const removedBox = textBox(RIGHT_COL, y - 5, COL_W, removedLines, LIGHT);
  parts.push(removedBox.svg);
  // Horizontal arrow from left col to removed box
  parts.push(arrow(LEFT + COL_W, y - 5 + removedBox.h / 2, RIGHT_COL, y - 5 + removedBox.h / 2, true));

  y += removedBox.h + 10;

  const identEnd = y;
  phaseBars.push({ x: 10, y: identStart, h: identEnd - identStart, label: "Identification" });

  // ── SCREENING ─────────────────────────────────────────────────────────
  const screenStart = y;

  // Row 1: screened → excluded
  const screenedBox = textBox(LEFT, y, COL_W, [{ text: `Studies screened (n = ${d.screened.toLocaleString()})`, bold: true, color: NAVY }]);
  const absExcLines: { text: string; bold?: boolean; indent?: boolean; color?: string }[] = [
    { text: `Studies excluded (n = ${d.abstractExcluded.toLocaleString()})`, bold: true, color: NAVY },
    ...d.abstractExcItems.map(it => ({ text: `${it.label} (n = ${it.count.toLocaleString()})`, indent: true, color: GRAY })),
  ];
  const absExcBox = textBox(RIGHT_COL, y, COL_W, absExcLines);
  const row1H = Math.max(screenedBox.h, absExcBox.h);
  parts.push(screenedBox.svg, absExcBox.svg);
  parts.push(arrow(LEFT + COL_W, y + screenedBox.h / 2, RIGHT_COL, y + screenedBox.h / 2, true));
  y += row1H + 8;
  parts.push(arrow(LEFT + COL_W / 2, y - 8, LEFT + COL_W / 2, y));

  // Row 2: sought for retrieval → not retrieved
  const soughtBox = textBox(LEFT, y, COL_W, [{ text: `Studies sought for retrieval (n = ${d.soughtRetrieval.toLocaleString()})`, bold: true, color: NAVY }]);
  const notRetBox = textBox(RIGHT_COL, y, COL_W, [{ text: `Studies not retrieved (n = ${d.notRetrieved.toLocaleString()})`, bold: true, color: NAVY }]);
  parts.push(soughtBox.svg, notRetBox.svg);
  parts.push(arrow(LEFT + COL_W, y + soughtBox.h / 2, RIGHT_COL, y + soughtBox.h / 2, true));
  y += Math.max(soughtBox.h, notRetBox.h) + 8;
  parts.push(arrow(LEFT + COL_W / 2, y - 8, LEFT + COL_W / 2, y));

  // Row 3: assessed → excluded at ft
  const assessedBox = textBox(LEFT, y, COL_W, [{ text: `Studies assessed for eligibility (n = ${d.assessed.toLocaleString()})`, bold: true, color: NAVY }]);
  const ftExcLines: { text: string; bold?: boolean; indent?: boolean; color?: string }[] = [
    { text: `Studies excluded (n = ${d.ftExcItems.reduce((s, it) => s + it.count, 0).toLocaleString()})`, bold: true, color: NAVY },
    ...d.ftExcItems.map(it => ({ text: `${it.label} (n = ${it.count.toLocaleString()})`, indent: true, color: GRAY })),
  ];
  const ftExcBox = textBox(RIGHT_COL, y, COL_W, ftExcLines.length > 1 ? ftExcLines : [{ text: `Studies excluded (n = 0)`, bold: true, color: NAVY }]);
  const row3H = Math.max(assessedBox.h, ftExcBox.h);
  parts.push(assessedBox.svg, ftExcBox.svg);
  parts.push(arrow(LEFT + COL_W, y + assessedBox.h / 2, RIGHT_COL, y + assessedBox.h / 2, true));
  y += row3H + 8;

  const screenEnd = y;
  phaseBars.push({ x: 10, y: screenStart, h: screenEnd - screenStart, label: "Screening" });

  // ── INCLUDED ──────────────────────────────────────────────────────────
  const inclStart = y;
  parts.push(arrow(LEFT + COL_W / 2, y - 8, LEFT + COL_W / 2, y));

  const inclBox = textBox(LEFT, y, COL_W, [{ text: `Studies included in review (n = ${d.included.toLocaleString()})`, bold: true, color: NAVY }], LIGHT);
  const ongoingBox = textBox(RIGHT_COL, y, COL_W, [
    { text: `Included studies ongoing (n = 0)`, bold: false, color: GRAY },
    { text: `Studies awaiting classification (n = 0)`, bold: false, color: GRAY },
  ], "#fff", true);
  parts.push(inclBox.svg, ongoingBox.svg);
  y += Math.max(inclBox.h, ongoingBox.h) + 20;

  const inclEnd = y;
  phaseBars.push({ x: 10, y: inclStart, h: inclEnd - inclStart, label: "Included" });

  // Draw phase bars
  for (const bar of phaseBars) {
    parts.push(`<rect x="${bar.x}" y="${bar.y}" width="${BAR_W}" height="${bar.h}" rx="4" fill="${NAVY}"/>`);
    const cx = bar.x + BAR_W / 2;
    const cy = bar.y + bar.h / 2;
    parts.push(`<text x="${cx}" y="${cy}" ${FONT} font-size="11" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 ${cx} ${cy})">${esc(bar.label)}</text>`);
  }

  parts.push(`<text x="${W / 2}" y="${y - 4}" ${FONT} font-size="9" fill="#94a3b8" text-anchor="middle">Generated by Evidence Engine</text>`);
  parts.push(`</svg>`);

  // Lock the final height onto the viewBox, the <svg> root, and the background
  // rect (all start at the 900 placeholder). The explicit root height matters:
  // WebKit/Safari report naturalHeight=0 for an SVG <img> without one, which
  // collapses the PNG/Word export canvas to zero height.
  const finalSvg = parts.join("")
    .replace(/viewBox="0 0 \d+ \d+"/, `viewBox="0 0 ${W} ${y}"`)
    .replace(/height="900"/g, `height="${y}"`);
  return finalSvg;
}
