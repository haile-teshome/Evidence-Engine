import { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { Download, Pencil } from "lucide-react";
import { toast } from "sonner";
import { QualityReport, QualityOverride, RoBJudgment, RoBDomain } from "../lib/mockServices";

const JUDGMENT_OPTIONS: RoBJudgment[] = [
  "Low", "Some Concerns", "High", "No information", "Not applicable",
];

function judgmentDotClass(j: RoBJudgment): string {
  switch (j) {
    case "Low": return "bg-emerald-500";
    case "Some Concerns": return "bg-amber-500";
    case "High": return "bg-rose-500";
    case "No information": return "bg-slate-300";
    case "Not applicable": return "bg-slate-200";
    default: return "bg-slate-200";
  }
}

function judgmentFillCss(j: RoBJudgment): string {
  switch (j) {
    case "Low": return "#10b981";          // emerald-500
    case "Some Concerns": return "#f59e0b"; // amber-500
    case "High": return "#f43f5e";          // rose-500
    case "No information": return "#cbd5e1"; // slate-300
    case "Not applicable": return "#e2e8f0"; // slate-200
    default: return "#e2e8f0";
  }
}

function shortDomainLabel(name: string): string {
  return name
    .replace(/^Bias (arising from|due to|in measurement of|in selection of|in classification of) /i, "")
    .replace(/^Bias /i, "")
    .replace(/^the /, "");
}

function latestOverride(paperId: string, domainId: string, overrides: QualityOverride[]): QualityOverride | undefined {
  let last: QualityOverride | undefined;
  for (const o of overrides) {
    if (o.paper_id === paperId && o.domain_id === domainId) last = o;
  }
  return last;
}

function effectiveJudgment(paperId: string, domain: { id: string; judgment: RoBJudgment }, overrides: QualityOverride[]): RoBJudgment {
  const o = latestOverride(paperId, domain.id, overrides);
  return o ? o.new_judgment : domain.judgment;
}

type Cell =
  | { kind: "na" }
  | { kind: "missing" }
  | { kind: "ok"; report: QualityReport; domain: RoBDomain; j: RoBJudgment; overridden: boolean };

function resolveCell(
  paperId: string,
  col: { rubric: string; domainId: string },
  reports: QualityReport[],
  overrides: QualityOverride[],
): Cell {
  const r = reports.find(x => x.paper_id === paperId);
  if (!r) return { kind: "na" };
  if (r.rubric !== col.rubric) return { kind: "na" };
  const d = r.domains.find(x => x.id === col.domainId);
  if (!d) return { kind: "missing" };
  const j = effectiveJudgment(r.paper_id, d, overrides);
  const overridden = !!latestOverride(r.paper_id, d.id, overrides);
  return { kind: "ok", report: r, domain: d, j, overridden };
}

function Swatch({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className={`inline-block size-3 rounded-sm ${className}`} /> {label}
    </span>
  );
}

export function CorpusHeatmap({
  reports,
  overrides,
  excluded,
  onOverride,
}: {
  reports: QualityReport[];
  overrides: QualityOverride[];
  excluded: Set<string>;
  /** When provided, cells become clickable to record a reviewer override. */
  onOverride?: (o: QualityOverride) => void;
}) {
  if (reports.length === 0) return null;

  const columns: { id: string; rubric: string; domainId: string; name: string }[] = [];
  const seen = new Set<string>();
  for (const r of reports) {
    for (const d of r.domains) {
      const key = `${r.rubric}::${d.id}`;
      if (!seen.has(key)) {
        seen.add(key);
        columns.push({ id: key, rubric: r.rubric, domainId: d.id, name: d.name });
      }
    }
  }

  function exportSvg() {
    const svg = buildHeatmapSvg(reports, overrides, columns);
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `risk-of-bias-${new Date().toISOString().slice(0, 10)}.svg`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Risk-of-bias heatmap exported. Open in any vector editor to relabel or restyle.");
  }

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h3 className="font-medium">Domain heatmap</h3>
        <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
          <Swatch className={judgmentDotClass("Low")} label="Low" />
          <Swatch className={judgmentDotClass("Some Concerns")} label="Some Concerns" />
          <Swatch className={judgmentDotClass("High")} label="High" />
          <Swatch className={judgmentDotClass("No information")} label="No info" />
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-3 rounded-full ring-1 ring-foreground/40 bg-emerald-500" /> overridden
          </span>
          <Button size="sm" variant="outline" onClick={exportSvg} className="h-7">
            <Download className="size-3 mr-1" /> Export SVG
          </Button>
        </div>
      </div>

      {onOverride && (
        <div className="text-xs text-muted-foreground mb-2">
          Click any cell to override the AI's judgment.
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead>
            <tr>
              <th className="text-left px-2 py-1 sticky left-0 bg-card">Paper</th>
              {columns.map(c => (
                <th key={c.id} className="px-1 py-1 align-bottom">
                  <div
                    className="inline-block whitespace-nowrap text-left text-muted-foreground"
                    style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", height: 120 }}
                    title={c.name}
                  >
                    {shortDomainLabel(c.name)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {reports.map(r => (
              <tr key={r.paper_id} className={excluded.has(r.paper_id) ? "opacity-40" : ""}>
                <td className="px-2 py-1 sticky left-0 bg-card max-w-[280px] truncate" title={r.title}>
                  {r.title}
                </td>
                {columns.map(c => {
                  const cell = resolveCell(r.paper_id, c, reports, overrides);
                  return (
                    <td key={c.id} className="px-1 py-1 text-center">
                      <CellMarker cell={cell} domainName={c.name} onOverride={onOverride} />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function CellMarker({
  cell,
  domainName,
  onOverride,
}: {
  cell: Cell;
  domainName: string;
  onOverride?: (o: QualityOverride) => void;
}) {
  const [open, setOpen] = useState(false);
  const [next, setNext] = useState<RoBJudgment>("Low");
  const [reason, setReason] = useState("");

  if (cell.kind === "na") {
    return (
      <span
        className={`inline-block size-3 rounded-sm ${judgmentDotClass("Not applicable")}`}
        title={`${domainName}: Not applicable to this rubric`}
      />
    );
  }
  if (cell.kind === "missing") {
    return (
      <span
        className={`inline-block size-3 rounded-sm ${judgmentDotClass("No information")}`}
        title={`${domainName}: No information`}
      />
    );
  }

  const { report, domain, j, overridden } = cell;
  const dot = (
    <span
      className={`inline-block size-3 ${overridden ? "rounded-full ring-1 ring-foreground/40" : "rounded-sm"} ${judgmentDotClass(j)} ${onOverride ? "cursor-pointer hover:ring-2 hover:ring-foreground/40" : ""}`}
      title={`${domain.name}: ${j}${overridden ? " (reviewer-edited)" : ""}`}
    />
  );

  if (!onOverride) return dot;

  function handleSave() {
    if (next === j) {
      toast.info("Pick a different judgment to record an override.");
      return;
    }
    if (!reason.trim()) {
      toast.error("Provide a brief reason for the override.");
      return;
    }
    onOverride!({
      paper_id: report.paper_id,
      domain_id: domain.id,
      original_judgment: domain.judgment,
      new_judgment: next,
      reason: reason.trim(),
      reviewer: "",
      timestamp: new Date().toISOString(),
    });
    setReason("");
    setOpen(false);
  }

  function handleOpen(v: boolean) {
    setOpen(v);
    if (v) { setNext(j); setReason(""); }
  }

  return (
    <Popover open={open} onOpenChange={handleOpen}>
      <PopoverTrigger asChild>
        <button className="inline-flex">{dot}</button>
      </PopoverTrigger>
      <PopoverContent className="w-72" align="start">
        <div className="space-y-3">
          <div className="flex items-center gap-1 text-xs uppercase tracking-wide text-muted-foreground">
            <Pencil className="size-3" /> Override judgment
          </div>
          <div className="text-sm font-medium break-words">{domain.name}</div>
          <div className="text-xs text-muted-foreground">
            AI judgment: <span className="font-medium text-foreground">{domain.judgment}</span>
            {overridden && (
              <> · current: <span className="font-medium text-foreground">{j}</span></>
            )}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">New judgment</Label>
            <Select value={next} onValueChange={(v) => setNext(v as RoBJudgment)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {JUDGMENT_OPTIONS.map(o => <SelectItem key={o} value={o}>{o}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Reason</Label>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why does this domain warrant a different judgment?"
              rows={3}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
            <Button size="sm" onClick={handleSave}>Save override</Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// ---------------------------------------------------------------------------
// SVG export
// ---------------------------------------------------------------------------

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max - 1) + "…";
}

function buildHeatmapSvg(
  reports: QualityReport[],
  overrides: QualityOverride[],
  columns: { id: string; rubric: string; domainId: string; name: string }[],
): string {
  const TITLE_W = 280;
  const CELL_W = 22;
  const CELL_H = 22;
  const DOT_R = 7;
  const HEADER_H = 160;
  const ROW_H = CELL_H;
  const PAD = 16;
  const LEGEND_H = 36;

  const gridX = PAD + TITLE_W;
  const gridY = LEGEND_H + PAD + HEADER_H;

  const totalW = gridX + columns.length * CELL_W + PAD;
  const totalH = gridY + reports.length * ROW_H + PAD;

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${totalW} ${totalH}" width="${totalW}" height="${totalH}" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">`,
  );
  parts.push(`<rect width="${totalW}" height="${totalH}" fill="#ffffff"/>`);

  // ---- Legend ------------------------------------------------------------
  const legend: { j: RoBJudgment; label: string }[] = [
    { j: "Low", label: "Low" },
    { j: "Some Concerns", label: "Some Concerns" },
    { j: "High", label: "High" },
    { j: "No information", label: "No info" },
  ];
  let lx = PAD;
  const ly = PAD + 12;
  legend.forEach(item => {
    parts.push(
      `<circle cx="${lx + 6}" cy="${ly - 4}" r="6" fill="${judgmentFillCss(item.j)}"/>`,
    );
    parts.push(
      `<text x="${lx + 18}" y="${ly}" font-size="12" fill="#334155">${escapeXml(item.label)}</text>`,
    );
    lx += 18 + item.label.length * 7 + 16;
  });
  // Overridden marker
  parts.push(
    `<circle cx="${lx + 6}" cy="${ly - 4}" r="6" fill="${judgmentFillCss("Low")}" stroke="#0f172a" stroke-width="1.5"/>`,
  );
  parts.push(
    `<text x="${lx + 18}" y="${ly}" font-size="12" fill="#334155">overridden</text>`,
  );

  // ---- Column headers (rotated 270°) ------------------------------------
  columns.forEach((c, ci) => {
    const x = gridX + ci * CELL_W + CELL_W / 2;
    const y = gridY - 8;
    const label = truncate(shortDomainLabel(c.name), 22);
    parts.push(
      `<text x="${x}" y="${y}" font-size="11" fill="#475569" text-anchor="start" transform="rotate(-65 ${x} ${y})">${escapeXml(label)}</text>`,
    );
  });

  // ---- Rows ----------------------------------------------------------------
  reports.forEach((r, ri) => {
    const yMid = gridY + ri * ROW_H + ROW_H / 2;
    // Title
    parts.push(
      `<text x="${PAD}" y="${yMid + 4}" font-size="12" fill="#0f172a">${escapeXml(truncate(r.title, 50))}</text>`,
    );
    // Cells
    columns.forEach((c, ci) => {
      const cx = gridX + ci * CELL_W + CELL_W / 2;
      const cell = resolveCell(r.paper_id, c, reports, overrides);
      let j: RoBJudgment;
      let overridden = false;
      if (cell.kind === "ok") { j = cell.j; overridden = cell.overridden; }
      else if (cell.kind === "na") { j = "Not applicable"; }
      else { j = "No information"; }
      const fill = judgmentFillCss(j);
      if (overridden) {
        parts.push(
          `<circle cx="${cx}" cy="${yMid}" r="${DOT_R}" fill="${fill}" stroke="#0f172a" stroke-width="1.5"/>`,
        );
      } else {
        const half = DOT_R;
        parts.push(
          `<rect x="${cx - half}" y="${yMid - half}" width="${half * 2}" height="${half * 2}" rx="2" ry="2" fill="${fill}"/>`,
        );
      }
    });
  });

  parts.push(`</svg>`);
  return parts.join("");
}

