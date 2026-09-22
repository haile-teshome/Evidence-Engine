import { useEffect, useMemo, useState } from "react";
import { useStore } from "../lib/store";
import {
  MetaAnalysisService,
  StudyEffect,
  EffectMeasure,
  Tau2Method,
  Paper,
} from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { SourcePane, TablePane, AskPane } from "../components/EvidenceWorkspace";
import { AIService, type ExtractedTable } from "../lib/apiClient";
import { Sigma, Play, Calculator, X as XIcon, AlertTriangle, ChevronDown, FileText, ListChecks, Percent } from "lucide-react";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { EmptyState } from "../components/EmptyState";
import { toast } from "sonner";

// ===========================================================================
// Effect-measure metadata
// ===========================================================================

/** Format a statistic that may legitimately not exist.
 *
 *  The backend sends null for an undefined statistic: a p-value with zero
 *  degrees of freedom, a between-group Q when there is only one group, a
 *  z-score with zero standard error. Those are real answers and "–" is the
 *  right way to show them. Calling .toFixed() on one throws, and because this
 *  is render-time it takes the entire page down with "Something went wrong"
 *  rather than leaving a blank cell. */
function num(v: number | null | undefined, dp = 3): string {
  return v == null || !Number.isFinite(v) ? "–" : v.toFixed(dp);
}

/** Same, for a value that must be exponentiated first. Math.exp(null) is 1,
 *  which would silently render a missing effect as a null effect. */
function expNum(v: number | null | undefined, dp = 2): string {
  return v == null || !Number.isFinite(v) ? "–" : Math.exp(v).toFixed(dp);
}

const MEASURE_META: Record<EffectMeasure, { label: string; group: string; logScale: boolean; nullValue: number; axis: string }> = {
  OR:       { label: "Odds Ratio (OR)",            group: "Binary (2x2)",      logScale: true,  nullValue: 0, axis: "log OR" },
  RR:       { label: "Risk Ratio (RR)",            group: "Binary (2x2)",      logScale: true,  nullValue: 0, axis: "log RR" },
  RD:       { label: "Risk Difference (RD)",       group: "Binary (2x2)",      logScale: false, nullValue: 0, axis: "Risk Difference" },
  PETO_OR:  { label: "Peto OR (rare events)",      group: "Binary (2x2)",      logScale: true,  nullValue: 0, axis: "log Peto OR" },
  HR:       { label: "Hazard Ratio (HR)",          group: "Time-to-event",     logScale: true,  nullValue: 0, axis: "log HR" },
  MD:       { label: "Mean Difference (MD)",       group: "Continuous",        logScale: false, nullValue: 0, axis: "Mean Difference" },
  SMD:      { label: "Standardised MD (Hedges g)", group: "Continuous",        logScale: false, nullValue: 0, axis: "Hedges' g" },
  PROP:     { label: "Proportion / Prevalence",    group: "Single-arm",        logScale: false, nullValue: 0, axis: "logit(p)" },
  IR:       { label: "Incidence Rate",             group: "Single-arm",        logScale: true,  nullValue: 0, axis: "log IR" },
  ZCOR:     { label: "Correlation (Fisher z)",     group: "Correlational",     logScale: false, nullValue: 0, axis: "Fisher's z" },
  GENERIC:  { label: "Generic (yi + SE supplied)", group: "Generic",           logScale: false, nullValue: 0, axis: "Effect size" },
  UNKNOWN:  { label: "Unknown (LLM uncertain)",    group: "Generic",           logScale: false, nullValue: 0, axis: "Effect size" },
};

const MEASURE_GROUPS = ["Binary (2x2)", "Time-to-event", "Continuous", "Single-arm", "Correlational", "Generic"];

function exponentiateLabel(measure: EffectMeasure, v: number | null | undefined): string {
  // Every axis and study label funnels through here, so one null reaching it
  // blanks the whole forest plot rather than one number.
  if (v == null || !Number.isFinite(v)) return "–";
  return MEASURE_META[measure]?.logScale ? Math.exp(v).toFixed(2) : v.toFixed(3);
}

function isLogScale(measures: EffectMeasure[] | undefined): boolean {
  if (!measures || measures.length === 0) return false;
  // Only apply exp() in axis labels when ALL studies use the same log-scale measure.
  return measures.length === 1 && !!MEASURE_META[measures[0]]?.logScale;
}

// ===========================================================================
// Generic axis utilities
// ===========================================================================

function niceTicks(a: number, b: number, n = 5): number[] {
  const step = (b - a) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(1e-9, Math.abs(step)))));
  const rounded = Math.max(0.05, Math.round(step / mag) * mag);
  const out: number[] = [];
  let t = Math.ceil(a / rounded) * rounded;
  while (t <= b + 1e-9) { out.push(Math.round(t * 1000) / 1000); t += rounded; }
  return out;
}

// ===========================================================================
// Forest plot
// ===========================================================================

function ForestPlot({
  studies,
  pooled,
  predInterval,
  measures,
  compact = false,
}: {
  studies: StudyEffect[];
  pooled: { estimate: number; ci_low: number; ci_high: number } | null;
  predInterval: { low: number; high: number } | null;
  measures: EffectMeasure[];
  /** Narrow layout for the side-by-side workspace. The label and weight
   *  columns are fixed widths totalling 440px, so in a pane narrower than that
   *  the plot area collapses to nothing and the chart renders as a column of
   *  study names with no data in it. */
  compact?: boolean;
}) {
  const rowH = 24;
  const padTop = 30;
  const padBottom = 60;
  const padLeftLabel = compact ? 150 : 280;
  const padRightWeights = compact ? 78 : 160;
  const plotW = 360;
  const totalRows = studies.length + (pooled ? 2 : 0) + (predInterval ? 1 : 0);
  const height = padTop + padBottom + Math.max(1, totalRows) * rowH;
  const width = padLeftLabel + plotW + padRightWeights + 20;

  const lows: number[] = [];
  const highs: number[] = [];
  studies.forEach(s => {
    if (typeof s.ci_low === "number") lows.push(s.ci_low);
    if (typeof s.ci_high === "number") highs.push(s.ci_high);
    if (typeof s.yi === "number") { lows.push(s.yi); highs.push(s.yi); }
  });
  if (pooled) { lows.push(pooled.ci_low); highs.push(pooled.ci_high); }
  if (predInterval) { lows.push(predInterval.low); highs.push(predInterval.high); }
  let lo = lows.length ? Math.min(...lows) : -1;
  let hi = highs.length ? Math.max(...highs) : 1;
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) { lo = -1; hi = 1; }
  const span = hi - lo;
  lo -= span * 0.1; hi += span * 0.1;
  const px = (x: number) => padLeftLabel + ((x - lo) / (hi - lo)) * plotW;
  const nullX = px(0);
  const ticks = niceTicks(lo, hi);
  const log = isLogScale(measures);
  const measure = measures.length === 1 ? measures[0] : "GENERIC";
  const axisLabel = MEASURE_META[measure]?.axis || "Effect size";

  return (
    <svg width={width} height={height} className="font-sans" role="img" aria-label="Forest plot">
      <text x={6} y={padTop - 12} fontSize={10} fill="currentColor" className="text-muted-foreground">Study</text>
      <text x={padLeftLabel + plotW + 8} y={padTop - 12} fontSize={10} fill="currentColor" className="text-muted-foreground">
        Weight · Effect [95% CI]{log ? "  (back-transformed)" : ""}
      </text>

      {studies.map((s, i) => {
        const rowY = padTop + i * rowH + rowH / 2;
        if (typeof s.yi !== "number" || typeof s.ci_low !== "number" || typeof s.ci_high !== "number") return null;
        const cx = px(s.yi);
        const xLo = Math.max(padLeftLabel, px(s.ci_low));
        const xHi = Math.min(padLeftLabel + plotW, px(s.ci_high));
        const sizePx = Math.max(4, Math.min(14, 4 + 30 * (s.weight_re ?? s.weight_fe ?? 0.05)));
        const weightPct = ((s.weight_re ?? s.weight_fe ?? 0) * 100).toFixed(1);
        const eff = log ? exponentiateLabel(s.effect_measure, s.yi) : num(s.yi, 2);
        const effLo = log ? exponentiateLabel(s.effect_measure, s.ci_low) : num(s.ci_low, 2);
        const effHi = log ? exponentiateLabel(s.effect_measure, s.ci_high) : num(s.ci_high, 2);
        return (
          <g key={s.paper_id || i}>
            <text x={6} y={rowY + 4} fontSize={11} fill="currentColor">
              <title>{s.title}</title>
              {(s.title || "(untitled)").slice(0, compact ? 22 : 50)}
            </text>
            <line x1={xLo} x2={xHi} y1={rowY} y2={rowY} stroke="currentColor" strokeWidth={1.2} className="text-foreground/70" />
            <rect x={cx - sizePx / 2} y={rowY - sizePx / 2} width={sizePx} height={sizePx} fill="currentColor" className="text-primary" />
            <text x={padLeftLabel + plotW + 8} y={rowY + 4} fontSize={10} fill="currentColor" className="text-muted-foreground tabular-nums">
              {weightPct}% · {eff} [{effLo}, {effHi}]
            </text>
          </g>
        );
      })}

      {nullX >= padLeftLabel && nullX <= padLeftLabel + plotW && (
        <line x1={nullX} x2={nullX}
          y1={padTop - 6}
          y2={padTop + studies.length * rowH + (pooled ? rowH * 1.5 : 0) + (predInterval ? rowH : 0)}
          stroke="currentColor" strokeDasharray="3 3" strokeWidth={1} className="text-muted-foreground" />
      )}

      {pooled && (() => {
        const rowY = padTop + studies.length * rowH + rowH;
        const xLo = Math.max(padLeftLabel, px(pooled.ci_low));
        const xHi = Math.min(padLeftLabel + plotW, px(pooled.ci_high));
        const cx = px(pooled.estimate);
        const dyHalf = 7;
        const diamond = `${xLo},${rowY} ${cx},${rowY - dyHalf} ${xHi},${rowY} ${cx},${rowY + dyHalf}`;
        const eff = log ? exponentiateLabel(measure, pooled.estimate) : num(pooled.estimate);
        const effLo = log ? exponentiateLabel(measure, pooled.ci_low) : num(pooled.ci_low);
        const effHi = log ? exponentiateLabel(measure, pooled.ci_high) : num(pooled.ci_high);
        return (
          <g>
            <line x1={padLeftLabel} x2={padLeftLabel + plotW} y1={rowY - rowH / 2} y2={rowY - rowH / 2} stroke="currentColor" strokeWidth={0.5} className="text-muted-foreground/40" />
            <text x={6} y={rowY + 4} fontSize={11} fontWeight="600" fill="currentColor">Pooled (RE)</text>
            <polygon points={diamond} fill="currentColor" className="text-primary" />
            <text x={padLeftLabel + plotW + 8} y={rowY + 4} fontSize={10} fontWeight="600" fill="currentColor" className="tabular-nums">
              {eff} [{effLo}, {effHi}]
            </text>
          </g>
        );
      })()}

      {predInterval && pooled && (() => {
        const rowY = padTop + (studies.length + 1) * rowH + rowH;
        const xLo = Math.max(padLeftLabel - 5, px(predInterval.low));
        const xHi = Math.min(padLeftLabel + plotW + 5, px(predInterval.high));
        const piLo = log ? exponentiateLabel(measure, predInterval.low) : num(predInterval.low);
        const piHi = log ? exponentiateLabel(measure, predInterval.high) : num(predInterval.high);
        return (
          <g>
            <text x={6} y={rowY + 4} fontSize={10} fill="currentColor" className="text-muted-foreground italic">95% prediction interval</text>
            <line x1={xLo} x2={xHi} y1={rowY} y2={rowY} stroke="currentColor" strokeWidth={1.5} strokeDasharray="6 3" className="text-amber-600" />
            <text x={padLeftLabel + plotW + 8} y={rowY + 4} fontSize={10} fill="currentColor" className="text-muted-foreground tabular-nums">
              [{piLo}, {piHi}]
            </text>
          </g>
        );
      })()}

      {(() => {
        const axisY = height - padBottom + 16;
        return (
          <g>
            <line x1={padLeftLabel} x2={padLeftLabel + plotW} y1={axisY} y2={axisY} stroke="currentColor" strokeWidth={1} />
            {ticks.map((t, i) => {
              const xx = px(t);
              if (xx < padLeftLabel || xx > padLeftLabel + plotW) return null;
              return (
                <g key={i}>
                  <line x1={xx} x2={xx} y1={axisY} y2={axisY + 4} stroke="currentColor" strokeWidth={1} />
                  <text x={xx} y={axisY + 16} fontSize={10} textAnchor="middle" fill="currentColor" className="text-muted-foreground tabular-nums">
                    {log ? Math.exp(t).toFixed(2) : t.toFixed(t % 1 === 0 ? 0 : 1)}
                  </text>
                </g>
              );
            })}
            <text x={padLeftLabel + plotW / 2} y={axisY + 32} fontSize={10} textAnchor="middle" fill="currentColor" className="text-muted-foreground">
              {log ? `${axisLabel}  (back-transformed scale)` : axisLabel}
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

// ===========================================================================
// Funnel plot
// ===========================================================================

function FunnelPlot({ studies, center, measure }: { studies: { yi: number; se: number; title: string }[]; center: number | null; measure: EffectMeasure }) {
  if (studies.length === 0 || center == null) return null;
  const width = 540;
  const height = 360;
  const padL = 60, padR = 30, padT = 20, padB = 50;

  const maxSE = Math.max(...studies.map(s => s.se)) * 1.15;
  const xs = studies.map(s => s.yi);
  let lo = Math.min(...xs, center - 1.96 * maxSE);
  let hi = Math.max(...xs, center + 1.96 * maxSE);
  const span = hi - lo;
  lo -= span * 0.1; hi += span * 0.1;
  const xpx = (x: number) => padL + ((x - lo) / (hi - lo)) * (width - padL - padR);
  const ypx = (se: number) => padT + (se / maxSE) * (height - padT - padB);

  // 95% pseudo-CI envelope: at SE=s, the bounds are center ± 1.96*s.
  const pathL: string[] = [];
  const pathR: string[] = [];
  const steps = 30;
  for (let i = 0; i <= steps; i++) {
    const se = (i / steps) * maxSE;
    pathL.push(`${i === 0 ? "M" : "L"} ${xpx(center - 1.96 * se)} ${ypx(se)}`);
    pathR.push(`${i === 0 ? "M" : "L"} ${xpx(center + 1.96 * se)} ${ypx(se)}`);
  }

  const ticks = niceTicks(lo, hi);
  const log = MEASURE_META[measure]?.logScale ?? false;
  const axisLabel = MEASURE_META[measure]?.axis || "Effect size";

  return (
    <svg width={width} height={height} className="font-sans" role="img" aria-label="Funnel plot">
      {/* Frame */}
      <line x1={padL} x2={padL} y1={padT} y2={height - padB} stroke="currentColor" strokeWidth={1} />
      <line x1={padL} x2={width - padR} y1={height - padB} y2={height - padB} stroke="currentColor" strokeWidth={1} />
      {/* Pseudo-CI envelope */}
      <path d={pathL.join(" ")} stroke="currentColor" strokeDasharray="4 3" strokeWidth={1} fill="none" className="text-muted-foreground" />
      <path d={pathR.join(" ")} stroke="currentColor" strokeDasharray="4 3" strokeWidth={1} fill="none" className="text-muted-foreground" />
      {/* Center line */}
      <line x1={xpx(center)} x2={xpx(center)} y1={padT} y2={height - padB} stroke="currentColor" strokeDasharray="3 3" strokeWidth={1} className="text-primary/60" />
      {/* Study points */}
      {studies.map((s, i) => (
        <circle key={i} cx={xpx(s.yi)} cy={ypx(s.se)} r={4} fill="currentColor" className="text-primary">
          <title>{s.title}</title>
        </circle>
      ))}
      {/* X axis ticks */}
      {ticks.map((t, i) => {
        const x = xpx(t);
        return (
          <g key={i}>
            <line x1={x} x2={x} y1={height - padB} y2={height - padB + 4} stroke="currentColor" strokeWidth={1} />
            <text x={x} y={height - padB + 16} fontSize={10} textAnchor="middle" fill="currentColor" className="text-muted-foreground tabular-nums">
              {log ? Math.exp(t).toFixed(2) : t.toFixed(t % 1 === 0 ? 0 : 1)}
            </text>
          </g>
        );
      })}
      <text x={(padL + width - padR) / 2} y={height - padB + 36} fontSize={11} textAnchor="middle" fill="currentColor" className="text-muted-foreground">
        {log ? `${axisLabel}  (back-transformed)` : axisLabel}
      </text>
      {/* Y axis: standard error, inverted */}
      <text x={padL - 40} y={padT + 10} fontSize={10} fill="currentColor" className="text-muted-foreground">SE = 0</text>
      <text x={padL - 40} y={height - padB} fontSize={10} fill="currentColor" className="text-muted-foreground">SE = {maxSE.toFixed(2)}</text>
      <text x={20} y={(padT + height - padB) / 2} fontSize={11} fill="currentColor" className="text-muted-foreground" transform={`rotate(-90 20 ${(padT + height - padB) / 2})`} textAnchor="middle">
        Standard Error
      </text>
    </svg>
  );
}

// ===========================================================================
// Meta-regression bubble plot
// ===========================================================================

function BubblePlot({
  studies,
  slope,
  intercept,
  measure,
}: {
  studies: StudyEffect[];
  slope: number | null;
  intercept: number | null;
  measure: EffectMeasure;
}) {
  const pts = studies.filter(s =>
    typeof s.moderator === "number" && typeof s.yi === "number" && typeof s.vi === "number" && s.vi > 0
  );
  if (pts.length < 2) {
    return <p className="text-sm text-muted-foreground">Meta-regression needs at least 3 studies with a moderator value. Edit study rows to add moderator data (e.g., year, dose, latitude).</p>;
  }
  const w = 540, h = 320, padL = 60, padR = 20, padT = 20, padB = 50;
  const xs = pts.map(p => p.moderator as number);
  const ys = pts.map(p => p.yi as number);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpan = Math.max(1e-9, xMax - xMin); xMin -= xSpan * 0.05; xMax += xSpan * 0.05;
  const ySpan = Math.max(1e-9, yMax - yMin); yMin -= ySpan * 0.1; yMax += ySpan * 0.1;
  const xp = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * (w - padL - padR);
  const yp = (y: number) => padT + (1 - (y - yMin) / (yMax - yMin)) * (h - padT - padB);

  const log = MEASURE_META[measure]?.logScale ?? false;
  const axisLabel = MEASURE_META[measure]?.axis || "Effect size";
  const xticks = niceTicks(xMin, xMax);
  const yticks = niceTicks(yMin, yMax);

  return (
    <svg width={w} height={h} className="font-sans" role="img" aria-label="Meta-regression bubble plot">
      <line x1={padL} x2={padL} y1={padT} y2={h - padB} stroke="currentColor" strokeWidth={1} />
      <line x1={padL} x2={w - padR} y1={h - padB} y2={h - padB} stroke="currentColor" strokeWidth={1} />
      {/* Regression line */}
      {slope !== null && intercept !== null && (
        <line
          x1={xp(xMin)} x2={xp(xMax)}
          y1={yp(intercept + slope * xMin)} y2={yp(intercept + slope * xMax)}
          stroke="currentColor" strokeWidth={1.5} className="text-primary"
        />
      )}
      {/* Bubbles, radius ∝ 1/SE */}
      {pts.map((p, i) => {
        const r = Math.max(3, Math.min(12, 2 + 6 / Math.sqrt(p.vi as number)));
        return (
          <circle key={i} cx={xp(p.moderator as number)} cy={yp(p.yi as number)} r={r}
            fill="currentColor" fillOpacity={0.55} className="text-primary">
            <title>{p.title}</title>
          </circle>
        );
      })}
      {xticks.map((t, i) => {
        const x = xp(t);
        return (
          <g key={i}>
            <line x1={x} x2={x} y1={h - padB} y2={h - padB + 4} stroke="currentColor" />
            <text x={x} y={h - padB + 16} fontSize={10} textAnchor="middle" fill="currentColor" className="text-muted-foreground tabular-nums">
              {t.toFixed(t % 1 === 0 ? 0 : 1)}
            </text>
          </g>
        );
      })}
      <text x={(padL + w - padR) / 2} y={h - padB + 36} fontSize={11} textAnchor="middle" fill="currentColor" className="text-muted-foreground">Moderator</text>
      {yticks.map((t, i) => {
        const y = yp(t);
        return (
          <g key={i}>
            <line x1={padL - 4} x2={padL} y1={y} y2={y} stroke="currentColor" />
            <text x={padL - 6} y={y + 3} fontSize={10} textAnchor="end" fill="currentColor" className="text-muted-foreground tabular-nums">
              {log ? Math.exp(t).toFixed(2) : t.toFixed(t % 1 === 0 ? 0 : 1)}
            </text>
          </g>
        );
      })}
      <text x={18} y={(padT + h - padB) / 2} fontSize={11} fill="currentColor" className="text-muted-foreground"
        transform={`rotate(-90 18 ${(padT + h - padB) / 2})`} textAnchor="middle">
        {axisLabel}
      </text>
    </svg>
  );
}

// ===========================================================================
// Study row editor. Switches input fields based on effect_measure
// ===========================================================================

function StudyRowEditor({
  study,
  onChange,
  onRemove,
}: {
  study: StudyEffect;
  onChange: (next: StudyEffect) => void;
  onRemove: () => void;
}) {
  const m = (study.effect_measure || "UNKNOWN") as EffectMeasure;
  const conf = typeof study.extraction_confidence === "number" ? study.extraction_confidence : null;

  function set<K extends keyof StudyEffect>(k: K, v: any) {
    onChange({ ...study, [k]: v, yi: null, vi: null, se: null, ci_low: null, ci_high: null });
  }
  function setNum<K extends keyof StudyEffect>(k: K, raw: string) {
    const v = raw === "" ? null : Number(raw);
    set(k, Number.isNaN(v as any) ? null : v);
  }
  function setStr<K extends keyof StudyEffect>(k: K, raw: string) {
    set(k, raw === "" ? null : raw);
  }

  const isBinary = m === "OR" || m === "RR" || m === "RD" || m === "PETO_OR";
  const isContinuous = m === "MD" || m === "SMD";
  const isHR = m === "HR";
  const isProp = m === "PROP";
  const isIR = m === "IR";
  const isZ = m === "ZCOR";
  const isGeneric = m === "GENERIC" || m === "UNKNOWN";

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm break-words">
            {study.url ? (
              <a href={study.url} target="_blank" rel="noreferrer" className="hover:underline">{study.title || "(untitled)"}</a>
            ) : (
              study.title || "(untitled)"
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1 flex-wrap">
            <Select value={m} onValueChange={(v: any) => set("effect_measure", v)}>
              <SelectTrigger className="h-7 w-44 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {MEASURE_GROUPS.map(g => (
                  <div key={g}>
                    <div className="px-2 py-1 text-[10px] uppercase text-muted-foreground">{g}</div>
                    {(Object.keys(MEASURE_META) as EffectMeasure[])
                      .filter(k => MEASURE_META[k].group === g)
                      .map(k => <SelectItem key={k} value={k}>{MEASURE_META[k].label}</SelectItem>)}
                  </div>
                ))}
              </SelectContent>
            </Select>
            {study.outcome && <span className="truncate">{study.outcome}</span>}
            {conf !== null && (
              <Badge variant="outline" className={conf >= 0.7 ? "text-emerald-700" : conf >= 0.4 ? "text-amber-700" : "text-rose-700"}>
                conf {(conf * 100).toFixed(0)}%
              </Badge>
            )}
            {typeof study.yi === "number" && (
              <span className="font-mono tabular-nums">
                yi={num(study.yi)}  SE={num(study.se)}
              </span>
            )}
          </div>
        </div>
        <Button variant="ghost" size="sm" onClick={onRemove} title="Remove from analysis"><XIcon className="size-4" /></Button>
      </div>

      {study.error && (
        <div className="flex items-start gap-1 text-xs text-amber-700">
          <AlertTriangle className="size-3 shrink-0 mt-0.5" />
          <span>{study.error}</span>
        </div>
      )}

      {isBinary && (
        <div className="grid grid-cols-4 gap-2 text-xs">
          {[["events_t","events_t"],["n_t","n_t"],["events_c","events_c"],["n_c","n_c"]].map(([key, lbl]) => (
            <div key={key}>
              <Label className="text-[10px]">{lbl}</Label>
              <Input className="h-7 font-mono text-xs" value={(study as any)[key] ?? ""} onChange={e => setNum(key as keyof StudyEffect, e.target.value)} />
            </div>
          ))}
        </div>
      )}

      {isContinuous && (
        <div className="grid grid-cols-6 gap-2 text-xs">
          {[["mean_t","mean_t"],["sd_t","sd_t"],["n_t","n_t"],["mean_c","mean_c"],["sd_c","sd_c"],["n_c","n_c"]].map(([key, lbl]) => (
            <div key={key}>
              <Label className="text-[10px]">{lbl}</Label>
              <Input className="h-7 font-mono text-xs" value={(study as any)[key] ?? ""} onChange={e => setNum(key as keyof StudyEffect, e.target.value)} />
            </div>
          ))}
        </div>
      )}

      {isHR && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          {[["hr","HR"],["hr_ci_low","CI low"],["hr_ci_high","CI high"]].map(([key, lbl]) => (
            <div key={key}>
              <Label className="text-[10px]">{lbl}</Label>
              <Input className="h-7 font-mono text-xs" value={(study as any)[key] ?? ""} onChange={e => setNum(key as keyof StudyEffect, e.target.value)} />
            </div>
          ))}
        </div>
      )}

      {isProp && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><Label className="text-[10px]">events_total</Label>
            <Input className="h-7 font-mono text-xs" value={study.events_total ?? ""} onChange={e => setNum("events_total", e.target.value)} /></div>
          <div><Label className="text-[10px]">n_total</Label>
            <Input className="h-7 font-mono text-xs" value={study.n_total ?? ""} onChange={e => setNum("n_total", e.target.value)} /></div>
        </div>
      )}

      {isIR && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><Label className="text-[10px]">events_total</Label>
            <Input className="h-7 font-mono text-xs" value={study.events_total ?? ""} onChange={e => setNum("events_total", e.target.value)} /></div>
          <div><Label className="text-[10px]">person-time</Label>
            <Input className="h-7 font-mono text-xs" value={study.person_time ?? ""} onChange={e => setNum("person_time", e.target.value)} /></div>
        </div>
      )}

      {isZ && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><Label className="text-[10px]">correlation (r)</Label>
            <Input className="h-7 font-mono text-xs" value={study.correlation ?? ""} onChange={e => setNum("correlation", e.target.value)} /></div>
          <div><Label className="text-[10px]">n_total</Label>
            <Input className="h-7 font-mono text-xs" value={study.n_total ?? ""} onChange={e => setNum("n_total", e.target.value)} /></div>
        </div>
      )}

      {isGeneric && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><Label className="text-[10px]">yi (log-scale effect)</Label>
            <Input className="h-7 font-mono text-xs" value={study.yi ?? ""} onChange={e => setNum("yi", e.target.value)} /></div>
          <div><Label className="text-[10px]">se</Label>
            <Input className="h-7 font-mono text-xs" value={study.se ?? ""} onChange={e => setNum("se", e.target.value)} /></div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div><Label className="text-[10px]">subgroup (categorical moderator)</Label>
          <Input className="h-7 text-xs" placeholder="e.g., RCT, Asia, high-dose" value={study.subgroup ?? ""} onChange={e => setStr("subgroup", e.target.value)} /></div>
        <div><Label className="text-[10px]">moderator (continuous, for meta-regression)</Label>
          <Input className="h-7 font-mono text-xs" placeholder="e.g., year, dose, age" value={study.moderator ?? ""} onChange={e => setNum("moderator", e.target.value)} /></div>
      </div>

      {study.extraction_quote && (
        <blockquote className="border-l-2 border-primary/30 pl-2 italic text-xs text-muted-foreground">
          “{study.extraction_quote}”
        </blockquote>
      )}
    </Card>
  );
}

// ===========================================================================
// Main page
// ===========================================================================

/** Shown in place of the forest plot when nothing could be pooled.
 *
 *  An empty panel here reads as a broken feature. In practice pooling fails for
 *  a small number of concrete, fixable reasons, and naming them per study turns
 *  a dead end into a worklist. */
function PoolBlocked({ extractions, ran }: { extractions: StudyEffect[]; ran: boolean }) {
  const blocked = extractions.filter(x => x.yi === null || x.yi === undefined);
  if (!ran) {
    return (
      <div className="flex h-full min-h-[280px] flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
        <Sigma className="size-7 opacity-40" />
        <div>Nothing pooled yet.</div>
        <div className="text-xs max-w-xs">
          Review the extracted values, then click <strong>Run analyses</strong>.
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-2 p-1 text-sm">
      <div className="flex items-center gap-2 text-amber-700">
        <AlertTriangle className="size-4 shrink-0" />
        <span className="font-medium">
          {blocked.length} of {extractions.length} studies could not be pooled
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        A study needs either its raw per-arm numbers, or the effect with its 95% CI.
        Fill either one below and re-run. Extractions made before the reported-CI
        support was added will often fill in if you extract again.
      </p>
      <ul className="max-h-[320px] space-y-1 overflow-auto">
        {blocked.slice(0, 20).map((x, i) => (
          <li key={x.paper_id || i} className="rounded border bg-muted/30 px-2 py-1.5">
            <div className="truncate text-xs font-medium">{x.title}</div>
            <div className="text-[11px] text-amber-700">
              {x.error || "no effect size could be derived"}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function MetaAnalysisPage() {
  const s = useStore();
  const [extracting, setExtracting] = useState(false);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState<string>("forest");
  // The settings panel collapses once there is something to look at, so the
  // evidence gets the screen. Same behaviour as the question panel on Text
  // Extraction, which this page is the sibling of.
  const [setupOpen, setSetupOpen] = useState((s.metaExtractions || []).length === 0);

  // Evidence pane: which study is open, which of its tables is shown, and which
  // page the source pane is parked on. citedPage is set by clicking a citation,
  // so "p. 7" in a provenance line actually moves the document beside it.
  const [evidenceId, setEvidenceId] = useState<string>("");
  const [tables, setTables] = useState<Record<string, ExtractedTable[]>>({});
  const [tableIdx, setTableIdx] = useState(0);
  const [citedPage, setCitedPage] = useState<number | null>(null);
  const [loadingTables, setLoadingTables] = useState(false);

  const candidatePapers: Paper[] = useMemo(() => {
    if (s.results && s.results.length > 0) {
      return s.results
        .filter(r => String(r.Decision ?? "").toUpperCase().includes("INCLUDE"))
        .map(r => ({ id: r.paper_id, source: r.Source, title: r.Title, abstract: r.Abstract || "", url: r.URL }));
    }
    if (s.uniquePapers && s.uniquePapers.length > 0) return s.uniquePapers;
    return s.rawPapers || [];
  }, [s.results, s.uniquePapers, s.rawPapers]);

  const fullTextMap = useMemo<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const k of Object.keys(s.fullTexts || {})) {
      const ft = s.fullTexts[k];
      if (ft && ft.text) out[k] = ft.text;
    }
    return out;
  }, [s.fullTexts]);

  /** Extract effect sizes, then pool them. One action, because they are one
   *  job: extracting without pooling leaves the reviewer holding numbers and
   *  no answer, and pooling is meaningless without a fresh extraction behind
   *  it. The two used to be separate buttons and the second was the one people
   *  forgot to press. */
  async function runCalculate() {
    if (!canCalculate) return;
    setExtracting(true);
    try {
      const r = await MetaAnalysisService.extract(
        candidatePapers, s.metaOutcome, s.metaMeasure || "", fullTextMap, tablesByPaperId,
      );
      s.setMetaExtractions(r.extractions);
      setExtracting(false);

      // Pool the rows we just received rather than reading them back out of
      // the store, which has not re-rendered yet.
      setRunning(true);
      const out = await MetaAnalysisService.run(r.extractions, s.metaTau2Method, s.metaUseKnappHartung);
      s.setMetaRun(out);

      const k = out.pool?.k ?? 0;
      if (k === 0) {
        toast.error(`Extracted ${r.extractions.length} studies, but none had enough data to pool. Fill in the missing values below and calculate again.`);
      } else {
        toast.success(`Pooled ${k} of ${r.extractions.length} studies.`);
      }
    } catch (e: any) {
      toast.error(`Calculation failed: ${e?.message?.slice(0, 100) || "unknown error"}`);
    } finally {
      setExtracting(false);
      setRunning(false);
    }
  }

  /** Re-pool the rows already on screen, without calling the LLM again.
   *  Used after a hand-correction, and when the estimator settings change. */
  async function repool() {
    const rows = s.metaExtractions || [];
    if (rows.length === 0) return;
    setRunning(true);
    try {
      s.setMetaRun(await MetaAnalysisService.run(rows, s.metaTau2Method, s.metaUseKnappHartung));
    } catch (e: any) {
      toast.error(`Pooling failed: ${e?.message?.slice(0, 100) || "unknown error"}`);
    } finally {
      setRunning(false);
    }
  }

  const extractions = s.metaExtractions || [];
  const run = s.metaRun;
  const busy = extracting || running;

  // The review already named its outcome during Planning, and the included
  // studies already report theirs. Offering those beats making the reviewer
  // retype something the app knows, and a mistyped outcome here silently
  // changes what gets extracted.
  const outcomeSuggestions = useMemo(() => {
    const seen = new Set<string>();
    const out: { label: string; value: string; hint?: string }[] = [];
    const add = (value: string, hint?: string) => {
      const v = (value || "").trim();
      const key = v.toLowerCase();
      if (!v || seen.has(key) || v.length > 120) return;
      seen.add(key);
      out.push({ label: v.length > 48 ? `${v.slice(0, 48)}…` : v, value: v, hint: hint || v });
    };
    add(s.pico?.outcome || "", "The outcome declared in this review's PICO frame");
    // Outcomes the extractor has already named on a previous pass.
    for (const x of s.metaExtractions || []) add(x.outcome || "", "Already extracted on a previous pass");
    return out.slice(0, 5);
  }, [s.pico?.outcome, s.metaExtractions]);
  // What the calculation needs before it can mean anything: an outcome to pull,
  // and studies to pull it from. Stated as a reason so the button can say what
  // is missing rather than just sitting greyed out.
  const blocker = !s.metaOutcome.trim() ? "Name the outcome to meta-analyse." : "";
  const canCalculate = !busy && !blocker;
  // Once pooling has run, the pool is the authority on what it could not use.
  // Counting rows with a null yi instead double-counts: the backend derives yi
  // server-side, so a row can be perfectly poolable while the copy held here
  // still carries only the raw inputs it was extracted with.
  const unpoolableCount = (p: typeof run) =>
    p?.pool ? p.pool.invalid_studies.length
            : extractions.filter(x => x.yi === null || x.yi === undefined).length;

  // ── Evidence pane derivations ──────────────────────────────────────────────
  const evidenceStudy = useMemo(
    () => extractions.find(x => x.paper_id === evidenceId) || extractions[0],
    [extractions, evidenceId],
  );
  const evidenceFt = evidenceStudy ? s.fullTexts?.[evidenceStudy.paper_id] : undefined;
  const evidenceFullText = evidenceFt?.text;
  // AcquisitionPage stores this as a ready-made URL; SourcePane wants the key.
  const evidencePdfKey = useMemo(() => {
    const u = evidenceFt?.pdf_url || "";
    const m = u.match(/\/fulltext\/pdf\/([^/?#]+)/);
    return m ? decodeURIComponent(m[1]) : undefined;
  }, [evidenceFt?.pdf_url]);

  // Default the pane to the first extracted study.
  useEffect(() => {
    if (!evidenceId && extractions.length > 0) setEvidenceId(extractions[0].paper_id);
  }, [extractions, evidenceId]);

  // Park the source pane on whatever page the current extraction cites, so the
  // document opens where the number is rather than at page one.
  useEffect(() => {
    if (evidenceStudy?.source_page) setCitedPage(evidenceStudy.source_page);
  }, [evidenceStudy?.paper_id, evidenceStudy?.source_page]);

  // Tables the Table Extraction tab already pulled, indexed for reuse.
  //
  // Those results are keyed by title and URL rather than paper_id, so matching
  // is on normalised title with URL as a fallback. Worth the fuzziness: without
  // it this page re-runs an extraction the reviewer already ran, which on the
  // LLM tier means paying for the same tokens twice and waiting again, and can
  // silently produce DIFFERENT tables from the ones they already reviewed,
  // because that tier is not deterministic.
  const tablesAlreadyExtracted = useMemo(() => {
    const norm = (v: string) => (v || "").toLowerCase().replace(/\s+/g, " ").trim();
    const byTitle = new Map<string, ExtractedTable[]>();
    const byUrl = new Map<string, ExtractedTable[]>();
    for (const p of s.extractedPapers || []) {
      const t = p.Extracted_Tables || [];
      if (!t.length) continue;
      if (p.Paper_Title) byTitle.set(norm(p.Paper_Title), t);
      if (p.Paper_URL) byUrl.set(norm(p.Paper_URL), t);
    }
    return (title?: string, url?: string | null) =>
      byTitle.get(norm(title || "")) || (url ? byUrl.get(norm(url)) : undefined);
  }, [s.extractedPapers]);

  // Tables to hand the extractor, keyed by paper id. Reusing what the Table
  // Extraction stage already produced means the effect sizes are read out of a
  // results table rather than out of prose that lost its columns in the PDF.
  const tablesByPaperId = useMemo(() => {
    const out: Record<string, ExtractedTable[]> = {};
    for (const p of candidatePapers) {
      const t = tablesAlreadyExtracted(p.title, (p as any).url);
      if (t?.length) out[p.id] = t;
    }
    return out;
  }, [candidatePapers, tablesAlreadyExtracted]);

  // Tables are resolved once per study: from the Table Extraction tab if it has
  // them, and only otherwise from the backend.
  useEffect(() => {
    if (!evidenceStudy || tables[evidenceStudy.paper_id]) return;
    const reuse = tablesAlreadyExtracted(evidenceStudy.title, evidenceStudy.url);
    if (reuse?.length) {
      setTables(prev => ({ ...prev, [evidenceStudy.paper_id]: reuse }));
      return;
    }
    let cancelled = false;
    const id = evidenceStudy.paper_id;
    setLoadingTables(true);
    AIService.extractTables({
      paper_id: id,
      Title: evidenceStudy.title,
      URL: evidenceStudy.url || "",
      Source: "",
      Abstract: "",
      full_text: s.fullTexts?.[id]?.text || "",
    } as any)
      .then(t => { if (!cancelled) setTables(prev => ({ ...prev, [id]: t || [] })); })
      .catch(() => { if (!cancelled) setTables(prev => ({ ...prev, [id]: [] })); })
      .finally(() => { if (!cancelled) setLoadingTables(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evidenceStudy?.paper_id, tablesAlreadyExtracted]);
  const pool = run?.pool;
  const measures = (pool?.effect_measures || []) as EffectMeasure[];
  const primaryMeasure: EffectMeasure = measures.length === 1 ? measures[0] : "GENERIC";

  // Prerequisites, as an early return rather than a disabled page. Every other
  // stage does this: until there is something to work on, the only useful thing
  // on screen is the way to go and get it. Showing the controls greyed out
  // alongside an empty state says the same thing three times and invites the
  // reviewer to fill in settings that cannot run.
  if (!s.results) {
    return <EmptyState icon={Sigma} title="Screening not complete yet"
      description="Run abstract screening first. The studies it includes are what gets extracted and pooled here."
      action={{ label: "Go to Abstract Screening", onClick: () => s.setPage("abstract"), icon: Sigma }} />;
  }
  if (candidatePapers.length === 0) {
    return <EmptyState icon={Sigma} title="No included articles to meta-analyse"
      description="The current screening run included 0 articles, so there are no effect sizes to pool. Include some articles, or restore a run that has includes."
      action={{ label: "Go to Abstract Screening", onClick: () => s.setPage("abstract"), icon: Sigma }} />;
  }

  return (
    <div className="space-y-4">
      {/* Counts and actions, then settings. Same two-part header every other
          working tab uses (ControlPane + a collapsible panel), so this page
          reads as part of the app rather than a form bolted to the top of it. */}
      <ControlPane
        stats={extractions.length > 0 ? <>
          <InlineStat icon={FileText} value={extractions.length} label="Extracted" />
          <PaneDivider />
          {pool && pool.k > 0 && <InlineStat icon={ListChecks} tone="success" value={pool.k} label="Pooled" />}
          {unpoolableCount(run) > 0 && (
            <InlineStat icon={AlertTriangle} tone="amber" value={unpoolableCount(run)} label="Unpoolable" />
          )}
          {pool?.heterogeneity && (
            <InlineStat icon={Percent}
              tone={(pool.heterogeneity.I2_pct ?? 0) > 50 ? "amber" : "default"}
              value={`${num(pool.heterogeneity.I2_pct, 0)}%`} label="I² heterogeneity" />
          )}
        </> : (
          <InlineStat icon={FileText} value={candidatePapers.length} label="Ready to extract" />
        )}
        actions={<>
          {/* Say what is missing rather than leaving a greyed-out button and no
              explanation of what would un-grey it. Shown whether or not there
              are already extractions, since the outcome can be cleared after a
              run and the button then disables for no visible reason. */}
          {blocker && <span className="text-xs text-muted-foreground mr-1">{blocker}</span>}
          {extractions.length > 0 && !busy && (
            <Button size="sm" variant="ghost" className="h-8" onClick={repool}
                    title="Re-pool the rows below without calling the model again">
              Re-pool
            </Button>
          )}
          <Button size="sm" className="h-8 shadow-sm" onClick={runCalculate}
                  disabled={!canCalculate} title={blocker || undefined}>
            <Calculator className="size-3.5 mr-1.5" />
            {extracting ? "Extracting…" : running ? "Pooling…"
              : extractions.length > 0 ? "Recalculate" : `Calculate from ${candidatePapers.length}`}
          </Button>
        </>}
      />

      <Card className="p-3 space-y-2">
        <button
          onClick={() => setSetupOpen(o => !o)}
          className="flex items-center gap-2 text-sm font-medium w-full text-left min-w-0 hover:text-primary"
          title={setupOpen ? "Collapse settings" : "Edit settings"}
        >
          <ChevronDown className={`size-4 shrink-0 transition-transform ${setupOpen ? "" : "-rotate-90"}`} />
          <Sigma className="size-4 text-primary shrink-0" />
          {setupOpen ? "Analysis settings" : (
            <span className="truncate text-muted-foreground font-normal">
              {s.metaOutcome || "No outcome set"} · {s.metaMeasure || "auto-detect"} · {s.metaTau2Method}
              {s.metaUseKnappHartung ? " · Knapp-Hartung" : ""}
            </span>
          )}
        </button>

        {setupOpen && (
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_1fr] md:items-end">
            <div className="md:col-span-4">
              <Label className="text-[11px] text-muted-foreground">
                Outcome to meta-analyse
              </Label>
              <Textarea
                value={s.metaOutcome}
                onChange={e => s.setMetaOutcome(e.target.value)}
                rows={2}
                placeholder="Describe the outcome in plain language, with its units and time point. e.g. all-cause mortality at 12 months"
              />
              {outcomeSuggestions.length > 0 && (
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  {outcomeSuggestions.map((o, i) => (
                    <Button key={i} size="sm" variant="outline" className="h-7 text-xs"
                            onClick={() => s.setMetaOutcome(o.value)} title={o.hint}>
                      {o.label}
                    </Button>
                  ))}
                </div>
              )}
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">Effect measure</Label>
              <Select value={s.metaMeasure || "auto"} onValueChange={v => s.setMetaMeasure(v === "auto" ? "" : (v as EffectMeasure))}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  {MEASURE_GROUPS.map(g => (
                    <div key={g}>
                      <div className="px-2 py-1 text-[10px] uppercase text-muted-foreground">{g}</div>
                      {(Object.keys(MEASURE_META) as EffectMeasure[])
                        .filter(k => MEASURE_META[k].group === g && k !== "UNKNOWN")
                        .map(k => <SelectItem key={k} value={k}>{MEASURE_META[k].label}</SelectItem>)}
                    </div>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">τ² estimator</Label>
              <Select value={s.metaTau2Method} onValueChange={v => s.setMetaTau2Method(v as Tau2Method)}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="DL">DerSimonian-Laird</SelectItem>
                  <SelectItem value="PM">Paule-Mandel</SelectItem>
                  <SelectItem value="REML">REML</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex h-9 items-center gap-2">
              <Switch checked={s.metaUseKnappHartung} onCheckedChange={s.setMetaUseKnappHartung} id="kh-toggle" />
              <Label htmlFor="kh-toggle" className="cursor-pointer text-xs whitespace-nowrap"
                     title="Widens the random-effects interval using a t distribution. Recommended when pooling few studies.">
                Knapp-Hartung
              </Label>
            </div>
          </div>
        )}
      </Card>

      {/* Headline pooled summary */}
      {pool && pool.k > 0 && (
        <Card className="p-4">
          <div className="grid md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Random-effects pooled</div>
              <div className="font-mono tabular-nums">
                {num(pool.random!.estimate)} [{num(pool.random!.ci_low)}, {num(pool.random!.ci_high)}]
              </div>
              {isLogScale(measures) && (
                <div className="text-xs text-muted-foreground tabular-nums">
                  ≈ {expNum(pool.random!.estimate)} [{expNum(pool.random!.ci_low)}, {expNum(pool.random!.ci_high)}] (back-transformed)
                </div>
              )}
              <div className="text-xs text-muted-foreground">τ² = {num(pool.random!.tau2, 4)} · p = {num(pool.random!.p_value, 4)}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Fixed-effects pooled</div>
              <div className="font-mono tabular-nums">
                {num(pool.fixed!.estimate)} [{num(pool.fixed!.ci_low)}, {num(pool.fixed!.ci_high)}]
              </div>
              <div className="text-xs text-muted-foreground">p = {num(pool.fixed!.p_value, 4)}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Heterogeneity</div>
              <div className="font-mono tabular-nums">
                I² = {num(pool.heterogeneity!.I2_pct, 1)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Q = {num(pool.heterogeneity!.Q, 2)} (df {pool.heterogeneity!.df}, p = {num(pool.heterogeneity!.Q_p_value, 4)})
              </div>
              <div className="text-xs text-muted-foreground">
                {pool.heterogeneity!.I2_pct < 25 ? "Low" : pool.heterogeneity!.I2_pct < 50 ? "Moderate" : pool.heterogeneity!.I2_pct < 75 ? "Substantial" : "Considerable"} heterogeneity
              </div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">95% prediction interval</div>
              {pool.prediction_interval ? (
                <>
                  <div className="font-mono tabular-nums">
                    [{pool.prediction_interval.low.toFixed(3)}, {pool.prediction_interval.high.toFixed(3)}]
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Range a new study from the same population would likely produce.
                  </div>
                </>
              ) : (
                <div className="text-xs text-muted-foreground">(needs ≥3 studies + nonzero τ²)</div>
              )}
            </div>
          </div>
          <Separator className="my-3" />
          <div className="text-xs text-muted-foreground">
            <span className="font-medium">{pool.k}</span> studies pooled · effect measures: {measures.join(", ") || "n/a"} ·
            τ² method: {pool.tau2_method}{pool.use_knapp_hartung ? " · Knapp-Hartung adjusted" : ""}
            {pool.invalid_studies.length > 0 && <> · <span className="text-amber-700">{pool.invalid_studies.length} unpoolable</span></>}
          </div>
        </Card>
      )}

      {/* Evidence workspace. Gated only on having something extracted, NOT on a
          successful pool: when the numbers are wrong the reviewer needs the
          source document beside them more than ever, and hiding it until the
          statistics succeed removes the tool exactly when it is needed. */}
      {extractions.length > 0 && (
        <Card className="p-3">
            <div className="grid gap-3 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,1.45fr)]">
              <div className="h-[560px]">
                <SourcePane
                  title={evidenceStudy?.title}
                  pdfKey={evidencePdfKey}
                  page={citedPage}
                  url={evidenceStudy?.url}
                  text={evidenceFullText}
                />
              </div>

              <div className="space-y-3 min-w-0">
                {extractions.length > 1 && (
                  <select
                    className="w-full rounded border bg-background px-2 py-1 text-xs"
                    value={evidenceId}
                    onChange={e => { setEvidenceId(e.target.value); setTableIdx(0); setCitedPage(null); }}
                  >
                    {extractions.map(x => (
                      <option key={x.paper_id} value={x.paper_id}>{x.title}</option>
                    ))}
                  </select>
                )}

                {loadingTables ? (
                  <div className="rounded-lg border bg-card p-6 text-center text-sm text-muted-foreground">
                    Extracting tables...
                  </div>
                ) : (
                  <TablePane
                    tables={tables[evidenceId] || []}
                    selected={tableIdx}
                    onSelect={setTableIdx}
                    onCite={p => setCitedPage(p ?? null)}
                  />
                )}

                {evidenceStudy && (
                  <AskPane
                    paper={{ id: evidenceStudy.paper_id, title: evidenceStudy.title }}
                    fullText={evidenceFullText}
                    tables={tables[evidenceId] || []}
                    onCite={p => setCitedPage(p ?? null)}
                  />
                )}
              </div>

              <Card className="p-3 min-w-0">
                <div className="mb-2 text-xs font-medium text-muted-foreground">
                  Forest plot · {s.metaOutcome || "pooled effect"} · random effects
                </div>
                {pool && pool.k > 0 ? (
                  <div className="overflow-x-auto">
                    <ForestPlot
                      studies={pool.valid_studies}
                      pooled={pool.random}
                      predInterval={pool.prediction_interval}
                      measures={measures}
                      compact
                    />
                  </div>
                ) : (
                  <PoolBlocked extractions={extractions} ran={!!run} />
                )}
              </Card>
            </div>
        </Card>
      )}

      {/* Tabs */}
      {run && pool && pool.k > 0 && (
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="forest">Forest plot</TabsTrigger>
            <TabsTrigger value="studies">Studies ({extractions.length})</TabsTrigger>
            <TabsTrigger value="sensitivity">Sensitivity</TabsTrigger>
            <TabsTrigger value="bias">Publication bias</TabsTrigger>
            <TabsTrigger value="subgroups">Subgroups</TabsTrigger>
            <TabsTrigger value="regression">Meta-regression</TabsTrigger>
          </TabsList>

          {/* Forest */}
          <TabsContent value="forest">
            <Card className="p-4">
              <div className="overflow-x-auto">
                <ForestPlot
                  studies={pool.valid_studies}
                  pooled={pool.random}
                  predInterval={pool.prediction_interval}
                  measures={measures}
                />
              </div>
            </Card>
          </TabsContent>

          {/* Studies (editable rows) */}
          <TabsContent value="studies">
            <Card className="p-4 space-y-3">
              <div className="text-xs text-muted-foreground">
                Edit any cell to correct LLM extractions. Switching the effect measure clears the computed yi/SE so the next "Run analyses" will recompute.
              </div>
              <div className="space-y-2">
                {extractions.map((x, i) => (
                  <StudyRowEditor
                    key={x.paper_id || i}
                    study={x}
                    onChange={next => s.setMetaExtractions(arr => (arr || []).map((it, j) => (j === i ? next : it)))}
                    onRemove={() => s.setMetaExtractions(arr => (arr || []).filter((_, j) => j !== i))}
                  />
                ))}
              </div>
            </Card>
          </TabsContent>

          {/* Sensitivity */}
          <TabsContent value="sensitivity">
            <div className="grid md:grid-cols-2 gap-4">
              <Card className="p-4">
                <div className="text-sm font-medium mb-2">Leave-one-out</div>
                <div className="text-xs text-muted-foreground mb-3">Drop each study; large |Δ| flags an influential study.</div>
                <table className="w-full text-xs">
                  <thead><tr className="text-left text-muted-foreground">
                    <th className="py-1 pr-2">Excluded study</th>
                    <th className="py-1 pr-2 text-right">Pooled w/o</th>
                    <th className="py-1 text-right">Δ</th>
                  </tr></thead>
                  <tbody>
                    {run.leave_one_out
                      .slice()
                      .sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0))
                      .map((r, i) => (
                        <tr key={i} className="border-t border-border/40">
                          <td className="py-1 pr-2 truncate max-w-[200px]" title={r.title}>{r.title.slice(0, 36)}</td>
                          <td className="py-1 pr-2 text-right font-mono tabular-nums">{num(r.estimate_without)}</td>
                          <td className={`py-1 text-right font-mono tabular-nums ${Math.abs(r.delta ?? 0) > 0.1 ? "text-amber-700" : ""}`}>
                            {r.delta !== null ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(3) : "–"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </Card>
              <Card className="p-4">
                <div className="text-sm font-medium mb-2">Cumulative meta-analysis</div>
                <div className="text-xs text-muted-foreground mb-3">Sequential pooling in the order shown in Studies tab.</div>
                <table className="w-full text-xs">
                  <thead><tr className="text-left text-muted-foreground">
                    <th className="py-1 pr-2">k</th>
                    <th className="py-1 pr-2">Last added</th>
                    <th className="py-1 pr-2 text-right">Pooled</th>
                    <th className="py-1 text-right">I²</th>
                  </tr></thead>
                  <tbody>
                    {run.cumulative.map((r, i) => (
                      <tr key={i} className="border-t border-border/40">
                        <td className="py-1 pr-2 tabular-nums">{r.k}</td>
                        <td className="py-1 pr-2 truncate max-w-[180px]">{r.last_added.slice(0, 30)}</td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">{num(r.estimate)}</td>
                        <td className="py-1 text-right font-mono tabular-nums">{num(r.I2_pct, 0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>
          </TabsContent>

          {/* Publication bias */}
          <TabsContent value="bias">
            <Card className="p-4 space-y-4">
              <div className="overflow-x-auto flex justify-center">
                <FunnelPlot studies={run.funnel.studies} center={run.funnel.center} measure={primaryMeasure} />
              </div>
              <Separator />
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Egger's regression test</div>
                  {run.egger.intercept !== null ? (
                    <>
                      <div className="font-mono tabular-nums">intercept = {num(run.egger.intercept)} · p = {num(run.egger.p_value, 4)}</div>
                      <div className="text-xs text-muted-foreground">{(run.egger.p_value ?? 1) < 0.1 ? "Some evidence of funnel asymmetry." : "No strong evidence of small-study effects."}</div>
                    </>
                  ) : <div className="text-xs text-muted-foreground">{run.egger.note}</div>}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Begg's rank test</div>
                  {run.begg.tau !== null ? (
                    <>
                      <div className="font-mono tabular-nums">τ = {num(run.begg.tau)} · p = {num(run.begg.p_value, 4)}</div>
                      <div className="text-xs text-muted-foreground">Less powerful than Egger; corroborative.</div>
                    </>
                  ) : <div className="text-xs text-muted-foreground">{run.begg.note}</div>}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Trim-and-fill (Duval-Tweedie L0)</div>
                  {typeof run.trim_fill.filled_estimate === "number" ? (
                    <>
                      <div className="font-mono tabular-nums">
                        k₀ = {run.trim_fill.k0} ({run.trim_fill.side}-side) · adjusted = {num(run.trim_fill.filled_estimate)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Adjusted CI [{run.trim_fill.filled_ci_low!.toFixed(3)}, {run.trim_fill.filled_ci_high!.toFixed(3)}]
                      </div>
                    </>
                  ) : <div className="text-xs text-muted-foreground">{run.trim_fill.note}</div>}
                </div>
              </div>
            </Card>
          </TabsContent>

          {/* Subgroups */}
          <TabsContent value="subgroups">
            <Card className="p-4 space-y-3">
              <div className="text-sm">
                Group studies by the <span className="font-mono">subgroup</span> field in the Studies tab. Fill it in (e.g. "RCT", "cohort", "Asia") then re-run analyses.
              </div>
              {run.subgroup.groups.length === 0 ? (
                <div className="text-xs text-muted-foreground">No subgroups defined yet. Add categorical labels in the Studies tab.</div>
              ) : (
                <>
                  <table className="w-full text-sm">
                    <thead><tr className="text-left text-muted-foreground text-xs">
                      <th className="py-1 pr-2">Group</th>
                      <th className="py-1 pr-2 text-right">k</th>
                      <th className="py-1 pr-2 text-right">Pooled (RE)</th>
                      <th className="py-1 pr-2 text-right">95% CI</th>
                      <th className="py-1 text-right">I²</th>
                    </tr></thead>
                    <tbody>
                      {run.subgroup.groups.map((g, i) => (
                        <tr key={i} className="border-t border-border/40">
                          <td className="py-1 pr-2 font-medium">{g.name}</td>
                          <td className="py-1 pr-2 text-right tabular-nums">{g.k}</td>
                          <td className="py-1 pr-2 text-right font-mono tabular-nums">{num(g.estimate)}</td>
                          <td className="py-1 pr-2 text-right font-mono tabular-nums text-xs">
                            {g.ci_low !== null && g.ci_high !== null ? `[${g.ci_low.toFixed(3)}, ${g.ci_high.toFixed(3)}]` : "–"}
                          </td>
                          <td className="py-1 text-right tabular-nums">{num(g.I2_pct, 0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <Separator />
                  <div className="text-xs text-muted-foreground">
                    Between-group Q = <span className="font-mono">{num(run.subgroup.Q_between, 2)}</span>
                    {" "}(df = {run.subgroup.df_between}, p = {num(run.subgroup.Q_between_p, 4)}).{" "}
                    {run.subgroup.Q_between_p < 0.05
                      ? "Statistically significant heterogeneity between subgroups: the moderator matters."
                      : "No significant between-group differences."}
                  </div>
                </>
              )}
            </Card>
          </TabsContent>

          {/* Meta-regression */}
          <TabsContent value="regression">
            <Card className="p-4 space-y-3">
              <div className="text-sm">
                Mixed-effects meta-regression on the continuous <span className="font-mono">moderator</span> field (e.g. publication year, mean age, dose).
              </div>
              <div className="overflow-x-auto flex justify-center">
                <BubblePlot
                  studies={pool.valid_studies}
                  slope={run.meta_regression.slope}
                  intercept={run.meta_regression.intercept}
                  measure={primaryMeasure}
                />
              </div>
              <Separator />
              {run.meta_regression.slope !== null ? (
                <div className="grid md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Slope</div>
                    <div className="font-mono tabular-nums">
                      {run.meta_regression.slope.toFixed(4)}
                      {run.meta_regression.se !== null && <span className="text-xs text-muted-foreground"> ± {run.meta_regression.se!.toFixed(4)}</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">p-value</div>
                    <div className="font-mono tabular-nums">{run.meta_regression.p_value!.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">R²-analog</div>
                    <div className="font-mono tabular-nums">{((run.meta_regression.R2 ?? 0) * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Studies with moderator</div>
                    <div className="font-mono tabular-nums">{run.meta_regression.k}</div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">{run.meta_regression.note || "Add moderator values to study rows and re-run."}</div>
              )}
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {/* Show extractions before analysis is run */}
      {extractions.length > 0 && !run && (
        <Card className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-sm">Extracted effect sizes ({extractions.length})</div>
              <div className="text-xs text-muted-foreground">Review and edit, then click <strong>Run analyses</strong>.</div>
            </div>
          </div>
          <div className="space-y-2">
            {extractions.map((x, i) => (
              <StudyRowEditor
                key={x.paper_id || i}
                study={x}
                onChange={next => s.setMetaExtractions(arr => (arr || []).map((it, j) => (j === i ? next : it)))}
                onRemove={() => s.setMetaExtractions(arr => (arr || []).filter((_, j) => j !== i))}
              />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
