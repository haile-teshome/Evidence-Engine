import { useMemo, useState } from "react";
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
import { Label } from "../components/ui/label";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Sigma, Play, Calculator, X as XIcon, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

// ===========================================================================
// Effect-measure metadata
// ===========================================================================

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

function exponentiateLabel(measure: EffectMeasure, v: number): string {
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
}: {
  studies: StudyEffect[];
  pooled: { estimate: number; ci_low: number; ci_high: number } | null;
  predInterval: { low: number; high: number } | null;
  measures: EffectMeasure[];
}) {
  const rowH = 24;
  const padTop = 30;
  const padBottom = 60;
  const padLeftLabel = 280;
  const padRightWeights = 160;
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
        const eff = log ? exponentiateLabel(s.effect_measure, s.yi) : s.yi.toFixed(2);
        const effLo = log ? exponentiateLabel(s.effect_measure, s.ci_low) : s.ci_low.toFixed(2);
        const effHi = log ? exponentiateLabel(s.effect_measure, s.ci_high) : s.ci_high.toFixed(2);
        return (
          <g key={s.paper_id || i}>
            <text x={6} y={rowY + 4} fontSize={11} fill="currentColor">
              <title>{s.title}</title>
              {(s.title || "(untitled)").slice(0, 50)}
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
        const eff = log ? exponentiateLabel(measure, pooled.estimate) : pooled.estimate.toFixed(3);
        const effLo = log ? exponentiateLabel(measure, pooled.ci_low) : pooled.ci_low.toFixed(3);
        const effHi = log ? exponentiateLabel(measure, pooled.ci_high) : pooled.ci_high.toFixed(3);
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
        const piLo = log ? exponentiateLabel(measure, predInterval.low) : predInterval.low.toFixed(3);
        const piHi = log ? exponentiateLabel(measure, predInterval.high) : predInterval.high.toFixed(3);
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
// Study row editor — switches input fields based on effect_measure
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
                yi={study.yi.toFixed(3)}  SE={study.se?.toFixed(3)}
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

export function MetaAnalysisPage() {
  const s = useStore();
  const [extracting, setExtracting] = useState(false);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState<string>("forest");

  const candidatePapers: Paper[] = useMemo(() => {
    if (s.results && s.results.length > 0) {
      return s.results
        .filter(r => r.Decision.toUpperCase().includes("INCLUDE"))
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

  async function runExtraction() {
    if (!s.metaOutcome.trim()) { toast.error("Specify the outcome you want to meta-analyse."); return; }
    if (candidatePapers.length === 0) { toast.error("No included papers available. Run abstract screening first."); return; }
    setExtracting(true);
    try {
      const r = await MetaAnalysisService.extract(candidatePapers, s.metaOutcome, s.metaMeasure || "", fullTextMap);
      s.setMetaExtractions(r.extractions);
      s.setMetaRun(null);
      const ok = r.extractions.filter(x => typeof x.yi === "number").length;
      toast.success(`Extracted effect-size data from ${ok} of ${r.extractions.length} papers.`);
    } catch (e: any) {
      toast.error(`Extraction failed: ${e?.message?.slice(0, 80) || "unknown error"}`);
    } finally {
      setExtracting(false);
    }
  }

  async function runAnalysis() {
    if (!s.metaExtractions || s.metaExtractions.length === 0) { toast.error("Run extraction first."); return; }
    setRunning(true);
    try {
      const r = await MetaAnalysisService.run(s.metaExtractions, s.metaTau2Method, s.metaUseKnappHartung);
      s.setMetaRun(r);
      if (r.pool.k === 0) toast.error("No studies had complete enough data to pool. Edit rows and re-run.");
      else toast.success(`Pooled ${r.pool.k} study${r.pool.k === 1 ? "" : "ies"}.`);
    } catch (e: any) {
      toast.error(`Analysis failed: ${e?.message?.slice(0, 80) || "unknown error"}`);
    } finally {
      setRunning(false);
    }
  }

  const extractions = s.metaExtractions || [];
  const run = s.metaRun;
  const pool = run?.pool;
  const measures = (pool?.effect_measures || []) as EffectMeasure[];
  const primaryMeasure: EffectMeasure = measures.length === 1 ? measures[0] : "GENERIC";

  return (
    <div className="space-y-4">
      {/* Setup card */}
      <Card className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Sigma className="size-5 text-primary" />
          <div className="flex-1">
            <div className="font-medium">Meta-analysis agent</div>
            <div className="text-xs text-muted-foreground">
              LLM-extracts effect sizes (OR / RR / RD / HR / MD / SMD / proportions / incidence / correlations / generic), then pools with FE + DL/PM/REML random-effects, plus subgroup / sensitivity / publication-bias / meta-regression diagnostics.
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          <div className="md:col-span-2">
            <Label className="text-sm">Outcome of interest</Label>
            <Input
              value={s.metaOutcome}
              onChange={e => s.setMetaOutcome(e.target.value)}
              placeholder="e.g., all-cause mortality at 12 months; depression score reduction; incidence of MI"
            />
            <p className="text-xs text-muted-foreground mt-1">Specific outcome → better extractions. Mention units / time point.</p>
          </div>
          <div>
            <Label className="text-sm">Preferred effect measure (optional hint)</Label>
            <Select value={s.metaMeasure || "auto"} onValueChange={v => s.setMetaMeasure(v === "auto" ? "" : (v as EffectMeasure))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
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
        </div>

        <div className="grid md:grid-cols-3 gap-3 items-end">
          <div>
            <Label className="text-sm">τ² estimator</Label>
            <Select value={s.metaTau2Method} onValueChange={v => s.setMetaTau2Method(v as Tau2Method)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="DL">DerSimonian-Laird (default)</SelectItem>
                <SelectItem value="PM">Paule-Mandel</SelectItem>
                <SelectItem value="REML">REML (restricted ML)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={s.metaUseKnappHartung} onCheckedChange={s.setMetaUseKnappHartung} id="kh-toggle" />
            <Label htmlFor="kh-toggle" className="text-sm cursor-pointer">Knapp-Hartung CI adjustment</Label>
          </div>
          <div className="flex gap-2 flex-wrap justify-end">
            <Button onClick={runExtraction} disabled={extracting || candidatePapers.length === 0}>
              <Play className="size-4 mr-2" />{extracting ? "Extracting…" : `Extract from ${candidatePapers.length}`}
            </Button>
            <Button variant="secondary" onClick={runAnalysis} disabled={running || extractions.length === 0}>
              <Calculator className="size-4 mr-2" />{running ? "Running…" : "Run analyses"}
            </Button>
          </div>
        </div>

        {candidatePapers.length === 0 && (
          <Alert><AlertDescription>No source papers found. Run Abstract Screening first — included papers feed the meta-analysis.</AlertDescription></Alert>
        )}
      </Card>

      {/* Headline pooled summary */}
      {pool && pool.k > 0 && (
        <Card className="p-4">
          <div className="grid md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Random-effects pooled</div>
              <div className="font-mono tabular-nums">
                {pool.random!.estimate.toFixed(3)} [{pool.random!.ci_low.toFixed(3)}, {pool.random!.ci_high.toFixed(3)}]
              </div>
              {isLogScale(measures) && (
                <div className="text-xs text-muted-foreground tabular-nums">
                  ≈ {Math.exp(pool.random!.estimate).toFixed(2)} [{Math.exp(pool.random!.ci_low).toFixed(2)}, {Math.exp(pool.random!.ci_high).toFixed(2)}] (back-transformed)
                </div>
              )}
              <div className="text-xs text-muted-foreground">τ² = {pool.random!.tau2.toFixed(4)} · p = {pool.random!.p_value?.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Fixed-effects pooled</div>
              <div className="font-mono tabular-nums">
                {pool.fixed!.estimate.toFixed(3)} [{pool.fixed!.ci_low.toFixed(3)}, {pool.fixed!.ci_high.toFixed(3)}]
              </div>
              <div className="text-xs text-muted-foreground">p = {pool.fixed!.p_value?.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Heterogeneity</div>
              <div className="font-mono tabular-nums">
                I² = {pool.heterogeneity!.I2_pct.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Q = {pool.heterogeneity!.Q.toFixed(2)} (df {pool.heterogeneity!.df}, p = {pool.heterogeneity!.Q_p_value.toFixed(4)})
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
                          <td className="py-1 pr-2 text-right font-mono tabular-nums">{r.estimate_without?.toFixed(3) ?? "—"}</td>
                          <td className={`py-1 text-right font-mono tabular-nums ${Math.abs(r.delta ?? 0) > 0.1 ? "text-amber-700" : ""}`}>
                            {r.delta !== null ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(3) : "—"}
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
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">{r.estimate?.toFixed(3) ?? "—"}</td>
                        <td className="py-1 text-right font-mono tabular-nums">{r.I2_pct?.toFixed(0)}%</td>
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
                      <div className="font-mono tabular-nums">intercept = {run.egger.intercept.toFixed(3)} · p = {run.egger.p_value!.toFixed(4)}</div>
                      <div className="text-xs text-muted-foreground">{(run.egger.p_value ?? 1) < 0.1 ? "Some evidence of funnel asymmetry." : "No strong evidence of small-study effects."}</div>
                    </>
                  ) : <div className="text-xs text-muted-foreground">{run.egger.note}</div>}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Begg's rank test</div>
                  {run.begg.tau !== null ? (
                    <>
                      <div className="font-mono tabular-nums">τ = {run.begg.tau.toFixed(3)} · p = {run.begg.p_value!.toFixed(4)}</div>
                      <div className="text-xs text-muted-foreground">Less powerful than Egger; corroborative.</div>
                    </>
                  ) : <div className="text-xs text-muted-foreground">{run.begg.note}</div>}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Trim-and-fill (Duval-Tweedie L0)</div>
                  {typeof run.trim_fill.filled_estimate === "number" ? (
                    <>
                      <div className="font-mono tabular-nums">
                        k₀ = {run.trim_fill.k0} ({run.trim_fill.side}-side) · adjusted = {run.trim_fill.filled_estimate.toFixed(3)}
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
                          <td className="py-1 pr-2 text-right font-mono tabular-nums">{g.estimate?.toFixed(3) ?? "—"}</td>
                          <td className="py-1 pr-2 text-right font-mono tabular-nums text-xs">
                            {g.ci_low !== null && g.ci_high !== null ? `[${g.ci_low.toFixed(3)}, ${g.ci_high.toFixed(3)}]` : "—"}
                          </td>
                          <td className="py-1 text-right tabular-nums">{g.I2_pct?.toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <Separator />
                  <div className="text-xs text-muted-foreground">
                    Between-group Q = <span className="font-mono">{run.subgroup.Q_between.toFixed(2)}</span>
                    {" "}(df = {run.subgroup.df_between}, p = {run.subgroup.Q_between_p.toFixed(4)}).{" "}
                    {run.subgroup.Q_between_p < 0.05
                      ? "Statistically significant heterogeneity between subgroups — the moderator matters."
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

      {/* Empty state */}
      {extractions.length === 0 && candidatePapers.length > 0 && (
        <Alert><AlertDescription>
          Specify an outcome above and click <strong>Extract from {candidatePapers.length}</strong> to run the LLM extractor.
        </AlertDescription></Alert>
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
