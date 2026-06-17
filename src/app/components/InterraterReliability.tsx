// Inter-rater reliability panel for the Diagramming page. Reads per-reviewer
// project decisions and computes the canonical agreement statistics:
//   - Cohen's κ for every reviewer pair                          (2 raters)
//   - Fleiss' κ across all reviewers                              (≥3 raters)
//   - Krippendorff's α (nominal)                                  (any N)
//   - Percent agreement (simple baseline)
//
// Cochrane's PRISMA-EBM extension expects κ to be reported in the methods
// section, so this panel also exports the matrix and per-pair table as SVG
// for inclusion in the final review write-up.

import { useEffect, useMemo, useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription } from "./ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Download, RefreshCcw, Users, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import {
  listDecisions, listProjectPapers, getProject,
  Decision, DecisionSummary, Adjudication, ProjectPaper, ProjectMember, DecisionValue, Stage,
} from "../lib/projects";

// ---- Statistics --------------------------------------------------------

const CATEGORIES: DecisionValue[] = ["include", "exclude", "maybe"];

function isFullDecision(d: Decision | DecisionSummary): d is Decision {
  return "reason" in d;
}

/** Build the per-paper × per-reviewer decision matrix. Missing cells = null. */
function buildMatrix(
  decisions: (Decision | DecisionSummary)[],
  paperIds: string[],
  reviewerIds: string[],
): Record<string, Record<string, DecisionValue | null>> {
  const out: Record<string, Record<string, DecisionValue | null>> = {};
  for (const pid of paperIds) {
    out[pid] = {};
    for (const rid of reviewerIds) out[pid][rid] = null;
  }
  for (const d of decisions) {
    if (!out[d.paper_id]) continue;
    out[d.paper_id][d.reviewer_user_id] = d.decision;
  }
  return out;
}

/** Cohen's κ between two raters on n papers. Missing cells (either rater
 *  hasn't decided) are dropped. */
function cohenKappa(
  a: (DecisionValue | null)[],
  b: (DecisionValue | null)[],
): { k: number; n: number; agree: number } {
  if (a.length !== b.length) throw new Error("Vectors must be same length");
  let n = 0, agree = 0;
  const ar = Object.fromEntries(CATEGORIES.map(c => [c, 0])) as Record<DecisionValue, number>;
  const br = Object.fromEntries(CATEGORIES.map(c => [c, 0])) as Record<DecisionValue, number>;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === null || b[i] === null) continue;
    n++;
    if (a[i] === b[i]) agree++;
    ar[a[i] as DecisionValue]++;
    br[b[i] as DecisionValue]++;
  }
  if (n === 0) return { k: NaN, n: 0, agree: 0 };
  const po = agree / n;
  let pe = 0;
  for (const c of CATEGORIES) {
    pe += (ar[c] / n) * (br[c] / n);
  }
  const k = pe === 1 ? 1 : (po - pe) / (1 - pe);
  return { k, n, agree };
}

/** Fleiss' κ across N raters. Subjects with < 2 raters who decided are dropped. */
function fleissKappa(
  matrix: Record<string, Record<string, DecisionValue | null>>,
  paperIds: string[],
  reviewerIds: string[],
): { k: number; n_subjects: number; n_raters: number; po: number; pe: number } {
  // Per subject, count how many raters voted each category (only counting raters who decided)
  const subjectCounts: Record<DecisionValue, number>[] = [];
  for (const pid of paperIds) {
    const counts = { include: 0, exclude: 0, maybe: 0 } as Record<DecisionValue, number>;
    let n = 0;
    for (const rid of reviewerIds) {
      const v = matrix[pid]?.[rid];
      if (v !== null && v !== undefined) {
        counts[v]++;
        n++;
      }
    }
    if (n >= 2) subjectCounts.push(counts);
  }
  if (subjectCounts.length === 0) {
    return { k: NaN, n_subjects: 0, n_raters: 0, po: NaN, pe: NaN };
  }
  // Use the most common number of raters across subjects. Fleiss strictly
  // assumes a constant N; for unequal we use the per-subject N inline.
  // P_i = (1 / (n_i*(n_i-1))) * Σ n_ij*(n_ij-1)
  let sumP = 0;
  let totalRaters = 0;
  const totalsPerCat = { include: 0, exclude: 0, maybe: 0 } as Record<DecisionValue, number>;
  for (const counts of subjectCounts) {
    const ni = counts.include + counts.exclude + counts.maybe;
    if (ni < 2) continue;
    let s = 0;
    for (const c of CATEGORIES) s += counts[c] * (counts[c] - 1);
    sumP += s / (ni * (ni - 1));
    totalRaters += ni;
    for (const c of CATEGORIES) totalsPerCat[c] += counts[c];
  }
  const po = sumP / subjectCounts.length;
  const p_j: Record<DecisionValue, number> = {
    include: totalsPerCat.include / totalRaters,
    exclude: totalsPerCat.exclude / totalRaters,
    maybe: totalsPerCat.maybe / totalRaters,
  };
  const pe = CATEGORIES.reduce((s, c) => s + p_j[c] ** 2, 0);
  const k = pe === 1 ? 1 : (po - pe) / (1 - pe);
  return { k, n_subjects: subjectCounts.length, n_raters: totalRaters, po, pe };
}

/** Krippendorff's α (nominal) across all rater-paper pairs. Robust to
 *  missing data. Returns α and the count of valued units. */
function krippendorffAlphaNominal(
  matrix: Record<string, Record<string, DecisionValue | null>>,
  paperIds: string[],
  reviewerIds: string[],
): { alpha: number; n_pairable: number } {
  // Build coincidence matrix o_{ck} : how many times category c and k
  // appear together within the same subject across all pairs of raters.
  const cats: DecisionValue[] = CATEGORIES;
  const coinc: Record<DecisionValue, Record<DecisionValue, number>> =
    Object.fromEntries(cats.map(c => [c, Object.fromEntries(cats.map(k => [k, 0])) as Record<DecisionValue, number>])) as any;

  let nPairable = 0;
  for (const pid of paperIds) {
    const vals: DecisionValue[] = [];
    for (const rid of reviewerIds) {
      const v = matrix[pid]?.[rid];
      if (v !== null && v !== undefined) vals.push(v);
    }
    if (vals.length < 2) continue;
    nPairable += vals.length;
    const m = vals.length;
    for (let i = 0; i < vals.length; i++) {
      for (let j = 0; j < vals.length; j++) {
        if (i === j) continue;
        coinc[vals[i]][vals[j]] += 1 / (m - 1);
      }
    }
  }
  if (nPairable < 2) return { alpha: NaN, n_pairable: 0 };

  // n_c = Σ_k o_{ck}
  const n_c: Record<DecisionValue, number> = { include: 0, exclude: 0, maybe: 0 };
  for (const c of cats) for (const k of cats) n_c[c] += coinc[c][k];
  const n = cats.reduce((s, c) => s + n_c[c], 0);

  // Observed disagreement Do (nominal δ(c,k) = 1 if c≠k else 0)
  let Do = 0;
  for (const c of cats) {
    for (const k of cats) {
      if (c !== k) Do += coinc[c][k];
    }
  }
  Do /= n;

  // Expected disagreement De
  let De = 0;
  for (const c of cats) {
    for (const k of cats) {
      if (c !== k) De += n_c[c] * n_c[k];
    }
  }
  De /= n * (n - 1);

  const alpha = De === 0 ? 1 : 1 - Do / De;
  return { alpha, n_pairable: nPairable };
}

function landisKochLabel(k: number): { label: string; cls: string } {
  if (Number.isNaN(k))   return { label: "n/a",           cls: "bg-slate-50 text-slate-500 border-slate-200" };
  if (k < 0)             return { label: "Poor",          cls: "bg-rose-50 text-rose-700 border-rose-200" };
  if (k < 0.2)           return { label: "Slight",        cls: "bg-rose-50 text-rose-700 border-rose-200" };
  if (k < 0.4)           return { label: "Fair",          cls: "bg-amber-50 text-amber-700 border-amber-200" };
  if (k < 0.6)           return { label: "Moderate",      cls: "bg-amber-50 text-amber-700 border-amber-200" };
  if (k < 0.8)           return { label: "Substantial",   cls: "bg-emerald-50 text-emerald-700 border-emerald-200" };
  return                       { label: "Almost perfect", cls: "bg-emerald-100 text-emerald-800 border-emerald-300" };
}

// ---- Component ---------------------------------------------------------

export function InterraterReliability({ projectId }: { projectId: string }) {
  const [decisions, setDecisions] = useState<(Decision | DecisionSummary)[]>([]);
  const [adjudications, setAdjudications] = useState<Adjudication[]>([]);
  const [papers, setPapers] = useState<ProjectPaper[]>([]);
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<Stage>("abstract");
  const [blinded, setBlinded] = useState(false);

  async function reload() {
    setLoading(true);
    try {
      const [decRes, papersRes, projRes] = await Promise.all([
        listDecisions(projectId, stage),
        listProjectPapers(projectId),
        getProject(projectId),
      ]);
      setDecisions(decRes.decisions);
      setAdjudications(decRes.adjudications);
      setBlinded(!!decRes.blinded);
      setPapers(papersRes);
      setMembers(projRes.members);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load decisions");
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { reload(); }, [projectId, stage]);

  // Filter members to only those who can decide (lead, reviewer, adjudicator).
  const decidingMembers = useMemo(
    () => members.filter(m => m.role === "lead" || m.role === "reviewer" || m.role === "adjudicator"),
    [members],
  );
  const reviewerIds = decidingMembers.map(m => m.user_id);
  const paperIds = papers.map(p => p.paper_id);

  const matrix = useMemo(
    () => buildMatrix(decisions, paperIds, reviewerIds),
    [decisions, paperIds, reviewerIds],
  );

  // Pairwise Cohen's κ
  const pairs = useMemo(() => {
    const out: { a: string; b: string; k: number; n: number; agree: number }[] = [];
    for (let i = 0; i < reviewerIds.length; i++) {
      for (let j = i + 1; j < reviewerIds.length; j++) {
        const aVec = paperIds.map(pid => matrix[pid][reviewerIds[i]]);
        const bVec = paperIds.map(pid => matrix[pid][reviewerIds[j]]);
        const r = cohenKappa(aVec, bVec);
        out.push({ a: reviewerIds[i], b: reviewerIds[j], ...r });
      }
    }
    return out;
  }, [matrix, paperIds, reviewerIds]);

  const fleiss = useMemo(() => fleissKappa(matrix, paperIds, reviewerIds), [matrix, paperIds, reviewerIds]);
  const alpha = useMemo(() => krippendorffAlphaNominal(matrix, paperIds, reviewerIds), [matrix, paperIds, reviewerIds]);

  // Coverage stats: how complete is the matrix?
  const totalCells = paperIds.length * reviewerIds.length;
  const filledCells = decisions.length;
  const coverage = totalCells > 0 ? filledCells / totalCells : 0;

  // ----- SVG export ----
  function exportSvg() {
    const svg = buildIrrSvg({
      projectId, stage, papers, reviewerIds, matrix,
      pairs, fleiss, alpha, coverage,
    });
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `interrater-${stage}-${new Date().toISOString().slice(0, 10)}.svg`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Inter-rater report exported.");
  }

  if (reviewerIds.length < 2) {
    return (
      <Card className="p-4">
        <Alert>
          <Users className="size-4 inline mr-1" />
          <AlertDescription className="text-xs">
            Inter-rater reliability requires at least two reviewers with decisions. This project currently has {reviewerIds.length} reviewer{reviewerIds.length === 1 ? "" : "s"}. Invite more reviewers (and have them screen the same papers) to populate this view.
          </AlertDescription>
        </Alert>
      </Card>
    );
  }

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Users className="size-4 text-primary" />
          <div className="font-medium">Inter-rater reliability</div>
        </div>
        <div className="flex items-center gap-2">
          <Select value={stage} onValueChange={(v) => setStage(v as Stage)}>
            <SelectTrigger className="w-44 h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="abstract">Abstract screening</SelectItem>
              <SelectItem value="fulltext">Full-text screening</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="ghost" size="sm" onClick={reload} disabled={loading} className="h-8">
            <RefreshCcw className={`size-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Button variant="outline" size="sm" onClick={exportSvg} className="h-8">
            <Download className="size-3 mr-1" />Export SVG
          </Button>
        </div>
      </div>

      {blinded && (
        <Alert className="border-amber-200 bg-amber-50">
          <AlertTriangle className="size-4 inline mr-1 text-amber-700" />
          <AlertDescription className="text-xs">
            Your role is reviewer in a dual-blinded project. Other reviewers' decisions are hidden until both have decided. Statistics shown below may be incomplete; the lead and adjudicator see the full matrix.
          </AlertDescription>
        </Alert>
      )}

      {/* ----- Summary stats ----- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatBlock
          label="Fleiss' κ"
          value={Number.isFinite(fleiss.k) ? fleiss.k.toFixed(3) : "—"}
          sublabel={`${fleiss.n_subjects} subjects, ${fleiss.n_raters} ratings`}
          badge={landisKochLabel(fleiss.k)}
        />
        <StatBlock
          label="Krippendorff's α"
          value={Number.isFinite(alpha.alpha) ? alpha.alpha.toFixed(3) : "—"}
          sublabel={`${alpha.n_pairable} pairable ratings`}
          badge={landisKochLabel(alpha.alpha)}
        />
        <StatBlock
          label="Mean Cohen's κ"
          value={pairs.length > 0 ? meanFinite(pairs.map(p => p.k)).toFixed(3) : "—"}
          sublabel={`${pairs.length} reviewer pairs`}
          badge={landisKochLabel(pairs.length > 0 ? meanFinite(pairs.map(p => p.k)) : NaN)}
        />
        <StatBlock
          label="Coverage"
          value={`${(coverage * 100).toFixed(0)}%`}
          sublabel={`${filledCells} of ${totalCells} cells filled`}
        />
      </div>

      {/* ----- Pairwise Cohen's κ table ----- */}
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Pairwise Cohen's κ</div>
        <div className="rounded border overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted">
              <tr>
                <th className="text-left px-2 py-1.5 font-medium">Reviewer A</th>
                <th className="text-left px-2 py-1.5 font-medium">Reviewer B</th>
                <th className="text-right px-2 py-1.5 font-medium">n papers</th>
                <th className="text-right px-2 py-1.5 font-medium">agree</th>
                <th className="text-right px-2 py-1.5 font-medium">κ</th>
                <th className="text-left px-2 py-1.5 font-medium">interpretation</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, i) => {
                const lk = landisKochLabel(p.k);
                return (
                  <tr key={i} className="border-t">
                    <td className="px-2 py-1.5 font-mono">{p.a.slice(0, 8)}…</td>
                    <td className="px-2 py-1.5 font-mono">{p.b.slice(0, 8)}…</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{p.n}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{p.agree}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums font-medium">
                      {Number.isFinite(p.k) ? p.k.toFixed(3) : "—"}
                    </td>
                    <td className="px-2 py-1.5">
                      <Badge variant="outline" className={`text-[10px] ${lk.cls}`}>{lk.label}</Badge>
                    </td>
                  </tr>
                );
              })}
              {pairs.length === 0 && (
                <tr><td colSpan={6} className="px-2 py-3 text-muted-foreground text-center">No pairwise data yet — reviewers need to decide on overlapping papers.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ----- Decision matrix heatmap ----- */}
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Decision matrix</div>
        <div className="text-xs text-muted-foreground">
          One row per paper, one column per reviewer. Green = include, red = exclude, amber = maybe, grey = not yet decided. Adjudicated papers ({adjudications.length}) are marked with a violet ring on every cell.
        </div>
        <DecisionMatrixGrid
          paperIds={paperIds}
          paperTitles={Object.fromEntries(papers.map(p => [p.paper_id, p.title]))}
          reviewerIds={reviewerIds}
          matrix={matrix}
          adjudicatedSet={new Set(adjudications.map(a => a.paper_id))}
        />
      </div>
    </Card>
  );
}

function StatBlock({ label, value, sublabel, badge }: {
  label: string; value: string; sublabel?: string;
  badge?: { label: string; cls: string };
}) {
  return (
    <Card className="p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        {badge && <Badge variant="outline" className={`text-[10px] ${badge.cls}`}>{badge.label}</Badge>}
      </div>
      <div className="text-2xl font-semibold tabular-nums">{value}</div>
      {sublabel && <div className="text-[11px] text-muted-foreground">{sublabel}</div>}
    </Card>
  );
}

function DecisionMatrixGrid({
  paperIds, paperTitles, reviewerIds, matrix, adjudicatedSet,
}: {
  paperIds: string[];
  paperTitles: Record<string, string>;
  reviewerIds: string[];
  matrix: Record<string, Record<string, DecisionValue | null>>;
  adjudicatedSet: Set<string>;
}) {
  const cellFor = (v: DecisionValue | null): string =>
    v === "include" ? "bg-emerald-500" :
    v === "exclude" ? "bg-rose-500" :
    v === "maybe"   ? "bg-amber-500" :
                      "bg-slate-200";

  return (
    <div className="rounded border overflow-x-auto max-h-[420px]">
      <table className="text-xs">
        <thead className="bg-muted sticky top-0 z-10">
          <tr>
            <th className="text-left px-2 py-1 sticky left-0 bg-muted z-20 max-w-[260px]">Paper</th>
            {reviewerIds.map(rid => (
              <th key={rid} className="px-1 py-1 text-center font-mono text-[10px]" title={rid}>
                {rid.slice(0, 6)}…
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paperIds.map(pid => (
            <tr key={pid} className="border-t">
              <td className="px-2 py-1 sticky left-0 bg-card max-w-[260px] truncate" title={paperTitles[pid]}>
                {paperTitles[pid] || pid}
              </td>
              {reviewerIds.map(rid => {
                const v = matrix[pid]?.[rid] ?? null;
                const adjudicated = adjudicatedSet.has(pid);
                return (
                  <td key={rid} className="px-1 py-1 text-center">
                    <span
                      title={`${paperTitles[pid] || pid} × ${rid}: ${v || "no decision"}${adjudicated ? " (adjudicated)" : ""}`}
                      className={`inline-block size-3 rounded ${cellFor(v)} ${adjudicated ? "ring-1 ring-violet-400 ring-offset-1 ring-offset-card" : ""}`}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
          {paperIds.length === 0 && (
            <tr><td colSpan={reviewerIds.length + 1} className="px-2 py-3 text-center text-muted-foreground">No papers in this project.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// ---- SVG export -------------------------------------------------------

function escapeXml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max - 1) + "…";
}

function meanFinite(xs: number[]): number {
  const finite = xs.filter(Number.isFinite);
  return finite.length === 0 ? NaN : finite.reduce((a, b) => a + b, 0) / finite.length;
}

function buildIrrSvg({
  stage, papers, reviewerIds, matrix, pairs, fleiss, alpha, coverage,
}: {
  projectId: string;
  stage: Stage;
  papers: ProjectPaper[];
  reviewerIds: string[];
  matrix: Record<string, Record<string, DecisionValue | null>>;
  pairs: { a: string; b: string; k: number; n: number; agree: number }[];
  fleiss: { k: number; n_subjects: number; n_raters: number; po: number; pe: number };
  alpha: { alpha: number; n_pairable: number };
  coverage: number;
}): string {
  const PAD = 16;
  const TITLE_W = 240;
  const CELL = 18;
  const HEADER_H = 32;
  const ROW_H = CELL;

  const gridX = PAD + TITLE_W;
  const gridY = PAD + 160 + HEADER_H;          // 160 = summary block
  const totalW = gridX + reviewerIds.length * CELL + PAD;
  const totalH = gridY + papers.length * ROW_H + PAD + 24 + pairs.length * 16;

  const colourFor = (v: DecisionValue | null): string =>
    v === "include" ? "#10b981" :
    v === "exclude" ? "#f43f5e" :
    v === "maybe"   ? "#f59e0b" :
                      "#e2e8f0";

  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${totalW} ${totalH}" width="${totalW}" height="${totalH}" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">`);
  out.push(`<rect width="${totalW}" height="${totalH}" fill="#ffffff"/>`);

  // Title + summary
  out.push(`<text x="${PAD}" y="${PAD + 14}" font-size="14" font-weight="600">Inter-rater reliability — ${stage}</text>`);
  out.push(`<text x="${PAD}" y="${PAD + 40}" font-size="11" fill="#475569">Fleiss' κ = ${Number.isFinite(fleiss.k) ? fleiss.k.toFixed(3) : "—"}   ·   Krippendorff's α = ${Number.isFinite(alpha.alpha) ? alpha.alpha.toFixed(3) : "—"}   ·   ${reviewerIds.length} reviewers   ·   ${papers.length} papers   ·   coverage ${(coverage * 100).toFixed(0)}%</text>`);

  // Pairwise table inline
  let py = PAD + 60;
  out.push(`<text x="${PAD}" y="${py}" font-size="11" font-weight="600" fill="#0f172a">Pairwise Cohen's κ</text>`);
  py += 14;
  for (const p of pairs.slice(0, 8)) {
    out.push(`<text x="${PAD}" y="${py}" font-size="10" fill="#334155">${escapeXml(p.a.slice(0, 8))}… × ${escapeXml(p.b.slice(0, 8))}…   κ = ${Number.isFinite(p.k) ? p.k.toFixed(3) : "—"}   (${p.agree}/${p.n} agree)</text>`);
    py += 13;
  }

  // Column headers
  reviewerIds.forEach((rid, i) => {
    const x = gridX + i * CELL + CELL / 2;
    const y = gridY - 6;
    out.push(`<text x="${x}" y="${y}" font-size="9" fill="#475569" text-anchor="start" transform="rotate(-60 ${x} ${y})">${escapeXml(truncate(rid, 8))}</text>`);
  });

  // Grid rows
  papers.forEach((p, ri) => {
    const yMid = gridY + ri * ROW_H + ROW_H / 2;
    out.push(`<text x="${PAD}" y="${yMid + 3}" font-size="9" fill="#0f172a">${escapeXml(truncate(p.title, 32))}</text>`);
    reviewerIds.forEach((rid, ci) => {
      const cx = gridX + ci * CELL + CELL / 2;
      const v = matrix[p.paper_id]?.[rid] ?? null;
      out.push(`<rect x="${cx - 6}" y="${yMid - 6}" width="12" height="12" rx="2" ry="2" fill="${colourFor(v)}"/>`);
    });
  });

  out.push(`</svg>`);
  return out.join("");
}
