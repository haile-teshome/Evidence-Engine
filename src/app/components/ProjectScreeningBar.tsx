import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import { useAuth } from "../lib/auth";
import { listDecisions, writeDecision, Decision, DecisionSummary, Adjudication, DecisionValue, effectiveProjectDecision } from "../lib/projects";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Alert, AlertDescription } from "./ui/alert";
import { Users, Eye, EyeOff, ShieldCheck, RefreshCcw, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

/** Banner shown above Abstract Screening (and any decision-bearing page) when
 *  the user is operating inside a project. Displays the project name, mode,
 *  my role, my screening progress, and other reviewers' progress (with
 *  blinding rules honoured server-side).
 *
 *  Also exposes a synchronous `recordDecision` helper that screening pages
 *  use to push a single decision into the project. Callers can subscribe to
 *  refresh signals via the optional `onRefresh` callback.
 */
export function ProjectScreeningBar({
  stage = "abstract",
  papers,
  refreshSignal,
}: {
  stage?: "abstract" | "fulltext";
  papers: { id: string }[];
  refreshSignal?: number;
}) {
  const s = useStore();
  const { user } = useAuth();
  const [decisions, setDecisions] = useState<(Decision | DecisionSummary)[]>([]);
  const [adjudications, setAdjudications] = useState<Adjudication[]>([]);
  const [blinded, setBlinded] = useState(false);
  const [loading, setLoading] = useState(false);

  async function reload() {
    if (!s.currentProjectId) return;
    setLoading(true);
    try {
      const r = await listDecisions(s.currentProjectId, stage);
      setDecisions(r.decisions);
      setAdjudications(r.adjudications);
      setBlinded(!!r.blinded);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load project decisions");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { reload(); }, [s.currentProjectId, stage, refreshSignal]);

  if (!s.currentProjectId || !user) return null;

  // Compute my own progress vs other reviewers
  const myDecisions = decisions.filter(d => d.reviewer_user_id === user.id);
  const otherDecisionCounts: Record<string, number> = {};
  for (const d of decisions) {
    if (d.reviewer_user_id !== user.id) {
      otherDecisionCounts[d.reviewer_user_id] = (otherDecisionCounts[d.reviewer_user_id] || 0) + 1;
    }
  }
  const otherReviewerCount = Object.keys(otherDecisionCounts).length;
  const otherMaxProgress = otherReviewerCount > 0 ? Math.max(...Object.values(otherDecisionCounts)) : 0;
  const totalPapers = papers.length;
  const adjudicatedCount = adjudications.length;

  // Conflicts visible to me (only adjudicator / lead see real conflicts; for
  // reviewers we show "n papers where others have decided" instead).
  const canSeeConflicts = s.currentProjectRole === "lead" || s.currentProjectRole === "adjudicator";

  return (
    <Card className="p-3 border-primary/30 bg-primary/5">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <Users className="size-4 text-primary shrink-0" />
          <span className="font-medium text-sm truncate">{s.currentProjectName}</span>
          <Badge variant="outline" className="text-[10px]">{s.currentProjectMode}</Badge>
          <Badge variant="outline" className="text-[10px]">role: {s.currentProjectRole}</Badge>
          {blinded && (
            <Badge variant="outline" className="text-[10px] bg-amber-50 text-amber-700 border-amber-200">
              <EyeOff className="size-3 mr-0.5 inline" />Blinded
            </Badge>
          )}
          {!blinded && s.currentProjectMode !== "single" && (
            <Badge variant="outline" className="text-[10px] bg-emerald-50 text-emerald-700 border-emerald-200">
              <Eye className="size-3 mr-0.5 inline" />Unblinded
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0">
          <span><strong className="text-foreground">{myDecisions.length}</strong> / {totalPapers} my decisions</span>
          {otherReviewerCount > 0 && (
            <span>{otherReviewerCount} other reviewer{otherReviewerCount === 1 ? "" : "s"}, max <strong className="text-foreground">{otherMaxProgress}</strong> / {totalPapers}</span>
          )}
          {canSeeConflicts && adjudicatedCount > 0 && (
            <span><ShieldCheck className="size-3 inline mr-0.5" /><strong className="text-foreground">{adjudicatedCount}</strong> adjudicated</span>
          )}
          <Button size="sm" variant="ghost" onClick={reload} disabled={loading} className="h-7">
            <RefreshCcw className={`size-3 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>
      {blinded && (
        <Alert className="mt-3 border-amber-200 bg-amber-50">
          <AlertDescription className="text-xs">
            You are in <strong>blinded mode</strong>: other reviewers' decisions and reasoning are hidden until you decide on each paper. Once both reviewers have decided, conflicts are routed to the adjudicator.
          </AlertDescription>
        </Alert>
      )}
    </Card>
  );
}

/** Imperative helper: write a decision through the project API.
 *  Use this from existing screening pages when a project is active. */
export async function recordProjectDecision(
  projectId: string,
  input: {
    paper_id: string;
    stage: "abstract" | "fulltext";
    decision: DecisionValue;
    reason?: string;
    per_pico_verdict?: any;
    ai_decision?: DecisionValue | null;
    is_override?: boolean;
  },
): Promise<Decision> {
  return writeDecision(projectId, input);
}

/** Returns the effective decision for a paper inside the current project's
 *  multi-reviewer state. Falls back to the AI's cached decision if provided. */
export function projectEffectiveDecision(
  paperId: string,
  decisions: (Decision | DecisionSummary)[],
  adjudications: Adjudication[],
  aiFallback?: DecisionValue,
) {
  return effectiveProjectDecision(paperId, decisions, adjudications, aiFallback);
}

/** Small inline status pill showing the effective decision source. */
export function DecisionSourcePill({ source }: { source: "adjudication" | "consensus" | "conflict" | "ai" | "none" }) {
  const map: Record<string, { label: string; cls: string; icon?: any }> = {
    adjudication: { label: "Adjudicated", cls: "bg-violet-50 text-violet-700 border-violet-200", icon: ShieldCheck },
    consensus:    { label: "Consensus",   cls: "bg-emerald-50 text-emerald-700 border-emerald-200", icon: CheckCircle2 },
    conflict:     { label: "Conflict",    cls: "bg-rose-50 text-rose-700 border-rose-200" },
    ai:           { label: "AI",          cls: "bg-slate-50 text-slate-700 border-slate-200" },
    none:         { label: "Pending",     cls: "bg-slate-50 text-slate-500 border-slate-200" },
  };
  const meta = map[source];
  const Icon = meta.icon;
  return (
    <Badge variant="outline" className={`text-[10px] ${meta.cls}`}>
      {Icon && <Icon className="size-3 inline mr-0.5" />}
      {meta.label}
    </Badge>
  );
}
