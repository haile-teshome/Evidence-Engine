// Inline conflicts panel — rendered inside the active project view on the
// Projects page. Only meaningful when a project is open AND the viewer is
// the lead or adjudicator (other roles would see other reviewers' decisions,
// which would defeat blinding). When the user lacks the role we render a
// short notice instead of the list.

import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import {
  Conflict, DecisionValue, listConflicts, writeAdjudication, listProjectPapers, ProjectPaper,
} from "../lib/projects";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Alert, AlertDescription } from "./ui/alert";
import { Badge } from "./ui/badge";
import { Textarea } from "./ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";
import { ShieldCheck, AlertTriangle, RefreshCcw, ChevronDown } from "lucide-react";
import { toast } from "sonner";

export function ConflictsSection({ projectId }: { projectId: string }) {
  const s = useStore();
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [papers, setPapers] = useState<Record<string, ProjectPaper>>({});
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<"abstract" | "fulltext">("abstract");

  async function reload() {
    setLoading(true);
    try {
      const [list, allPapers] = await Promise.all([
        listConflicts(projectId, stage),
        listProjectPapers(projectId),
      ]);
      setConflicts(list);
      const byId: Record<string, ProjectPaper> = {};
      for (const p of allPapers) byId[p.paper_id] = p;
      setPapers(byId);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load conflicts");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { reload(); }, [projectId, stage]);

  // Only lead + adjudicator should see conflicts. Reviewers seeing them would
  // defeat blinding. Viewer is intentionally not gated in (read-only is fine).
  const canSee = s.currentProjectRole === "lead" || s.currentProjectRole === "adjudicator" || s.currentProjectRole === "viewer";
  if (!canSee) {
    return (
      <Card className="p-4">
        <Alert>
          <AlertTriangle className="size-4 inline mr-1" />
          <AlertDescription className="text-xs">
            Conflicts are visible to the project lead, adjudicator, and viewers — not to reviewers (it would defeat blinding). Ask the lead to grant you the adjudicator role if you need to resolve conflicts.
          </AlertDescription>
        </Alert>
      </Card>
    );
  }

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-primary" />
          <div className="font-medium">Conflicts</div>
          <Badge variant="outline" className="text-[10px] tabular-nums">{conflicts.length}</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Select value={stage} onValueChange={(v) => setStage(v as "abstract" | "fulltext")}>
            <SelectTrigger className="w-44 h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="abstract">Abstract screening</SelectItem>
              <SelectItem value="fulltext">Full-text screening</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="ghost" size="sm" onClick={reload} disabled={loading} className="h-8">
            <RefreshCcw className={`size-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      <div className="text-xs text-muted-foreground">
        Papers where two or more reviewers disagree. The adjudicator's decision becomes the effective decision for the project.
      </div>

      {!loading && conflicts.length === 0 && (
        <Alert>
          <AlertDescription className="text-xs">
            No outstanding conflicts at this stage. As reviewers complete papers, disagreements will appear here.
          </AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        {conflicts.map(c => (
          <ConflictRow
            key={c.paper_id}
            projectId={projectId}
            conflict={c}
            paper={papers[c.paper_id]}
            stage={stage}
            onResolved={reload}
            canAdjudicate={s.currentProjectRole === "lead" || s.currentProjectRole === "adjudicator"}
          />
        ))}
      </div>
    </Card>
  );
}

function ConflictRow({
  projectId, conflict, paper, stage, onResolved, canAdjudicate,
}: {
  projectId: string;
  conflict: Conflict;
  paper?: ProjectPaper;
  stage: "abstract" | "fulltext";
  onResolved: () => void;
  canAdjudicate: boolean;
}) {
  const [finalDecision, setFinalDecision] = useState<DecisionValue>("include");
  const [rationale, setRationale] = useState("");
  const [busy, setBusy] = useState(false);

  async function adjudicate() {
    if (!rationale.trim()) {
      toast.error("Provide a rationale so the audit trail records why this conflict was resolved.");
      return;
    }
    setBusy(true);
    try {
      await writeAdjudication(projectId, {
        paper_id: conflict.paper_id,
        stage,
        final_decision: finalDecision,
        rationale: rationale.trim(),
      });
      toast.success("Conflict adjudicated");
      onResolved();
    } catch (e: any) {
      toast.error(e?.message || "Adjudication failed");
    } finally {
      setBusy(false);
    }
  }

  const decisions = conflict.decisions;
  const distinct = new Set(decisions.map(d => d.decision));

  return (
    <Collapsible>
      <Card className="p-3 border-amber-300">
        <CollapsibleTrigger asChild>
          <button className="w-full flex items-start justify-between gap-3 text-left">
            <div className="min-w-0">
              <div className="font-medium text-sm truncate">{paper?.title || conflict.paper_id}</div>
              <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2 flex-wrap">
                <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200 text-[10px]">
                  {[...distinct].join(" vs. ")}
                </Badge>
                {paper?.source && <Badge variant="outline" className="text-[10px]">{paper.source}</Badge>}
                <span>{decisions.length} reviewer decisions</span>
              </div>
            </div>
            <ChevronDown className="size-4 mt-1 shrink-0" />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3 space-y-3">
          {paper?.abstract && (
            <div className="text-xs bg-muted/30 rounded p-3 max-h-40 overflow-auto leading-relaxed">
              {paper.abstract.slice(0, 1200)}{paper.abstract.length > 1200 && "…"}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {decisions.map(d => (
              <div key={d.reviewer_user_id} className="border rounded p-3 space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs font-mono truncate" title={d.reviewer_user_id}>
                    {d.reviewer_user_id.slice(0, 8)}…
                  </div>
                  <Badge variant="outline" className={
                    d.decision === "include" ? "bg-emerald-50 text-emerald-700 border-emerald-200 text-[10px]"
                    : d.decision === "exclude" ? "bg-rose-50 text-rose-700 border-rose-200 text-[10px]"
                    : "bg-amber-50 text-amber-700 border-amber-200 text-[10px]"
                  }>
                    {d.decision}
                  </Badge>
                </div>
                {("reason" in d) && d.reason && (
                  <div className="text-xs text-muted-foreground italic leading-relaxed">"{d.reason}"</div>
                )}
                {("ai_decision" in d) && d.ai_decision && (
                  <div className="text-[11px] text-muted-foreground">
                    AI suggested: <span className="font-medium">{d.ai_decision}</span>
                    {(d as any).is_override && " · reviewer override"}
                  </div>
                )}
                <div className="text-[11px] text-muted-foreground">Decided {new Date(d.decided_at).toLocaleString()}</div>
              </div>
            ))}
          </div>

          {canAdjudicate ? (
            <div className="space-y-2 pt-2 border-t">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                <ShieldCheck className="size-3" />Adjudicate
              </div>
              <div className="flex gap-2">
                <Select value={finalDecision} onValueChange={(v) => setFinalDecision(v as DecisionValue)}>
                  <SelectTrigger className="w-36 h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="include">Include</SelectItem>
                    <SelectItem value="exclude">Exclude</SelectItem>
                    <SelectItem value="maybe">Maybe (defer)</SelectItem>
                  </SelectContent>
                </Select>
                <Textarea
                  value={rationale}
                  onChange={(e) => setRationale(e.target.value)}
                  placeholder="Required: one sentence explaining the adjudication. Recorded in the audit trail."
                  rows={2}
                  className="flex-1"
                />
                <Button onClick={adjudicate} disabled={busy || !rationale.trim()} className="shrink-0">
                  {busy ? "Saving…" : "Resolve"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground italic pt-2 border-t">
              View-only: only the project lead and adjudicator can resolve conflicts.
            </div>
          )}
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
