import { useMemo, useState } from "react";
import { useStore } from "../lib/store";
import { AIService } from "../lib/apiClient";
import { effectiveAbstractDecision, effectiveFullTextDecision } from "../lib/exclusionBucketing";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ShieldAlert, Loader2, CheckCircle2, AlertTriangle, HelpCircle, ExternalLink } from "lucide-react";
import { toast } from "sonner";

type Result = { paper_id: string; status: string; detail: string; source: string; doi: string };

// Research-integrity check: flags included studies that have been retracted or
// carry an expression of concern (OpenAlex is_retracted + Crossref updates).
// Including a retracted study is a serious, avoidable error; this catches it.
export function IntegrityCheck() {
  const s = useStore();
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Result[] | null>(null);

  // The included set: full-text-included if screened, else abstract-included,
  // else the whole corpus.
  const items = useMemo(() => {
    if (s.fullTextResults?.length) {
      return s.fullTextResults
        .filter(r => effectiveFullTextDecision(r, s.fullTextOverrides) === "Include")
        .map(r => ({ paper_id: r.paper_id, doi: r.URL, title: r.Title }));
    }
    if (s.results?.length) {
      return s.results
        .filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE")
        .map(r => ({ paper_id: r.paper_id, doi: r.URL, title: r.Title }));
    }
    return (s.uniquePapers || []).map(p => ({ paper_id: p.id, doi: p.url, title: p.title }));
  }, [s.fullTextResults, s.results, s.uniquePapers, s.fullTextOverrides, s.abstractOverrides]);

  const titleOf = (id: string) => items.find(i => i.paper_id === id)?.title || id;

  async function run() {
    if (!items.length) { toast.error("No included studies to check yet."); return; }
    setBusy(true);
    try {
      setResults(await AIService.checkIntegrity(items));
    } catch (e: any) {
      toast.error(e?.message || "Integrity check failed");
    } finally {
      setBusy(false);
    }
  }

  const flagged = (results || []).filter(r => r.status === "retracted" || r.status === "concern" || r.status === "correction");
  const unknown = (results || []).filter(r => r.status === "unknown").length;
  const clean = (results || []).filter(r => r.status === "ok").length;

  const badge = (st: string) =>
    st === "retracted" ? <Badge variant="destructive">Retracted</Badge>
    : st === "concern" ? <Badge className="bg-amber-500">Expression of concern</Badge>
    : st === "correction" ? <Badge variant="secondary">Correction / erratum</Badge>
    : null;

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          <ShieldAlert className="size-4 text-primary mt-0.5 shrink-0" />
          <div className="min-w-0">
            <div className="text-sm font-medium leading-tight">Research integrity check</div>
            <p className="text-xs text-muted-foreground leading-snug">Flags included studies that are retracted or under an expression of concern (OpenAlex + Crossref).</p>
          </div>
        </div>
        <Button size="sm" variant="outline" className="shrink-0" onClick={run} disabled={busy || !items.length}>
          {busy ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <ShieldAlert className="size-3.5 mr-1.5" />}Check {items.length} included
        </Button>
      </div>

      {results && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {flagged.length > 0
              ? <Badge variant="destructive">{flagged.length} flagged</Badge>
              : <span className="inline-flex items-center gap-1 text-emerald-600"><CheckCircle2 className="size-3.5" />No retractions found</span>}
            <span className="inline-flex items-center gap-1 text-muted-foreground"><CheckCircle2 className="size-3.5" />{clean} clear</span>
            {unknown > 0 && <span className="inline-flex items-center gap-1 text-muted-foreground"><HelpCircle className="size-3.5" />{unknown} not checkable (no DOI)</span>}
          </div>
          {flagged.length > 0 && (
            <div className="rounded-md border divide-y">
              {flagged.map(r => (
                <div key={r.paper_id} className="px-3 py-2 text-xs flex items-start gap-2">
                  <AlertTriangle className="size-3.5 mt-0.5 shrink-0 text-amber-500" />
                  <span className="flex-1 min-w-0">
                    <span className="block font-medium line-clamp-2">{titleOf(r.paper_id)}</span>
                    <span className="text-muted-foreground">{r.detail}{r.source ? ` · ${r.source}` : ""}</span>
                  </span>
                  <span className="shrink-0 flex items-center gap-2">
                    {badge(r.status)}
                    {r.doi && <a href={`https://doi.org/${r.doi}`} target="_blank" rel="noreferrer" className="text-primary"><ExternalLink className="size-3.5" /></a>}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
