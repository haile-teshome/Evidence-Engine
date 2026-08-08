import { useState } from "react";
import { useStore } from "../lib/store";
import { DataAggregator, Paper } from "../lib/mockServices";
import { refKey, paperToRef } from "../lib/references";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { RefreshCw, Loader2, Plus, ExternalLink } from "lucide-react";
import { toast } from "sonner";

// Living-review updates: re-run the current search and diff it against the
// records already in the corpus, so you can find (and pull in) studies
// published since the last run. Reuses the search + screening plumbing.
export function LivingUpdates() {
  const s = useStore();
  const [checking, setChecking] = useState(false);
  const [fresh, setFresh] = useState<Paper[] | null>(null);
  const [checkedAt, setCheckedAt] = useState<string>("");

  const corpus = s.uniquePapers || s.rawPapers || [];
  const query = s.unifiedSearchQuery || s.query || "";
  const canCheck = corpus.length > 0 && (query || Object.keys(s.perDbQueries).length > 0) && s.sources.length > 0;

  async function check() {
    setChecking(true);
    setFresh(null);
    try {
      const known = new Set(corpus.map(p => refKey(paperToRef(p))));
      const { papers } = await DataAggregator.fetchForScreening(
        s.sources, s.perDbQueries, query, null, s.pico, { cap: 200 },
      );
      const seen = new Set<string>();
      const novel: Paper[] = [];
      for (const p of papers) {
        const k = refKey(paperToRef(p));
        if (!k || known.has(k) || seen.has(k)) continue;
        seen.add(k);
        novel.push(p);
      }
      setFresh(novel);
      setCheckedAt(new Date().toLocaleString());
      toast.success(novel.length ? `${novel.length} new record${novel.length === 1 ? "" : "s"} since your corpus` : "No new records found");
    } catch (e: any) {
      toast.error(e?.message || "Update check failed");
    } finally {
      setChecking(false);
    }
  }

  function addToCorpus() {
    if (!fresh?.length) return;
    const merged = [...corpus, ...fresh];
    s.setRawPapers(merged);
    s.setUniquePapers(merged);
    toast.success(`Added ${fresh.length} record${fresh.length === 1 ? "" : "s"}. Re-run Abstract Screening to screen them.`);
    setFresh([]);
  }

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-medium"><RefreshCw className="size-4 text-primary" />Living-review updates</div>
        <Button size="sm" variant="outline" onClick={check} disabled={!canCheck || checking}>
          {checking ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <RefreshCw className="size-3.5 mr-1.5" />}Check for new records
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Re-runs the current search and compares it against the {corpus.length.toLocaleString()} record{corpus.length === 1 ? "" : "s"} already in your corpus, so you can keep the review current.
        {!canCheck && " Run a search first."}
      </p>

      {fresh && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Badge variant={fresh.length ? "default" : "secondary"}>{fresh.length} new</Badge>
            <span className="text-muted-foreground">checked {checkedAt}</span>
            {fresh.length > 0 && (
              <Button size="sm" variant="outline" className="ml-auto h-7" onClick={addToCorpus}>
                <Plus className="size-3.5 mr-1.5" />Add {fresh.length} to corpus
              </Button>
            )}
          </div>
          {fresh.length > 0 && (
            <div className="rounded-md border max-h-64 overflow-y-auto divide-y">
              {fresh.map(p => (
                <div key={p.id} className="px-3 py-2 text-xs flex items-start gap-2">
                  <span className="flex-1 min-w-0">
                    <span className="block font-medium line-clamp-2">{p.title}</span>
                    <span className="text-muted-foreground">{p.source}{p.year ? ` · ${p.year}` : ""}</span>
                  </span>
                  {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="text-primary shrink-0"><ExternalLink className="size-3.5" /></a>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
