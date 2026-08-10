import { useState } from "react";
import { useStore } from "../lib/store";
import { DataAggregator, Paper } from "../lib/mockServices";
import { refKey, paperToRef } from "../lib/references";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Checkbox } from "./ui/checkbox";
import { RefreshCw, Loader2, Plus, ExternalLink } from "lucide-react";
import { toast } from "sonner";

// Living-review updates: re-run the current search and diff it against the
// records already in the corpus, so you can find (and pull in) studies
// published since the last run. Reuses the search + screening plumbing.
export function LivingUpdates({ bare = false }: { bare?: boolean }) {
  const s = useStore();
  const Wrap: any = bare ? "div" : Card;
  const [checking, setChecking] = useState(false);
  const [fresh, setFresh] = useState<Paper[] | null>(null);
  const [checkedAt, setCheckedAt] = useState<string>("");
  // Which new records the reviewer has picked to pull in (default: all).
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const toggle = (id: string) => setSelected(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

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
      setSelected(new Set(novel.map(p => p.id)));
      setCheckedAt(new Date().toLocaleString());
      toast.success(novel.length ? `${novel.length} new record${novel.length === 1 ? "" : "s"} since your corpus` : "No new records found");
    } catch (e: any) {
      toast.error(e?.message || "Update check failed");
    } finally {
      setChecking(false);
    }
  }

  function addToCorpus() {
    const toAdd = (fresh || []).filter(p => selected.has(p.id));
    if (!toAdd.length) return;
    const merged = [...corpus, ...toAdd];
    s.setRawPapers(merged);
    s.setUniquePapers(merged);
    toast.success(`Added ${toAdd.length} record${toAdd.length === 1 ? "" : "s"}. Re-run Abstract Screening to screen them.`);
    setFresh(prev => (prev || []).filter(p => !selected.has(p.id)));
    setSelected(new Set());
  }

  const allSelected = !!fresh && fresh.length > 0 && selected.size === fresh.length;
  const toggleAll = () => setSelected(allSelected ? new Set() : new Set((fresh || []).map(p => p.id)));

  return (
    <Wrap className={bare ? "space-y-3" : "p-3 space-y-3"}>
      <div className="flex items-center justify-between gap-2">
        {bare
          ? <p className="text-xs text-muted-foreground">Re-run the search and keep only what's new since your {corpus.length.toLocaleString()} corpus record{corpus.length === 1 ? "" : "s"}.{!canCheck && " Run a search first."}</p>
          : <div className="flex items-center gap-1.5 text-sm font-medium"><RefreshCw className="size-4 text-primary" />Living-review updates</div>}
        <Button size="sm" variant="outline" className="shrink-0" onClick={check} disabled={!canCheck || checking}>
          {checking ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <RefreshCw className="size-3.5 mr-1.5" />}Check for new
        </Button>
      </div>

      {fresh && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Badge variant={fresh.length ? "default" : "secondary"}>{fresh.length} new</Badge>
            <span className="text-muted-foreground">checked {checkedAt}</span>
            {fresh.length > 0 && (
              <>
                <button onClick={toggleAll} className="text-primary hover:underline">{allSelected ? "Deselect all" : "Select all"}</button>
                <Button size="sm" className="ml-auto h-7" onClick={addToCorpus} disabled={selected.size === 0}>
                  <Plus className="size-3.5 mr-1.5" />Add {selected.size} to corpus
                </Button>
              </>
            )}
          </div>
          {fresh.length > 0 && (
            <div className="rounded-md border max-h-64 overflow-y-auto divide-y">
              {fresh.map(p => (
                <label key={p.id} className="px-3 py-2 text-xs flex items-start gap-2.5 cursor-pointer hover:bg-muted/40">
                  <Checkbox checked={selected.has(p.id)} onCheckedChange={() => toggle(p.id)} className="mt-0.5 shrink-0" />
                  <span className="flex-1 min-w-0">
                    <span className="block font-medium line-clamp-2">{p.title}</span>
                    <span className="text-muted-foreground">{p.source}{p.year ? ` · ${p.year}` : ""}</span>
                  </span>
                  {p.url && <a href={p.url} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()} className="text-primary shrink-0"><ExternalLink className="size-3.5" /></a>}
                </label>
              ))}
            </div>
          )}
        </div>
      )}
    </Wrap>
  );
}
