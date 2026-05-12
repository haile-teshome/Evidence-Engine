import { useState } from "react";
import { useStore } from "../lib/store";
import { AIService, ScreenResult } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Slider } from "../components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Network, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

type SnowballType = "Both" | "Backward (References)" | "Forward (Cited by)";

export function SnowballPage() {
  const s = useStore();
  const [type, setType] = useState<SnowballType>("Both");
  const [maxCit, setMaxCit] = useState(20);
  const fetchTask = s.tasks["snowball"];
  const screenTask = s.tasks["snowball-screen"];
  const running = fetchTask?.status === "running";
  const screening = screenTask?.status === "running";

  if (!s.fullTextResults) {
    return <Alert><AlertDescription>Complete Full-Text Evidence screening first to unlock Citation Snowballing.</AlertDescription></Alert>;
  }
  const seeds = s.fullTextResults.filter(r => r.Decision === "Include");
  if (seeds.length === 0) return <Alert><AlertDescription>No papers passed full-text screening. Cannot perform snowballing.</AlertDescription></Alert>;

  async function start() {
    const { abort } = s.startTask("snowball", [{ id: "snow", label: "Fetching citations", status: "running" }]);
    s.updateTask("snowball", { progress: { done: 0, total: seeds.length } });
    const signal = abort.signal;
    try {
      const all: any[] = [];
      for (let i = 0; i < seeds.length; i++) {
        if (signal.aborted) break;
        const seed = seeds[i];
        s.updateTask("snowball", {
          progress: { done: i, total: seeds.length, label: seed.Title.slice(0, 80) },
          detail: seed.Title.slice(0, 80),
        });
        try {
          const cits = await AIService.fetchCitations(seed.Title, type, maxCit, s.sources, signal);
          cits.forEach(c => { c.seed_paper_title = seed.Title; });
          all.push(...cits);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`snowball ${i + 1} failed:`, e?.message);
        }
        s.updateTask("snowball", { progress: { done: i + 1, total: seeds.length } });
      }
      const seenT = new Set<string>();
      const unique = all.filter(p => seenT.has((p.title || "").toLowerCase()) ? false : (seenT.add((p.title || "").toLowerCase()), true));
      s.setSnowballResults(unique);
      if (signal.aborted) {
        s.updateTask("snowball", { status: "canceled" });
        toast.info(`Canceled — ${unique.length} unique citations gathered`);
      } else {
        s.updateTask("snowball", { status: "done" });
        toast.success(`Found ${unique.length} unique papers via snowballing`);
      }
    } catch (e: any) {
      s.updateTask("snowball", { status: "error", detail: e?.message });
    }
  }

  async function screen() {
    if (!s.snowballResults) return;
    const { abort } = s.startTask("snowball-screen", [{ id: "scr", label: "Screening snowballed", status: "running" }]);
    s.updateTask("snowball-screen", { progress: { done: 0, total: s.snowballResults.length } });
    const signal = abort.signal;
    try {
      const out: ScreenResult[] = [];
      for (let i = 0; i < s.snowballResults.length; i++) {
        if (signal.aborted) break;
        const p = s.snowballResults[i];
        s.updateTask("snowball-screen", {
          progress: { done: i, total: s.snowballResults.length, label: (p.title || "").slice(0, 80) },
          detail: (p.title || "").slice(0, 80),
        });
        try {
          const r = await AIService.screenPaperMultiAgent(
            { id: p.id, source: p.source, title: p.title, abstract: p.abstract, url: p.url },
            s.pico, s.inclusion, s.exclusion, signal,
          );
          out.push(r);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`snowball-screen ${i + 1} failed:`, e?.message);
        }
        s.updateTask("snowball-screen", { progress: { done: i + 1, total: s.snowballResults.length } });
      }
      s.setSnowballScreened(out);
      if (signal.aborted) {
        s.updateTask("snowball-screen", { status: "canceled" });
        toast.info(`Canceled — ${out.length} of ${s.snowballResults.length} screened`);
      } else {
        s.updateTask("snowball-screen", { status: "done" });
        toast.success(`Screened ${out.length} snowballed papers`);
      }
    } catch (e: any) {
      s.updateTask("snowball-screen", { status: "error", detail: e?.message });
    }
  }

  function addToMain() {
    if (!s.snowballScreened || !s.results) return;
    const passed = s.snowballScreened.filter(r => r.Decision === "INCLUDE");
    s.setResults([...s.results, ...passed]);
    s.setPrisma(p => ({ ...p, identified: p.identified + (s.snowballResults?.length || 0), included_final: p.included_final + passed.length }));
    toast.success(`Added ${passed.length} papers to main results`);
  }
  function clearAll() {
    s.setSnowballResults(null); s.setSnowballScreened(null);
  }

  return (
    <div className="space-y-4">
      <Alert><AlertDescription><strong>Seed Papers:</strong> {seeds.length} papers passed full-text screening.</AlertDescription></Alert>

      <Card className="p-4 grid grid-cols-3 gap-4 items-end">
        <div>
          <label className="text-sm text-muted-foreground">Snowball Type</label>
          <Select value={type} onValueChange={v => setType(v as SnowballType)}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="Both">Both</SelectItem>
              <SelectItem value="Backward (References)">Backward (References)</SelectItem>
              <SelectItem value="Forward (Cited by)">Forward (Cited by)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="text-sm text-muted-foreground">Max per paper: {maxCit}</label>
          <Slider value={[maxCit]} min={5} max={50} step={5} onValueChange={v => setMaxCit(v[0])} />
        </div>
        <Button onClick={start} disabled={running}><Network className="size-4 mr-2" />{running ? "Fetching..." : "Start Snowballing"}</Button>
      </Card>

      {fetchTask && fetchTask.status === "running" && (
        <TaskProgressCard
          task={fetchTask}
          title="Fetching citations"
          onCancel={() => s.cancelTask("snowball")}
        />
      )}
      {screenTask && screenTask.status === "running" && (
        <TaskProgressCard
          task={screenTask}
          title="Screening snowballed papers"
          onCancel={() => s.cancelTask("snowball-screen")}
        />
      )}

      {s.snowballResults && (
        <Card className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="secondary">{s.snowballResults.length} found</Badge>
            <Badge variant="outline">{s.snowballResults.filter(p => p.citation_type === "backward").length} backward</Badge>
            <Badge variant="outline">{s.snowballResults.filter(p => p.citation_type === "forward").length} forward</Badge>
          </div>
          <div className="rounded-md border max-h-[400px] overflow-auto">
            <Table>
              <TableHeader><TableRow><TableHead>Title</TableHead><TableHead>Type</TableHead><TableHead>Source</TableHead><TableHead>Seed Paper</TableHead></TableRow></TableHeader>
              <TableBody>
                {s.snowballResults.map((p, i) => (
                  <TableRow key={i}>
                    <TableCell className="max-w-md"><a href={p.url} target="_blank" rel="noreferrer" className="hover:underline">{p.title}</a></TableCell>
                    <TableCell><Badge variant="outline">{p.citation_type}</Badge></TableCell>
                    <TableCell>{p.source}</TableCell>
                    <TableCell className="text-xs text-muted-foreground max-w-xs truncate">{p.seed_paper_title}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {!s.snowballScreened ? (
            <Button onClick={screen} disabled={screening} className="w-full">{screening ? "Screening..." : "Screen Snowballed Papers"}</Button>
          ) : (
            <>
              <Alert><AlertDescription>{s.snowballScreened.filter(r => r.Decision === "INCLUDE").length} of {s.snowballScreened.length} passed screening.</AlertDescription></Alert>
              <div className="grid grid-cols-2 gap-2">
                <Button onClick={addToMain}><Plus className="size-4 mr-2" />Add to Main Results</Button>
                <Button variant="outline" onClick={clearAll}><Trash2 className="size-4 mr-2" />Clear Snowball Results</Button>
              </div>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
