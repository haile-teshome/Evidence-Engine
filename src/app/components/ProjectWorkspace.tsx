import { useEffect, useRef, useState } from "react";
import {
  Participant, ProjectRole, AutoAssignResult,
  listParticipants, addParticipant, updateParticipant, removeParticipant,
  autoAssign, listProjectPapers,
  exportProjectBundle, importIntoProject, reproReport,
} from "../lib/projects";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { Skeleton } from "./ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Users, Plus, Trash2, Shuffle, X, Loader2, UserPlus, Download, Upload, FileText, Share2 } from "lucide-react";
import { toast } from "sonner";

function downloadFile(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

type Strategy = "dual" | "overlap" | "weighted" | "manual";

// Author's cockpit for a project: define reviewer slots and auto-assign the
// papers across them with granular control. The author's copy stays the master.
export function ProjectWorkspace({ projectId, projectName, onClose }: { projectId: string; projectName: string; onClose: () => void }) {
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [paperCount, setPaperCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState<ProjectRole>("reviewer");

  const [strategy, setStrategy] = useState<Strategy>("dual");
  const [overlapPct, setOverlapPct] = useState(20);
  const [perPaper, setPerPaper] = useState(2);
  const [includeCalib, setIncludeCalib] = useState(true);
  const [assigning, setAssigning] = useState(false);
  const [result, setResult] = useState<AutoAssignResult | null>(null);
  const [shareBusy, setShareBusy] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  const safeName = projectName.replace(/[^a-z0-9]+/gi, "-").toLowerCase().replace(/^-+|-+$/g, "") || "project";

  async function exportBundle() {
    setShareBusy(true);
    try {
      const b = await exportProjectBundle(projectId);
      downloadFile(`${safeName}-bundle.json`, JSON.stringify(b, null, 2), "application/json");
      toast.success("Project bundle exported");
    } catch (e: any) { toast.error(e?.message || "Export failed"); }
    finally { setShareBusy(false); }
  }
  async function exportReport() {
    setShareBusy(true);
    try {
      const b = await exportProjectBundle(projectId);
      downloadFile(`${safeName}-reproducibility.md`, reproReport(b), "text/markdown");
      toast.success("Reproducibility report exported");
    } catch (e: any) { toast.error(e?.message || "Export failed"); }
    finally { setShareBusy(false); }
  }
  async function onImport(file: File) {
    setShareBusy(true);
    try {
      const bundle = JSON.parse(await file.text());
      const { merged } = await importIntoProject(projectId, bundle);
      const total = Object.values(merged).reduce((a, b) => a + b, 0);
      toast.success(`Merged ${total} record(s): ${Object.entries(merged).filter(([, n]) => n).map(([k, n]) => `${n} ${k}`).join(", ") || "nothing new"}`);
    } catch (e: any) { toast.error(e?.message || "Import failed. Is this an Evidence Engine bundle?"); }
    finally { setShareBusy(false); if (importRef.current) importRef.current.value = ""; }
  }

  async function reload() {
    setLoading(true);
    try {
      const [ps, papers] = await Promise.all([listParticipants(projectId), listProjectPapers(projectId)]);
      setParticipants(ps);
      setPaperCount(papers.length);
    } catch (e: any) { toast.error(e?.message || "Failed to load workspace"); }
    finally { setLoading(false); }
  }
  useEffect(() => { reload(); /* eslint-disable-next-line */ }, [projectId]);

  const reviewers = participants.filter(p => p.role === "reviewer" || p.role === "lead");

  async function add() {
    const name = newName.trim();
    if (!name) return;
    try {
      const p = await addParticipant(projectId, name, newRole, 1);
      setParticipants(prev => [...prev, p]);
      setNewName("");
    } catch (e: any) { toast.error(e?.message || "Could not add reviewer"); }
  }
  async function patch(id: string, patchObj: { name?: string; role?: ProjectRole; weight?: number }) {
    setParticipants(prev => prev.map(p => p.id === id ? { ...p, ...patchObj } : p));
    try { await updateParticipant(projectId, id, patchObj); } catch { reload(); }
  }
  async function remove(id: string) {
    setParticipants(prev => prev.filter(p => p.id !== id));
    try { await removeParticipant(projectId, id); } catch { reload(); }
  }

  async function runAssign() {
    if (!reviewers.length) { toast.error("Add at least one reviewer first."); return; }
    setAssigning(true);
    try {
      const r = await autoAssign(projectId, {
        strategy,
        overlap_pct: overlapPct,
        reviewers_per_paper: perPaper,
        include_calibration: includeCalib,
      });
      setResult(r);
      toast.success(`Assigned ${r.assigned} paper-slots across ${r.per_reviewer.length} reviewers`);
    } catch (e: any) { toast.error(e?.message || "Assignment failed"); }
    finally { setAssigning(false); }
  }

  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-medium flex items-center gap-2"><Users className="size-4 text-primary" />Team &amp; assignment: {projectName}</h3>
          <p className="text-xs text-muted-foreground mt-0.5">{paperCount.toLocaleString()} papers · {reviewers.length} reviewer{reviewers.length === 1 ? "" : "s"}. Define reviewers and split the papers across them.</p>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}><X className="size-4" /></Button>
      </div>

      {loading ? (
        <div className="space-y-3 py-1">
          <Skeleton className="h-3 w-24" />
          {[0, 1, 2].map(i => (
            <div key={i} className="flex items-center gap-2">
              <Skeleton className="h-8 flex-1" />
              <Skeleton className="h-8 w-32" />
              <Skeleton className="h-8 w-16" />
            </div>
          ))}
          <Skeleton className="h-8 w-40" />
        </div>
      ) : (
        <>
          {/* Reviewers */}
          <div className="space-y-2">
            <Label className="text-xs">Reviewers</Label>
            <div className="space-y-1.5">
              {participants.map(p => (
                <div key={p.id} className="flex items-center gap-2 text-sm">
                  <Input value={p.name} onChange={e => patch(p.id, { name: e.target.value })} className="h-8 flex-1" />
                  <Select value={p.role} onValueChange={v => patch(p.id, { role: v as ProjectRole })}>
                    <SelectTrigger className="h-8 w-32"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="reviewer">Reviewer</SelectItem>
                      <SelectItem value="adjudicator">Adjudicator</SelectItem>
                      <SelectItem value="viewer">Viewer</SelectItem>
                      <SelectItem value="lead">Lead</SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="flex items-center gap-1" title="Relative share of papers (weighted assignment)">
                    <span className="text-[10px] text-muted-foreground">wt</span>
                    <Input type="number" min={0} step={0.5} value={p.weight}
                      onChange={e => patch(p.id, { weight: Math.max(0, parseFloat(e.target.value) || 0) })}
                      className="h-8 w-16" />
                  </div>
                  <button onClick={() => remove(p.id)} className="text-muted-foreground hover:text-destructive"><Trash2 className="size-3.5" /></button>
                </div>
              ))}
              {participants.length === 0 && <div className="text-xs text-muted-foreground">No reviewers yet. Add people below.</div>}
            </div>
            <div className="flex items-center gap-2 pt-1">
              <Input value={newName} onChange={e => setNewName(e.target.value)} onKeyDown={e => { if (e.key === "Enter") add(); }} placeholder="Reviewer name" className="h-8 flex-1" />
              <Select value={newRole} onValueChange={v => setNewRole(v as ProjectRole)}>
                <SelectTrigger className="h-8 w-32"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="reviewer">Reviewer</SelectItem>
                  <SelectItem value="adjudicator">Adjudicator</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
              <Button size="sm" onClick={add} disabled={!newName.trim()}><UserPlus className="size-3.5 mr-1.5" />Add</Button>
            </div>
          </div>

          {/* Assignment */}
          <div className="space-y-3 border-t pt-3">
            <Label className="text-xs">Auto-assign papers</Label>
            <div className="flex flex-wrap items-center gap-3">
              <Select value={strategy} onValueChange={v => setStrategy(v as Strategy)}>
                <SelectTrigger className="h-8 w-52"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="dual">Dual: every paper by N reviewers</SelectItem>
                  <SelectItem value="overlap">Overlap %: split, with QC double-check</SelectItem>
                  <SelectItem value="weighted">Weighted split: by reviewer weight</SelectItem>
                </SelectContent>
              </Select>
              {strategy === "dual" && (
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground">Reviewers / paper
                  <Input type="number" min={1} max={Math.max(1, reviewers.length)} value={perPaper} onChange={e => setPerPaper(Math.max(1, parseInt(e.target.value) || 1))} className="h-8 w-16" />
                </label>
              )}
              {strategy === "overlap" && (
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground">Double-screened
                  <Input type="number" min={0} max={100} value={overlapPct} onChange={e => setOverlapPct(Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))} className="h-8 w-16" />%
                </label>
              )}
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <input type="checkbox" checked={includeCalib} onChange={e => setIncludeCalib(e.target.checked)} />
                Give everyone the calibration set
              </label>
              <Button size="sm" onClick={runAssign} disabled={assigning || !reviewers.length || paperCount === 0}>
                {assigning ? <Loader2 className="size-4 mr-1.5 animate-spin" /> : <Shuffle className="size-3.5 mr-1.5" />}Auto-assign
              </Button>
            </div>

            {result && (
              <div className="rounded-md border bg-muted/20 p-3 text-sm space-y-1.5">
                <div className="text-xs text-muted-foreground">{result.assigned} paper-slots assigned{result.calibration ? ` · ${result.calibration} calibration paper(s) to all` : ""}</div>
                <div className="flex flex-wrap gap-2">
                  {result.per_reviewer.map(r => (
                    <Badge key={r.id} variant="secondary">{r.name}: {r.count}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Share & reproducibility */}
          <div className="space-y-2 border-t pt-3">
            <Label className="text-xs flex items-center gap-1.5"><Share2 className="size-3.5" />Share &amp; reproducibility</Label>
            <p className="text-xs text-muted-foreground">
              Export a bundle to hand this project to a reviewer, then import their returned bundle to merge their decisions and extractions. The reproducibility report is a complete, timestamped record of the review.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="outline" onClick={exportBundle} disabled={shareBusy}>
                {shareBusy ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Download className="size-3.5 mr-1.5" />}Export bundle (.json)
              </Button>
              <Button size="sm" variant="outline" onClick={exportReport} disabled={shareBusy}>
                <FileText className="size-3.5 mr-1.5" />Reproducibility report (.md)
              </Button>
              <Button size="sm" variant="outline" onClick={() => importRef.current?.click()} disabled={shareBusy}>
                <Upload className="size-3.5 mr-1.5" />Import returned bundle
              </Button>
              <input ref={importRef} type="file" accept=".json,application/json" className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) onImport(f); }} />
            </div>
          </div>
        </>
      )}
    </Card>
  );
}
