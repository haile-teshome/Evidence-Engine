import { useEffect, useMemo, useState } from "react";
import { useStore } from "../lib/store";
import { getReviewerId } from "../lib/backendClient";
import {
  ExtractionField, Extraction, ExtractionFinal, ExtractionConflict, ProjectRole, ProjectPaper,
  getExtractionTemplate, setExtractionTemplate, listExtractions, submitExtraction,
  listExtractionConflicts, reconcileExtraction, listProjectPapers,
} from "../lib/projects";
import { AIService } from "../lib/apiClient";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Loader2, Wand2, Save, Check, Download, Plus, Trash2, GitMerge, ClipboardList, Users } from "lucide-react";
import { toast } from "sonner";

// Dual independent data extraction for a project. Reviewers each extract the
// same structured template for a study; the lead/adjudicator reconciles any
// per-field disagreements into one agreed value. Mirrors the screening
// decisions/conflicts/adjudication flow, keyed by (paper, field).
export function ExtractionWorkspace({ projectId, role }: { projectId: string; role: ProjectRole | null }) {
  const s = useStore();
  const me = getReviewerId();
  const isLead = role === "lead";
  const canReconcile = role === "lead" || role === "adjudicator";

  const [tab, setTab] = useState<"extract" | "reconcile" | "template">("extract");
  const [loading, setLoading] = useState(true);
  const [fields, setFields] = useState<ExtractionField[]>([]);
  const [papers, setPapers] = useState<ProjectPaper[]>([]);
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [finals, setFinals] = useState<ExtractionFinal[]>([]);
  const [conflicts, setConflicts] = useState<ExtractionConflict[]>([]);

  const [selPaper, setSelPaper] = useState<string>("");
  const [form, setForm] = useState<Record<string, any>>({});
  const [aiBusy, setAiBusy] = useState(false);
  const [saving, setSaving] = useState(false);

  async function reload() {
    setLoading(true);
    try {
      const [tpl, ps, ex] = await Promise.all([
        getExtractionTemplate(projectId), listProjectPapers(projectId), listExtractions(projectId),
      ]);
      setFields(tpl.fields);
      setPapers(ps);
      setExtractions(ex.extractions);
      setFinals(ex.finals);
      if (canReconcile) {
        try { setConflicts(await listExtractionConflicts(projectId)); } catch { /* non-fatal */ }
      }
    } catch (e: any) {
      toast.error(e?.message || "Failed to load extraction");
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { reload(); /* eslint-disable-next-line */ }, [projectId]);

  // Load my saved values when the selected paper changes.
  useEffect(() => {
    if (!selPaper) { setForm({}); return; }
    const mine = extractions.find(e => e.paper_id === selPaper && e.reviewer_user_id === me);
    setForm(mine?.values ? { ...mine.values } : {});
  }, [selPaper, extractions, me]);

  const paperTitle = (pid: string) => papers.find(p => p.paper_id === pid)?.title || pid;
  const mySubmitted = useMemo(
    () => new Set(extractions.filter(e => e.reviewer_user_id === me && e.submitted).map(e => e.paper_id)),
    [extractions, me],
  );
  const finalByPaper = useMemo(() => new Map(finals.map(f => [f.paper_id, f])), [finals]);

  async function aiPrefill() {
    if (!selPaper) return;
    const p = papers.find(x => x.paper_id === selPaper);
    const text = s.fullTexts[selPaper]?.text || p?.abstract || "";
    if (!text) { toast.error("No text for this paper. Acquire full text first, or fill it in manually."); return; }
    setAiBusy(true);
    try {
      const vals = await AIService.extractFields(
        text, fields.map(f => ({ id: f.id, label: f.label, type: f.type, options: f.options })), p?.title || "",
      );
      // Fill only empty fields; never overwrite what the reviewer already typed.
      setForm(prev => {
        const next = { ...prev };
        for (const [k, v] of Object.entries(vals)) {
          if (v !== "" && v != null && (next[k] === undefined || next[k] === "")) next[k] = v;
        }
        return next;
      });
      toast.success("Pre-filled empty fields from the text. Confirm every value.");
    } catch (e: any) {
      toast.error(e?.message || "AI pre-fill failed");
    } finally {
      setAiBusy(false);
    }
  }

  async function save(submit: boolean) {
    if (!selPaper) return;
    setSaving(true);
    try {
      await submitExtraction(projectId, { paper_id: selPaper, values: form, submitted: submit });
      toast.success(submit ? "Extraction submitted" : "Draft saved");
      await reload();
    } catch (e: any) {
      toast.error(e?.message || "Save failed");
    } finally {
      setSaving(false);
    }
  }

  // Effective value per (paper, field): reconciled final wins, else a unanimous
  // value across submitted extractions, else blank (unresolved).
  function effective(paperId: string, fieldId: string): string {
    const fin = finalByPaper.get(paperId);
    if (fin && fin.values[fieldId] != null && fin.values[fieldId] !== "") return String(fin.values[fieldId]);
    const subs = extractions.filter(e => e.paper_id === paperId && e.submitted);
    if (!subs.length) return "";
    const vals = subs.map(e => String((e.values || {})[fieldId] ?? "").trim());
    const distinct = new Set(vals.map(v => v.toLowerCase()));
    return distinct.size === 1 ? vals[0] : "";
  }

  function exportCsv() {
    const cols = ["Paper", ...fields.map(f => f.label)];
    const rows = papers
      .filter(p => extractions.some(e => e.paper_id === p.paper_id && e.submitted) || finalByPaper.has(p.paper_id))
      .map(p => [p.title, ...fields.map(f => effective(p.paper_id, f.id))]);
    if (!rows.length) { toast.error("No submitted extractions to export yet."); return; }
    const esc = (v: string) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const csv = [cols, ...rows].map(r => r.map(esc).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url; a.download = "extraction.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  const grouped = useMemo(() => {
    const m = new Map<string, ExtractionField[]>();
    for (const f of fields) { if (!m.has(f.group)) m.set(f.group, []); m.get(f.group)!.push(f); }
    return [...m.entries()];
  }, [fields]);

  function renderInput(f: ExtractionField) {
    const v = form[f.id] ?? "";
    const set = (val: any) => setForm(prev => ({ ...prev, [f.id]: val }));
    if (f.type === "category") {
      return (
        <Select value={String(v)} onValueChange={set}>
          <SelectTrigger className="h-8"><SelectValue placeholder="Choose" /></SelectTrigger>
          <SelectContent>{f.options.map(o => <SelectItem key={o} value={o}>{o}</SelectItem>)}</SelectContent>
        </Select>
      );
    }
    if (f.type === "number") return <Input type="number" className="h-8" value={v} onChange={e => set(e.target.value)} />;
    if (f.type === "date") return <Input type="date" className="h-8" value={v} onChange={e => set(e.target.value)} />;
    return <Input className="h-8" value={v} onChange={e => set(e.target.value)} />;
  }

  if (loading) {
    return <Card className="p-8 flex items-center justify-center text-muted-foreground"><Loader2 className="size-5 animate-spin" /></Card>;
  }

  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-medium flex items-center gap-2"><Users className="size-4 text-primary" />Data extraction: dual and independent</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {mySubmitted.size} of {papers.length} papers submitted by you{finals.length ? ` · ${finals.length} reconciled` : ""}. Two reviewers extract the same fields, then disagreements are reconciled.
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={exportCsv}><Download className="size-3.5 mr-1.5" />Export CSV</Button>
      </div>

      <Tabs value={tab} onValueChange={v => setTab(v as any)}>
        <TabsList>
          <TabsTrigger value="extract"><ClipboardList className="size-3.5 mr-1.5" />Extract</TabsTrigger>
          {canReconcile && <TabsTrigger value="reconcile"><GitMerge className="size-3.5 mr-1.5" />Reconcile{conflicts.length ? <Badge variant="secondary" className="ml-1.5">{conflicts.length}</Badge> : null}</TabsTrigger>}
          {isLead && <TabsTrigger value="template">Template</TabsTrigger>}
        </TabsList>

        {/* ---- Extract ---- */}
        <TabsContent value="extract" className="pt-3">
          <div className="flex gap-4 min-h-[320px]">
            <div className="w-64 shrink-0 border rounded-md overflow-y-auto max-h-[460px]">
              {papers.length === 0 && <div className="p-3 text-xs text-muted-foreground">No papers in this project yet.</div>}
              {papers.map(p => (
                <button key={p.paper_id} onClick={() => setSelPaper(p.paper_id)}
                  className={`w-full text-left px-3 py-2 border-b text-xs flex items-start gap-2 hover:bg-muted/60 ${p.paper_id === selPaper ? "bg-primary/10" : ""}`}>
                  <span className="flex-1 line-clamp-2">{p.title}</span>
                  {mySubmitted.has(p.paper_id) && <Check className="size-3.5 text-primary shrink-0 mt-0.5" />}
                </button>
              ))}
            </div>

            <div className="flex-1 min-w-0">
              {!selPaper ? (
                <div className="h-full flex items-center justify-center text-sm text-muted-foreground">Select a paper to extract.</div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-medium line-clamp-1">{paperTitle(selPaper)}</div>
                    <Button size="sm" variant="outline" onClick={aiPrefill} disabled={aiBusy}
                      title="Draft values from the paper text. You confirm each one.">
                      {aiBusy ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Wand2 className="size-3.5 mr-1.5" />}AI pre-fill
                    </Button>
                  </div>
                  {grouped.map(([group, gfields]) => (
                    <div key={group} className="space-y-2">
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{group}</div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        {gfields.map(f => (
                          <label key={f.id} className="text-xs space-y-1">
                            <span className="text-muted-foreground">{f.label}</span>
                            {renderInput(f)}
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                  <div className="flex items-center gap-2 pt-1">
                    <Button size="sm" variant="outline" onClick={() => save(false)} disabled={saving}><Save className="size-3.5 mr-1.5" />Save draft</Button>
                    <Button size="sm" onClick={() => save(true)} disabled={saving}>{saving ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Check className="size-3.5 mr-1.5" />}Submit</Button>
                    {mySubmitted.has(selPaper) && <span className="text-xs text-muted-foreground">You have submitted this paper.</span>}
                  </div>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        {/* ---- Reconcile ---- */}
        {canReconcile && (
          <TabsContent value="reconcile" className="pt-3">
            {conflicts.length === 0 ? (
              <div className="text-sm text-muted-foreground py-6 text-center">No field-level disagreements to reconcile. Papers agree, or fewer than two reviewers have submitted.</div>
            ) : (
              <ReconcilePanel projectId={projectId} conflicts={conflicts} fields={fields} paperTitle={paperTitle} onDone={reload} />
            )}
          </TabsContent>
        )}

        {/* ---- Template ---- */}
        {isLead && (
          <TabsContent value="template" className="pt-3">
            <TemplateEditor projectId={projectId} fields={fields} onSaved={setFields} />
          </TabsContent>
        )}
      </Tabs>
    </Card>
  );
}

// ---- Reconciliation panel -------------------------------------------------
function ReconcilePanel({
  projectId, conflicts, fields, paperTitle, onDone,
}: {
  projectId: string; conflicts: ExtractionConflict[]; fields: ExtractionField[];
  paperTitle: (id: string) => string; onDone: () => void;
}) {
  const [selPaper, setSelPaper] = useState<string>(conflicts[0]?.paper_id || "");
  const [chosen, setChosen] = useState<Record<string, any>>({});
  const [rationale, setRationale] = useState("");
  const [saving, setSaving] = useState(false);
  const label = (fid: string) => fields.find(f => f.id === fid)?.label || fid;

  useEffect(() => { setChosen({}); setRationale(""); }, [selPaper]);

  const conflict = conflicts.find(c => c.paper_id === selPaper);

  async function save() {
    if (!conflict) return;
    const missing = conflict.fields.filter(fid => chosen[fid] === undefined);
    if (missing.length) { toast.error(`Choose a value for: ${missing.map(label).join(", ")}`); return; }
    setSaving(true);
    try {
      await reconcileExtraction(projectId, { paper_id: selPaper, values: chosen, rationale });
      toast.success("Reconciled");
      onDone();
    } catch (e: any) {
      toast.error(e?.message || "Reconcile failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex gap-4">
      <div className="w-56 shrink-0 border rounded-md overflow-y-auto max-h-[440px]">
        {conflicts.map(c => (
          <button key={c.paper_id} onClick={() => setSelPaper(c.paper_id)}
            className={`w-full text-left px-3 py-2 border-b text-xs hover:bg-muted/60 ${c.paper_id === selPaper ? "bg-primary/10" : ""}`}>
            <span className="line-clamp-2">{paperTitle(c.paper_id)}</span>
            <span className="text-[10px] text-muted-foreground">{c.fields.length} field{c.fields.length === 1 ? "" : "s"} in conflict</span>
          </button>
        ))}
      </div>
      <div className="flex-1 min-w-0 space-y-4">
        {!conflict ? (
          <div className="text-sm text-muted-foreground">Select a paper.</div>
        ) : (
          <>
            <div className="text-sm font-medium line-clamp-1">{paperTitle(selPaper)}</div>
            {conflict.fields.map(fid => {
              // distinct submitted values for this field, with who gave each
              const options = conflict.extractions
                .map(e => ({ who: e.reviewer_user_id, val: String((e.values || {})[fid] ?? "").trim() }))
                .filter(o => o.val !== "");
              const seen = new Set<string>();
              const distinct = options.filter(o => (seen.has(o.val.toLowerCase()) ? false : seen.add(o.val.toLowerCase())));
              return (
                <div key={fid} className="space-y-1.5 border-b pb-3">
                  <div className="text-xs font-medium">{label(fid)}</div>
                  <div className="flex flex-wrap gap-2">
                    {distinct.map(o => (
                      <button key={o.val} onClick={() => setChosen(prev => ({ ...prev, [fid]: o.val }))}
                        className={`text-xs rounded-md border px-2.5 py-1 ${String(chosen[fid]) === o.val ? "bg-primary text-primary-foreground border-primary" : "hover:bg-muted"}`}>
                        {o.val} <span className="opacity-60">· {o.who}</span>
                      </button>
                    ))}
                  </div>
                  <Input className="h-8 text-xs" placeholder="or type the agreed value"
                    value={chosen[fid] ?? ""} onChange={e => setChosen(prev => ({ ...prev, [fid]: e.target.value }))} />
                </div>
              );
            })}
            <Textarea placeholder="Reconciliation note (optional)" value={rationale} onChange={e => setRationale(e.target.value)} className="text-xs min-h-[60px]" />
            <Button size="sm" onClick={save} disabled={saving}>{saving ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <GitMerge className="size-3.5 mr-1.5" />}Save reconciled values</Button>
          </>
        )}
      </div>
    </div>
  );
}

// ---- Template editor (lead only) -----------------------------------------
function TemplateEditor({
  projectId, fields, onSaved,
}: {
  projectId: string; fields: ExtractionField[]; onSaved: (f: ExtractionField[]) => void;
}) {
  const [draft, setDraft] = useState<ExtractionField[]>(fields);
  const [saving, setSaving] = useState(false);

  function patch(i: number, p: Partial<ExtractionField>) {
    setDraft(d => d.map((f, idx) => idx === i ? { ...f, ...p } : f));
  }
  function add() {
    setDraft(d => [...d, { id: "", label: "New field", group: "General", type: "text", options: [] }]);
  }
  function remove(i: number) { setDraft(d => d.filter((_, idx) => idx !== i)); }

  async function save() {
    setSaving(true);
    try {
      const saved = await setExtractionTemplate(projectId, draft);
      onSaved(saved);
      setDraft(saved);
      toast.success("Template saved");
    } catch (e: any) {
      toast.error(e?.message || "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">Define the fields both reviewers extract. Changes apply to new and existing extractions.</p>
      <div className="space-y-1.5">
        {draft.map((f, i) => (
          <div key={i} className="flex items-center gap-2">
            <Input className="h-8 flex-1" value={f.label} onChange={e => patch(i, { label: e.target.value })} placeholder="Field label" />
            <Input className="h-8 w-32" value={f.group} onChange={e => patch(i, { group: e.target.value })} placeholder="Group" />
            <Select value={f.type} onValueChange={v => patch(i, { type: v as any })}>
              <SelectTrigger className="h-8 w-28"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="number">Number</SelectItem>
                <SelectItem value="category">Category</SelectItem>
                <SelectItem value="date">Date</SelectItem>
              </SelectContent>
            </Select>
            <Input className="h-8 w-48" value={(f.options || []).join(", ")}
              onChange={e => patch(i, { options: e.target.value.split(",").map(s => s.trim()).filter(Boolean) })}
              placeholder={f.type === "category" ? "option, option" : ""} disabled={f.type !== "category"} />
            <button onClick={() => remove(i)} className="text-muted-foreground hover:text-destructive"><Trash2 className="size-3.5" /></button>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 pt-1">
        <Button size="sm" variant="outline" onClick={add}><Plus className="size-3.5 mr-1.5" />Add field</Button>
        <Button size="sm" onClick={save} disabled={saving}>{saving ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Save className="size-3.5 mr-1.5" />}Save template</Button>
      </div>
    </div>
  );
}
