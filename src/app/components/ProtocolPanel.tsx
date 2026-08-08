import { useState } from "react";
import { useStore } from "../lib/store";
import { AIService, ProtocolDeviation } from "../lib/apiClient";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { FileText, Loader2, Download, Plus, Trash2, Wand2, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";

function dl(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// Register-first protocol: draft a PROSPERO-style protocol from the PICO and
// eligibility criteria, then track any deviations from it. Both are exportable
// for transparent reporting.
export function ProtocolPanel() {
  const s = useStore();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [title, setTitle] = useState("");
  const protocol = s.protocol;
  const deviations = s.protocolDeviations;

  async function generate() {
    setBusy(true);
    try {
      const text = await AIService.generateProtocol({
        title: title || s.query || "",
        pico: s.pico, inclusion: s.inclusion, exclusion: s.exclusion, sources: s.sources,
      });
      if (!text) { toast.error("Could not generate a protocol. Is a model selected?"); return; }
      s.setProtocol({ text, generatedAt: new Date().toISOString(), registered: protocol?.registered, registrationId: protocol?.registrationId });
      toast.success("Protocol drafted. Review and edit before registering.");
    } catch (e: any) {
      toast.error(e?.message || "Protocol generation failed");
    } finally { setBusy(false); }
  }

  function addDeviation() {
    s.setProtocolDeviations(prev => [...prev, {
      id: Date.now().toString(36), date: new Date().toISOString().slice(0, 10),
      section: "", change: "", reason: "",
    }]);
  }
  function patchDeviation(id: string, p: Partial<ProtocolDeviation>) {
    s.setProtocolDeviations(prev => prev.map(d => d.id === id ? { ...d, ...p } : d));
  }

  function exportMd() {
    if (!protocol?.text) { toast.error("Generate the protocol first."); return; }
    let md = protocol.text;
    if (protocol.registrationId) md = `Registration: ${protocol.registrationId}\n\n${md}`;
    if (deviations.length) {
      md += "\n\n## Deviations from the protocol\n\n";
      md += "| Date | Section | Change | Reason |\n|---|---|---|---|\n";
      for (const d of deviations) md += `| ${d.date} | ${d.section} | ${d.change} | ${d.reason} |\n`;
    }
    dl("protocol.md", md, "text/markdown");
    toast.success("Protocol exported");
  }

  return (
    <Card className="p-3 space-y-2">
      <button className="flex items-center gap-1.5 text-sm font-medium w-full text-left" onClick={() => setOpen(o => !o)}>
        {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
        <FileText className="size-4 text-primary" />Protocol (register first)
        {protocol && <span className="text-xs text-muted-foreground font-normal ml-1">drafted{deviations.length ? `, ${deviations.length} deviation${deviations.length === 1 ? "" : "s"}` : ""}</span>}
      </button>

      {open && (
        <div className="space-y-3 pt-1">
          <p className="text-xs text-muted-foreground">
            Draft a PROSPERO-style protocol from your question and criteria, register it before screening, then log any deviations for transparent reporting.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Input value={title} onChange={e => setTitle(e.target.value)} placeholder="Review title (optional)" className="h-8 text-sm flex-1 min-w-[200px]" />
            <Button size="sm" onClick={generate} disabled={busy}>
              {busy ? <Loader2 className="size-3.5 mr-1.5 animate-spin" /> : <Wand2 className="size-3.5 mr-1.5" />}{protocol ? "Regenerate" : "Draft protocol"}
            </Button>
            <Button size="sm" variant="outline" onClick={exportMd} disabled={!protocol?.text}><Download className="size-3.5 mr-1.5" />Export (.md)</Button>
          </div>

          {protocol && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <input type="checkbox" checked={!!protocol.registered} onChange={e => s.setProtocol({ ...protocol, registered: e.target.checked })} />
                  Registered
                </label>
                <Input value={protocol.registrationId || ""} onChange={e => s.setProtocol({ ...protocol, registrationId: e.target.value })}
                  placeholder="Registration ID (e.g. PROSPERO CRD…)" className="h-7 text-xs w-64" />
                <span className="text-[11px] text-muted-foreground">Drafted {new Date(protocol.generatedAt).toLocaleDateString()}</span>
              </div>
              <Textarea value={protocol.text} onChange={e => s.setProtocol({ ...protocol, text: e.target.value })}
                className="text-xs font-mono min-h-[280px]" />

              <div className="space-y-1.5 border-t pt-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-medium">Deviations from the protocol</div>
                  <Button size="sm" variant="outline" onClick={addDeviation}><Plus className="size-3.5 mr-1.5" />Add deviation</Button>
                </div>
                {deviations.length === 0 && <div className="text-[11px] text-muted-foreground">None logged. Record any change made after registration.</div>}
                {deviations.map(d => (
                  <div key={d.id} className="grid grid-cols-[110px_1fr_1fr_1fr_auto] gap-1.5 items-center">
                    <Input type="date" value={d.date} onChange={e => patchDeviation(d.id, { date: e.target.value })} className="h-7 text-xs" />
                    <Input value={d.section} onChange={e => patchDeviation(d.id, { section: e.target.value })} placeholder="Section" className="h-7 text-xs" />
                    <Input value={d.change} onChange={e => patchDeviation(d.id, { change: e.target.value })} placeholder="What changed" className="h-7 text-xs" />
                    <Input value={d.reason} onChange={e => patchDeviation(d.id, { reason: e.target.value })} placeholder="Reason" className="h-7 text-xs" />
                    <button onClick={() => s.setProtocolDeviations(prev => prev.filter(x => x.id !== d.id))} className="text-muted-foreground hover:text-destructive"><Trash2 className="size-3.5" /></button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
