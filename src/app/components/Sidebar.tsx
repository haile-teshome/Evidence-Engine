import { useEffect, useRef, useState } from "react";
import { Card } from "./ui/card";
import { Label } from "./ui/label";
import { Checkbox } from "./ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Upload, FileText, X, Home, BarChart3, FileSearch, FlaskConical, Network, Table2, GitBranch, ShieldCheck, FileDown, ScanText, Sigma, Loader2, Users, PenLine } from "lucide-react";
import { Logo } from "./Logo";
import { ALL_SOURCES } from "../lib/mockServices";
import { useStore, PageId } from "../lib/store";
import { SessionsPanel } from "./SessionsPanel";

const TASK_LABEL: Record<string, string> = {
  "home-analysis": "Strategy analysis",
  "ai-optimize": "AI Optimize",
  "quality-assess": "Quality assessment",
  "abstract-screen": "Abstract screening",
  "fulltext-fetch": "Full-text fetch",
  "full-text-screen": "Full-text screening",
  "snowball": "Citation snowball",
  "snowball-screen": "Snowball screening",
  "table-extract": "Table extraction",
  "text-extract": "Text extraction",
};

// Friendly display names for known model tags. Keep the value field
// pointing at the actual Ollama / provider id; only the label changes.
function formatModelName(m: string): string {
  if (/leads.*mistral/i.test(m)) return "LEADS-Mistral 7B  (default — screening)";
  if (/medgemma.*27b/i.test(m)) return "MedGemma 27B  (clinical)";
  if (/medgemma/i.test(m)) return "MedGemma";
  if (/qwen2\.5.*7b/i.test(m)) return "Qwen 2.5 7B";
  if (/qwen2\.5/i.test(m)) return "Qwen 2.5";
  if (/llama3\.2.*3b/i.test(m)) return "Llama 3.2 3B  (fast)";
  if (/llama3\.1/i.test(m)) return "Llama 3.1";
  if (/llama/i.test(m)) return m.replace(/:latest$/, "");
  return m.replace(/^hf\.co\//, "").replace(/-GGUF.*$/, "").replace(/:latest$/, "");
}

// Full literal class strings (so Tailwind's scanner emits them) mapping each
// nav icon to a one-shot hover flourish. `motion-safe` respects reduced-motion.
const ANIM: Record<string, string> = {
  pop:    "motion-safe:group-hover:animate-[nav-pop_0.4s_ease-in-out]",
  bob:    "motion-safe:group-hover:animate-[nav-bob_0.4s_ease-in-out]",
  dip:    "motion-safe:group-hover:animate-[nav-dip_0.4s_ease-in-out]",
  spin:   "motion-safe:group-hover:animate-[nav-spin_0.5s_ease-in-out]",
  wiggle: "motion-safe:group-hover:animate-[nav-wiggle_0.5s_ease-in-out]",
  swing:  "motion-safe:group-hover:animate-[nav-swing_0.5s_ease-in-out]",
};

const NAV: { id: PageId; label: string; icon: any; anim: keyof typeof ANIM }[] = [
  { id: "home", label: "Home", icon: Home, anim: "bob" },
  { id: "simulation", label: "Planning", icon: BarChart3, anim: "pop" },
  { id: "projects", label: "Projects", icon: Users, anim: "pop" },
  { id: "abstract", label: "Abstract Screening", icon: FileSearch, anim: "wiggle" },
  { id: "acquisition", label: "Full-Text Acquisition", icon: FileDown, anim: "dip" },
  { id: "fulltext", label: "Full-Text Evidence", icon: FlaskConical, anim: "wiggle" },
  { id: "snowball", label: "Citation Snowball", icon: Network, anim: "spin" },
  { id: "extraction", label: "Table Extraction", icon: Table2, anim: "pop" },
  { id: "textextraction", label: "Text Extraction", icon: ScanText, anim: "bob" },
  { id: "quality", label: "Quality Assessment", icon: ShieldCheck, anim: "pop" },
  { id: "prisma", label: "Diagramming", icon: GitBranch, anim: "swing" },
  // Meta-analysis tab hidden for now — re-add to restore.
  // { id: "meta", label: "Meta-analysis", icon: Sigma, anim: "pop" },
  { id: "writing", label: "Writing Assistant", icon: PenLine, anim: "wiggle" },
];

export function Sidebar() {
  const s = useStore();
  const fileRef = useRef<HTMLInputElement>(null);
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [ollamaRunning, setOllamaRunning] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/api/models/local")
      .then(r => r.json())
      .then(d => {
        const models: string[] = Array.isArray(d.models) ? d.models : [];
        setLocalModels(models);
        setOllamaRunning(!!d.running);
        // Pick a sensible installed model. LEADS-mistral wins by benchmark
        // (recall=1.0, spec=0.68); fall back to medical-tuned > qwen2.5 > llama.
        const leadsTag = models.find(m => /leads.*mistral/i.test(m));
        const isLeadsAlias = s.model === "leads";
        if (isLeadsAlias && leadsTag) {
          // Resolve the "leads" alias to the actual Ollama tag so the dropdown
          // selection matches a real <SelectItem> and renders the friendly name.
          s.setModel(leadsTag);
        } else if (models.length > 0 && !models.includes(s.model) && !/^(claude|gpt|gemini)/.test(s.model)) {
          const preferred = leadsTag
            || models.find(m => /medgemma/i.test(m))
            || models.find(m => /qwen2\.5/i.test(m))
            || models.find(m => /llama3\.1/i.test(m))
            || models[0];
          s.setModel(preferred);
        }
      })
      .catch(() => setOllamaRunning(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleSource = (src: string) => {
    s.setSources(s.sources.includes(src) ? s.sources.filter(x => x !== src) : [...s.sources, src]);
  };

  // Only open-access sources are offered, so every listed database works without
  // any institutional login or subscription.
  const visibleSources = ALL_SOURCES;

  return (
    <aside className="w-72 shrink-0 border-r bg-muted/30 h-screen sticky top-0 flex flex-col">
      {/* Brand header — fixed bar, distinct from the navigation below. */}
      <div className="shrink-0 px-4 py-3.5 border-b bg-background/80 backdrop-blur-sm">
        <Logo />
      </div>

      {/* Fixed: active tasks + nav tabs stay put while the panels below scroll. */}
      <div className="shrink-0 px-4 pt-4 pb-2">
        {/* Active tasks (persists across page navigation) */}
        {Object.values(s.tasks).filter(t => t.status === "running").length > 0 && (
          <Card className="p-2 mb-3 bg-primary/5 border-primary/30 space-y-1">
            {Object.values(s.tasks)
              .filter(t => t.status === "running")
              .map(t => (
                <div key={t.kind} className="flex items-center gap-2 text-xs">
                  <Loader2 className="size-3 animate-spin text-primary shrink-0" />
                  <div className="flex-1 truncate font-medium">{TASK_LABEL[t.kind] || t.kind}</div>
                  <button
                    className="text-muted-foreground hover:text-foreground"
                    onClick={() => s.cancelTask(t.kind)}
                    title="Cancel"
                  >
                    <X className="size-3" />
                  </button>
                </div>
              ))}
          </Card>
        )}

        {/* Navigation */}
        <nav className="space-y-1 mb-4">
          {NAV.map(n => {
            const Icon = n.icon;
            const active = s.page === n.id;
            return (
              <button key={n.id} onClick={() => s.setPage(n.id)}
                className={`group w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors ${active ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}>
                <span className="inline-flex shrink-0 transition-transform duration-200 ease-out group-hover:scale-125">
                  <Icon className={`size-4 ${ANIM[n.anim]}`} />
                </span>
                {n.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Scrollable region: sessions, model, databases, local PDFs. */}
      <div className="flex-1 overflow-y-auto border-t px-4 pt-3 pb-4">
        <div className="mb-3">
          <SessionsPanel />
        </div>

        <Card className="p-3 mb-3">
          <Label className="mb-2 block">AI Model</Label>
          <Select value={s.model} onValueChange={s.setModel}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {localModels.length > 0 && (
                <>
                  <div className="px-2 py-1 text-xs text-muted-foreground">Local (Ollama)</div>
                  {/* Sort so the LEADS tag floats to the top of the local list. */}
                  {[...localModels]
                    .sort((a, b) => {
                      const aLeads = /leads.*mistral/i.test(a) ? 0 : 1;
                      const bLeads = /leads.*mistral/i.test(b) ? 0 : 1;
                      return aLeads - bLeads || a.localeCompare(b);
                    })
                    .map(m => (
                      <SelectItem key={m} value={m}>{formatModelName(m)}</SelectItem>
                    ))}
                </>
              )}
              <div className="px-2 pt-2 pb-1 text-xs text-muted-foreground">Cloud (API key required)</div>
              <SelectItem value="claude-opus-4-7">Claude Opus 4.7</SelectItem>
              <SelectItem value="claude-sonnet-4-6">Claude Sonnet 4.6</SelectItem>
              <SelectItem value="claude-haiku-4-5">Claude Haiku 4.5</SelectItem>
              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
              <SelectItem value="gpt-4o-mini">GPT-4o mini</SelectItem>
              <SelectItem value="gemini-1.5-pro">Gemini 1.5 Pro</SelectItem>
            </SelectContent>
          </Select>
          {ollamaRunning === false && (
            <p className="text-xs text-amber-600 mt-2">
              Setting up the local AI engine… the launcher installs and starts Ollama
              automatically. If this doesn't clear, relaunch — or pick a cloud model above.
            </p>
          )}
          {ollamaRunning && localModels.length === 0 && (
            <p className="text-xs text-amber-600 mt-2">
              The default AI model is downloading in the background (~4 GB, one time). Local
              screening will work once it finishes. You can use a cloud model in the meantime.
            </p>
          )}
        </Card>

        <Card className="p-3 mb-3">
          <Label className="mb-2 block">Active Databases</Label>
          <div className="space-y-2">
            {visibleSources.map(src => (
              <label key={src} className="flex items-center gap-2 cursor-pointer">
                <Checkbox checked={s.sources.includes(src)} onCheckedChange={() => toggleSource(src)} />
                <span className="text-sm">{src}</span>
              </label>
            ))}
          </div>
          {/* Papers-per-source and relevance-threshold sliders were removed in
              favour of automatic behaviour: the fetch budget is a fixed sane
              default, and the rerank endpoint auto-detects the natural
              relevance break from the score distribution itself. See
              `_auto_relevance_cutoff` in Backend/api.py. */}
        </Card>

      </div>
    </aside>
  );
}
