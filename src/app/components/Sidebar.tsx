import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Card } from "./ui/card";
import { Label } from "./ui/label";
import { Checkbox } from "./ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Upload, FileText, X, Home, BarChart3, FileSearch, FlaskConical, Network, Table2, GitBranch, ShieldCheck, FileDown, ScanText, Sigma, Loader2, Users, PenLine, KeyRound } from "lucide-react";
import { Logo } from "./Logo";
import { ALL_SOURCES } from "../lib/mockServices";
import { providerForModel } from "../lib/apiClient";
import { providerReady, needsUnlock as ksNeedsUnlock, subscribe as ksSubscribe } from "../lib/keystore";
import { subscribeDbKeys, hasDbKey, type DbSource } from "../lib/dbKeys";
import { ApiKeysDialog } from "./ApiKeysDialog";
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
  if (/leads.*mistral/i.test(m)) return "LEADS-Mistral 7B";
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
  // { id: "meta", label: "Meta-analysis", icon: Sigma, anim: "pop" },   // hidden for now
  { id: "writing", label: "Writing Assistant", icon: PenLine, anim: "wiggle" },
];

// Grouping + per-source annotations for the Active Databases panel. Groups make the
// growing source list scannable; notes flag the ones that need a key or behave
// differently, so the panel is honest about what each source will actually return.
type SourceMeta = { name: string; group: string; note?: string; key?: "required" | "optional" };
const SOURCE_META: SourceMeta[] = [
  { name: "PubMed", group: "Bibliographic", key: "optional" },
  { name: "Europe PMC", group: "Bibliographic" },
  { name: "OpenAlex", group: "Bibliographic" },
  { name: "CrossRef", group: "Bibliographic" },
  { name: "Semantic Scholar", group: "Bibliographic", key: "optional" },
  { name: "DOAJ", group: "Bibliographic" },
  { name: "CORE", group: "Bibliographic", key: "required" },
  { name: "Springer Nature", group: "Bibliographic", key: "required" },
  { name: "IEEE Xplore", group: "Bibliographic", key: "required" },
  { name: "arXiv", group: "Preprints" },
  { name: "bioRxiv", group: "Preprints" },
  { name: "medRxiv", group: "Preprints" },
  { name: "ClinicalTrials.gov", group: "Trials & grey literature" },
  { name: "Scopus", group: "Subscription", key: "required" },
  { name: "Web of Science", group: "Subscription", key: "required" },
];
const SOURCE_GROUPS = ["Bibliographic", "Preprints", "Trials & grey literature", "Subscription"];
// The data-source key that backs each keyed source, for the 🔑 status indicator.
const DB_KEY_FOR: Record<string, DbSource> = {
  "PubMed": "ncbi",
  "Semantic Scholar": "semantic_scholar",
  "CORE": "core",
  "Springer Nature": "springer",
  "IEEE Xplore": "ieee",
  "Scopus": "scopus",
  "Web of Science": "wos",
};

export function Sidebar() {
  const s = useStore();
  const fileRef = useRef<HTMLInputElement>(null);
  // Active-nav pill: one glass element positioned over the active tab. We measure
  // each nav button and spring the pill's y/height whenever the active page
  // changes, so it always animates from its current spot to the clicked tab.
  const navBtns = useRef<Record<string, HTMLButtonElement | null>>({});
  const [pill, setPill] = useState<{ top: number; height: number } | null>(null);
  useLayoutEffect(() => {
    const el = navBtns.current[s.page];
    if (el) setPill({ top: el.offsetTop, height: el.offsetHeight });
  }, [s.page]);
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [ollamaRunning, setOllamaRunning] = useState<boolean | null>(null);
  const [keysOpen, setKeysOpen] = useState(false);
  const [, forceKeys] = useState(0);
  // Re-render when the keystore or a database key changes (so the 🔑 status updates).
  useEffect(() => {
    const bump = () => forceKeys(v => v + 1);
    const offKs = ksSubscribe(bump);
    const offDb = subscribeDbKeys(bump);
    return () => { offKs(); offDb(); };
  }, []);

  // Cloud model selected: does it need a key, or an unlock of encrypted keys?
  const modelProvider = providerForModel(s.model);
  const needUnlock = !!modelProvider && ksNeedsUnlock();
  const missingKey = !!modelProvider && !needUnlock && !providerReady(modelProvider);

  useEffect(() => {
    fetch("/api/models/local")
      .then(r => r.json())
      .then(d => {
        const models: string[] = Array.isArray(d.models) ? d.models : [];
        setLocalModels(models);
        setOllamaRunning(!!d.running);
        // Pick a sensible installed model. LEADS-mistral wins by benchmark
        // (recall=1.0, spec=0.68); otherwise prefer Qwen, which is the house
        // default for local work. Llama tags are never auto-selected.
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
            || models.find(m => /qwen/i.test(m))
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

  // Most listed databases are open access and work with no login. The
  // "Subscription" group (Scopus, Web of Science) and the keyed Bibliographic
  // entries need an API key, which is why each carries `key: "required"` and a
  // 🔑 status indicator. "Local PDFs" is not a searchable database (attaching
  // your own PDFs is done from the chat's + button, which records them
  // automatically), so it isn't shown as a togglable source here.
  const visibleSources = ALL_SOURCES.filter(src => src !== "Local PDFs");

  return (
    <aside className="w-72 shrink-0 h-screen sticky top-0 flex flex-col border-r border-black/[0.08] dark:border-white/[0.1] bg-background/55 backdrop-blur-2xl backdrop-saturate-150 shadow-[inset_-1px_0_0_0_rgba(255,255,255,0.12)]">
      {/* Brand header: fixed bar, distinct from the navigation below. */}
      <div className="shrink-0 px-4 py-3.5 border-b border-white/30 dark:border-white/[0.06] bg-background/40 backdrop-blur-xl backdrop-saturate-150">
        <Logo />
      </div>

      {/* Fixed: active tasks + nav tabs stay put while the panels below scroll. */}
      <div className="shrink-0 px-4 pt-4 pb-2">
        {/* Active tasks bar: pinned ABOVE the nav so it stays in the same place
            across every tab and action. The nav's sliding pill is measured
            relative to the <nav> element, so this card mounting/unmounting shifts
            the nav as a whole but never drags the pill out of alignment. */}
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
        {/* Navigation. A single glass pill sits over the active tab and springs its
            y/height whenever the active page changes, so it always animates from its
            current spot to the clicked tab regardless of unrelated re-renders. */}
        <nav className="relative space-y-1 mb-4">
          {pill && (
            <motion.span
              className="pointer-events-none absolute left-0 right-0 top-0 rounded-xl overflow-hidden bg-gradient-to-b from-primary/80 to-primary/60 backdrop-blur-xl backdrop-saturate-150 border border-white/40 dark:border-white/20 ring-1 ring-inset ring-white/25 shadow-[0_8px_24px_-8px_rgba(0,0,0,0.28),0_1px_0_0_rgba(255,255,255,0.45)_inset,0_-10px_18px_-10px_rgba(0,0,0,0.2)_inset]"
              initial={false}
              animate={{ y: pill.top, height: pill.height }}
              transition={{ type: "spring", stiffness: 420, damping: 34, mass: 0.7 }}
            >
              {/* Specular glass sheen: bright top-lit highlight fading down */}
              <span className="absolute inset-0 bg-gradient-to-b from-white/35 via-white/5 to-transparent" />
              {/* Crisp top edge line */}
              <span className="absolute inset-x-1.5 top-0 h-px bg-white/60 rounded-full blur-[0.3px]" />
            </motion.span>
          )}
          {NAV.map(n => {
            const Icon = n.icon;
            const active = s.page === n.id;
            return (
              <button key={n.id} ref={el => { navBtns.current[n.id] = el; }} onClick={() => s.setPage(n.id)}
                className={`group relative z-10 w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors duration-200 ${
                  active
                    ? "text-primary-foreground"
                    : "text-foreground/80 hover:bg-white/50 dark:hover:bg-white/[0.06] hover:backdrop-blur-sm"
                }`}>
                <span className="relative z-10 inline-flex shrink-0 transition-transform duration-200 ease-out group-hover:scale-125">
                  <Icon className={`size-4 ${ANIM[n.anim]}`} />
                </span>
                <span className="relative z-10">{n.label}</span>
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
              Setting up the local AI engine. The launcher installs and starts Ollama
              automatically. If this doesn't clear, relaunch or pick a cloud model above.
            </p>
          )}
          {ollamaRunning && localModels.length === 0 && (
            <p className="text-xs text-amber-600 mt-2">
              The default AI model is downloading in the background (~4 GB, one time). Local
              screening will work once it finishes. You can use a cloud model in the meantime.
            </p>
          )}
          {needUnlock || missingKey ? (
            <div className="mt-2 flex items-center gap-2 rounded-md border border-amber-300/70 bg-amber-50 dark:border-amber-800/50 dark:bg-amber-950/30 px-2.5 py-2">
              <KeyRound className="size-4 shrink-0 text-amber-600 dark:text-amber-400" />
              <span className="flex-1 text-xs text-amber-800 dark:text-amber-200 leading-snug">
                {needUnlock ? "Unlock your keys to use this model." : "This model needs an API key to run."}
              </span>
              <button type="button" onClick={() => setKeysOpen(true)}
                className="shrink-0 rounded-md bg-amber-600 px-2 py-1 text-xs font-medium leading-none text-white hover:bg-amber-700">
                {needUnlock ? "Unlock" : "Add key"}
              </button>
            </div>
          ) : (
            <button type="button" onClick={() => setKeysOpen(true)}
              className="mt-2 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
              <KeyRound className="size-3.5" />Manage API keys
            </button>
          )}
        </Card>

        <ApiKeysDialog open={keysOpen} onOpenChange={setKeysOpen} highlight={modelProvider} />

        <Card className="p-3 mb-3">
          <Label className="block mb-2.5">Active Databases</Label>
          <div className="space-y-3">
            {SOURCE_GROUPS.map(group => {
              const items = SOURCE_META.filter(m => m.group === group && visibleSources.includes(m.name));
              if (!items.length) return null;
              return (
                <div key={group}>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80 mb-1.5">{group}</div>
                  <div className="space-y-1.5">
                    {items.map(m => (
                      <label key={m.name} className="flex items-center gap-2 cursor-pointer">
                        <Checkbox checked={s.sources.includes(m.name)} onCheckedChange={() => toggleSource(m.name)} />
                        <span className="text-sm flex-1">{m.name}</span>
                        {m.note && <span className="text-[10px] shrink-0 text-muted-foreground">{m.note}</span>}
                        {m.key && (
                          <button type="button"
                            onClick={e => { e.preventDefault(); e.stopPropagation(); setKeysOpen(true); }}
                            title={hasDbKey(DB_KEY_FOR[m.name])
                              ? "API key set — click to manage"
                              : m.key === "required"
                                ? "Needs a free API key — click to add"
                                : "Optional API key: avoids rate limits — click to add"}
                            className={`inline-flex shrink-0 text-muted-foreground hover:text-foreground ${hasDbKey(DB_KEY_FOR[m.name]) ? "opacity-30 hover:opacity-100" : ""}`}
                          >
                            <KeyRound className="size-3.5" />
                          </button>
                        )}
                      </label>
                    ))}
                  </div>
                </div>
              );
            })}
            {/* Any source not yet categorised in SOURCE_META still shows up. */}
            {(() => {
              const known = new Set(SOURCE_META.map(m => m.name));
              const others = visibleSources.filter(x => !known.has(x));
              return others.length ? (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80 mb-1.5">Other</div>
                  <div className="space-y-1.5">
                    {others.map(src => (
                      <label key={src} className="flex items-center gap-2 cursor-pointer">
                        <Checkbox checked={s.sources.includes(src)} onCheckedChange={() => toggleSource(src)} />
                        <span className="text-sm flex-1">{src}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ) : null;
            })()}
          </div>
          {/* Papers-per-source and relevance-threshold sliders were removed in
              favour of automatic behaviour: the fetch budget is a fixed sane
              default, and the rerank endpoint auto-detects the natural
              relevance break from the score distribution itself. See
              `_auto_relevance_cutoff` in Backend/api.py. */}
          <button type="button" onClick={() => setKeysOpen(true)}
            className="mt-3 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
            <KeyRound className="size-3.5" />Manage API keys
          </button>
        </Card>

      </div>
    </aside>
  );
}
