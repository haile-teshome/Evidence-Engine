import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import { useAuth } from "../lib/auth";
import {
  Project, ProjectMember, ScreeningMode, ProjectRole, Invite, AssignmentStrategy,
  listProjects, createProject, getProject, createInvite, setMemberRole,
  lockProject, acceptInvite, previewInvite, setProjectPapers, assignPapers,
} from "../lib/projects";
import { effectiveAbstractDecision } from "../lib/exclusionBucketing";
import { ConflictsSection } from "../components/ConflictsSection";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Badge } from "../components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Textarea } from "../components/ui/textarea";
import { Users, Plus, X, Link as LinkIcon, ShieldCheck, Lock, CheckCircle2, AlertTriangle, Copy } from "lucide-react";
import { toast } from "sonner";

export function ProjectsPage() {
  const s = useStore();
  const { user } = useAuth();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [acceptToken, setAcceptToken] = useState("");
  const [acceptPreview, setAcceptPreview] = useState<{ invite: Invite; project: { id: string; name: string } | null } | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      setProjects(await listProjects());
    } catch (e: any) {
      toast.error(e?.message || "Failed to load projects");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { if (user) refresh(); }, [user]);

  // Pick up `?invite=TOKEN` from the URL (the invite-link landing flow).
  useEffect(() => {
    const url = new URL(window.location.href);
    const tok = url.searchParams.get("invite");
    if (tok) {
      setAcceptToken(tok);
      previewInvite(tok).then(setAcceptPreview).catch(e => toast.error(e?.message || "Invalid invite"));
    }
  }, []);

  async function openProject(p: Project) {
    try {
      const { project, members } = await getProject(p.id);
      s.setCurrentProjectId(project.id);
      s.setCurrentProjectName(project.name);
      s.setCurrentProjectRole(project.my_role || null);
      s.setCurrentProjectMode(project.screening_mode);
      // Hydrate PICO / criteria into the legacy fields so existing screening
      // pages work unchanged. The project is the source of truth; on save
      // these flow back into the project via the screening API.
      s.setPico(project.pico);
      s.setInclusion(project.inclusion);
      s.setExclusion(project.exclusion);
      toast.success(`Opened ${project.name} (${members.length} member${members.length === 1 ? "" : "s"})`);
      s.setPage("home");
    } catch (e: any) {
      toast.error(e?.message || "Failed to open project");
    }
  }

  async function handleAcceptInvite() {
    if (!acceptToken) return;
    try {
      const r = await acceptInvite(acceptToken);
      toast.success(r.already_member ? "Already a member" : `Joined project as ${r.role}`);
      setAcceptToken("");
      setAcceptPreview(null);
      // Strip the ?invite= param from the URL so reload doesn't re-prompt
      const url = new URL(window.location.href);
      url.searchParams.delete("invite");
      window.history.replaceState({}, "", url.toString());
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || "Failed to accept invite");
    }
  }

  if (!user) {
    return (
      <Alert>
        <AlertDescription>
          Sign in (top-right) to create projects and collaborate with other reviewers.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      {acceptPreview && (
        <Card className="p-4 border-primary/40">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="font-medium">Invitation to join {acceptPreview.project?.name || "a project"}</div>
              <div className="text-sm text-muted-foreground mt-1">
                You'll join as <strong>{acceptPreview.invite.role}</strong>. The invite was created on{" "}
                {new Date(acceptPreview.invite.created_at).toLocaleDateString()}.
              </div>
            </div>
            <div className="flex gap-2 shrink-0">
              <Button variant="ghost" onClick={() => { setAcceptToken(""); setAcceptPreview(null); }}>Dismiss</Button>
              <Button onClick={handleAcceptInvite}><CheckCircle2 className="size-4 mr-2" />Accept</Button>
            </div>
          </div>
        </Card>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-medium">Your projects</h2>
          <div className="text-xs text-muted-foreground">
            A project is a shared systematic review with multiple reviewers, blinded screening, and adjudication.
          </div>
        </div>
        <Button onClick={() => setShowCreate(true)}><Plus className="size-4 mr-2" />New project</Button>
      </div>

      {showCreate && <CreateProjectCard onCreated={(p) => { setShowCreate(false); refresh(); openProject(p); }} onCancel={() => setShowCreate(false)} />}

      {s.currentProjectId && (
        <>
          <ProjectDetailCard projectId={s.currentProjectId} onChanged={refresh} />
          <ConflictsSection projectId={s.currentProjectId} />
        </>
      )}

      <Card className="p-4">
        <div className="font-medium mb-3">All projects ({projects.length})</div>
        {loading && <div className="text-sm text-muted-foreground">Loading…</div>}
        {!loading && projects.length === 0 && (
          <Alert>
            <AlertDescription>
              No projects yet. Create one to invite collaborators, or paste an invite token below.
            </AlertDescription>
          </Alert>
        )}
        <div className="space-y-2">
          {projects.map(p => (
            <button
              key={p.id}
              onClick={() => openProject(p)}
              className="w-full text-left p-3 border rounded hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-medium truncate">{p.name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2 flex-wrap">
                    <ModeBadge mode={p.screening_mode} />
                    <Badge variant="outline" className="text-[10px]">{p.my_role}</Badge>
                    {p.locked_at && <Badge variant="outline" className="text-[10px] bg-amber-50 text-amber-700 border-amber-200">
                      <Lock className="size-3 mr-0.5 inline" />Locked
                    </Badge>}
                    <span>Updated {new Date(p.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
                <Users className="size-4 text-muted-foreground shrink-0" />
              </div>
            </button>
          ))}
        </div>
      </Card>

      <Card className="p-4">
        <div className="font-medium mb-2">Have an invite token?</div>
        <div className="flex gap-2">
          <Input
            value={acceptToken}
            onChange={(e) => setAcceptToken(e.target.value)}
            placeholder="Paste invite token here"
            className="flex-1"
          />
          <Button
            variant="outline"
            disabled={!acceptToken}
            onClick={async () => {
              try {
                setAcceptPreview(await previewInvite(acceptToken));
              } catch (e: any) {
                toast.error(e?.message || "Invalid invite");
              }
            }}
          >
            Preview
          </Button>
        </div>
      </Card>
    </div>
  );
}

// ---- Create form ----------------------------------------------------------

// =========================================================================
// Create Review Wizard
// -------------------------------------------------------------------------
// A 4-step flow that turns the current single-user workspace into a shared
// multi-reviewer project:
//   1. Name + screening mode
//   2. Source: which subset of the workspace's papers to seed with
//   3. Reviewers: bulk invite + role assignment
//   4. Assignment: full overlap (Cochrane default) or round-robin split
// At the end, the project is created, papers imported, invite links generated,
// and (optional) per-paper assignments materialised. The user gets a summary
// and the invite links to copy.
// =========================================================================

type SourceKind = "quality" | "unique" | "raw" | "screened" | "included" | "custom";

function CreateProjectCard({
  onCreated, onCancel,
}: {
  onCreated: (p: Project) => void;
  onCancel: () => void;
}) {
  const s = useStore();
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [saving, setSaving] = useState(false);

  // Step 1
  const [name, setName] = useState("");
  const [mode, setMode] = useState<ScreeningMode>("dual_blinded");

  // Step 2
  const [source, setSource] = useState<SourceKind>("unique");
  const [customCsv, setCustomCsv] = useState("");

  // Step 3
  const [pendingInvites, setPendingInvites] = useState<{ role: ProjectRole; label: string }[]>([
    { role: "reviewer", label: "Reviewer #1" },
    { role: "reviewer", label: "Reviewer #2" },
    { role: "adjudicator", label: "Adjudicator" },
  ]);

  // Step 4
  const [strategy, setStrategy] = useState<AssignmentStrategy>("full_overlap");
  const [reviewersPerPaper, setReviewersPerPaper] = useState(2);

  // ----- Source preview --------------------------------------------------
  const sources = useMemoSources(s);
  const chosenSource = sources[source];
  const seedCount = source === "custom" ? customCsv.split("\n").filter(l => l.trim()).length - 1 : chosenSource.count;

  const expectedReviewers = pendingInvites.filter(i => i.role !== "viewer").length + 1; // + the lead (you)

  async function finalize() {
    if (!name.trim()) { toast.error("Name is required"); return; }
    setSaving(true);
    try {
      // 1. Create the project
      const project = await createProject({
        name: name.trim(),
        screening_mode: mode,
        pico: s.pico,
        inclusion: s.inclusion,
        exclusion: s.exclusion,
      });

      // 2. Seed papers from the chosen source
      const seeds = source === "custom"
        ? parseCustomCsv(customCsv)
        : chosenSource.materialise();
      if (seeds.length > 0) {
        await setProjectPapers(project.id, seeds);
      }

      // 3. Generate invite links
      const links: { role: ProjectRole; url: string }[] = [];
      for (const inv of pendingInvites) {
        const created = await createInvite(project.id, inv.role);
        links.push({
          role: inv.role,
          url: `${window.location.origin}/?invite=${created.token}`,
        });
      }

      // 4. (Optional) materialise assignments. Note: only the *lead* exists
      //    in the project at this moment — reviewers join when they accept
      //    invites. So if the user picked split-assignment we skip the API
      //    call now and surface a hint to re-run it once members join. For
      //    full_overlap, no assignment write is required (the project
      //    behaves as full-overlap by default).
      const willPostAssignNow = strategy === "split" && false; // never now
      if (willPostAssignNow) {
        await assignPapers(project.id, { strategy, reviewers_per_paper: reviewersPerPaper });
      }

      // 5. Show summary
      toast.success(
        `Created ${project.name}: ${seeds.length} papers, ${links.length} invite links${strategy === "split" ? ` · split (${reviewersPerPaper}/paper) — assign after reviewers join` : ""}`,
        { duration: 6000 },
      );

      // 6. Copy the first invite link to the clipboard as a convenience
      if (links.length > 0 && links[0].url) {
        try { await navigator.clipboard.writeText(links.map(l => `${l.role}: ${l.url}`).join("\n")); } catch {}
      }

      onCreated(project);
    } catch (e: any) {
      toast.error(e?.message || "Failed to create project");
    } finally {
      setSaving(false);
    }
  }

  // -----------------------------------------------------------------------

  return (
    <Card className="p-4 space-y-4 border-primary/40">
      <div className="flex items-center justify-between">
        <div className="font-medium">New review · Step {step} of 4</div>
        <StepDots step={step} />
      </div>

      {step === 1 && (
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs">Review name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Pet ownership and cardiovascular health 2026" />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Screening mode</Label>
            <Select value={mode} onValueChange={(v) => setMode(v as ScreeningMode)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="dual_blinded">Dual blinded — Cochrane standard, conflicts adjudicated</SelectItem>
                <SelectItem value="dual">Dual unblinded — reviewers see each other in real time</SelectItem>
                <SelectItem value="single">Single — one reviewer (legacy mode)</SelectItem>
              </SelectContent>
            </Select>
            <div className="text-xs text-muted-foreground">
              {mode === "dual_blinded" && "Each reviewer screens independently; only their own decisions are visible until both have decided. Conflicts route to the adjudicator. Recommended for published reviews."}
              {mode === "dual" && "Each reviewer sees the others' decisions live. Faster, but introduces bias."}
              {mode === "single" && "One reviewer. Equivalent to running the platform without a project."}
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            Your current PICO and {s.inclusion.length + s.exclusion.length} inclusion/exclusion criteria will be copied to the review and can be edited later.
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="space-y-3">
          <div className="text-sm">Pick which subset of your workspace to seed the review with. Reviewers will screen exactly this set.</div>
          <div className="space-y-2">
            {Object.entries(sources).map(([k, v]) => (
              <SourceOption
                key={k}
                value={k as SourceKind}
                selected={source}
                onSelect={setSource}
                title={v.title}
                count={v.count}
                description={v.description}
                disabled={v.count === 0 && k !== "custom"}
              />
            ))}
          </div>
          {source === "custom" && (
            <div className="space-y-1.5">
              <Label className="text-xs">Paste CSV (title,abstract,source,url,doi — first row = headers)</Label>
              <Textarea
                value={customCsv}
                onChange={(e) => setCustomCsv(e.target.value)}
                rows={5}
                className="font-mono text-[11px]"
                placeholder="title,abstract,source,url,doi&#10;Some paper,Its abstract...,PubMed,https://...,10.1234/..."
              />
            </div>
          )}
        </div>
      )}

      {step === 3 && (
        <div className="space-y-3">
          <div className="text-sm">Invite reviewers. We'll generate one-time links you can share by email or Slack.</div>
          <div className="space-y-2">
            {pendingInvites.map((inv, i) => (
              <div key={i} className="flex items-center gap-2">
                <Input
                  value={inv.label}
                  onChange={(e) => {
                    const next = [...pendingInvites];
                    next[i] = { ...next[i], label: e.target.value };
                    setPendingInvites(next);
                  }}
                  placeholder="Reviewer label (e.g. 'Jane' — for your tracking only)"
                  className="flex-1"
                />
                <Select
                  value={inv.role}
                  onValueChange={(v) => {
                    const next = [...pendingInvites];
                    next[i] = { ...next[i], role: v as ProjectRole };
                    setPendingInvites(next);
                  }}
                >
                  <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="reviewer">Reviewer</SelectItem>
                    <SelectItem value="adjudicator">Adjudicator</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setPendingInvites(pendingInvites.filter((_, j) => j !== i))}
                  className="shrink-0"
                >
                  <X className="size-4" />
                </Button>
              </div>
            ))}
            <Button
              size="sm"
              variant="outline"
              onClick={() => setPendingInvites([...pendingInvites, { role: "reviewer", label: `Reviewer #${pendingInvites.filter(p => p.role === "reviewer").length + 1}` }])}
            >
              <Plus className="size-4 mr-1" /> Add reviewer
            </Button>
          </div>
          <Alert>
            <AlertDescription className="text-xs">
              The lead (you) is automatically a member with full access. Invitees receive one-time links — after they click and sign up, they're added to the project with the role above.
            </AlertDescription>
          </Alert>
        </div>
      )}

      {step === 4 && (
        <div className="space-y-3">
          <div className="text-sm">How should papers be distributed across reviewers?</div>
          <div className="space-y-2">
            <StrategyOption
              value="full_overlap"
              selected={strategy}
              onSelect={setStrategy}
              title="Full overlap (Cochrane standard)"
              description={`Every reviewer screens every paper. Required for dual-blinded review with adjudication. Workload: ${seedCount} papers × ${expectedReviewers} reviewers = ${seedCount * expectedReviewers} screenings.`}
            />
            <StrategyOption
              value="split"
              selected={strategy}
              onSelect={setStrategy}
              title="Round-robin split"
              description={`Each paper is assigned to ${reviewersPerPaper} reviewers, distributed round-robin. Faster, no full duplicate screening. Workload: ${seedCount} papers × ${reviewersPerPaper} reviewers = ${seedCount * reviewersPerPaper} screenings.`}
            />
          </div>
          {strategy === "split" && (
            <div className="space-y-1.5 pl-6">
              <Label className="text-xs">Reviewers per paper</Label>
              <Input
                type="number"
                min={1}
                max={Math.max(1, expectedReviewers)}
                value={reviewersPerPaper}
                onChange={(e) => setReviewersPerPaper(Math.max(1, parseInt(e.target.value || "1", 10)))}
                className="w-24"
              />
              <div className="text-[11px] text-muted-foreground">
                Cochrane requires ≥ 2 reviewers per paper for primary screening; setting to 1 skips dual screening.
              </div>
            </div>
          )}
          {strategy === "split" && (
            <Alert className="bg-amber-50 border-amber-200">
              <AlertTriangle className="size-4 inline mr-1 text-amber-700" />
              <AlertDescription className="text-xs text-amber-900">
                Assignment runs after reviewers accept their invites — the project starts as full-overlap by default. Open the project's <strong>Detail</strong> card and click <em>Re-distribute</em> once everyone has joined.
              </AlertDescription>
            </Alert>
          )}

          <div className="text-xs text-muted-foreground border-t pt-3">
            <strong>Summary:</strong> Create review <em>{name || "(unnamed)"}</em> in <em>{mode}</em> mode with {seedCount} seed papers from <em>{chosenSource.title}</em>, {pendingInvites.length} invite link{pendingInvites.length === 1 ? "" : "s"}, and <em>{strategy === "full_overlap" ? "full-overlap" : `split (${reviewersPerPaper}/paper, applied after join)`}</em> assignment.
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-2 border-t">
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
        <div className="flex gap-2">
          {step > 1 && (
            <Button variant="outline" onClick={() => setStep((step - 1) as 1 | 2 | 3 | 4)}>Back</Button>
          )}
          {step < 4 && (
            <Button
              onClick={() => setStep((step + 1) as 1 | 2 | 3 | 4)}
              disabled={step === 1 && !name.trim()}
            >
              Next
            </Button>
          )}
          {step === 4 && (
            <Button onClick={finalize} disabled={saving || !name.trim()}>
              {saving ? "Creating…" : "Create review"}
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}

// ---- Wizard support pieces ------------------------------------------------

function StepDots({ step }: { step: 1 | 2 | 3 | 4 }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4].map(i => (
        <span key={i} className={`size-2 rounded-full ${i <= step ? "bg-primary" : "bg-muted-foreground/30"}`} />
      ))}
    </div>
  );
}

function SourceOption({
  value, selected, onSelect, title, count, description, disabled,
}: {
  value: SourceKind;
  selected: SourceKind;
  onSelect: (v: SourceKind) => void;
  title: string;
  count: number;
  description: string;
  disabled?: boolean;
}) {
  const active = selected === value;
  return (
    <button
      type="button"
      onClick={() => !disabled && onSelect(value)}
      disabled={disabled}
      className={`w-full text-left p-3 border rounded transition-colors ${
        active ? "border-primary bg-primary/5" :
        disabled ? "opacity-50 cursor-not-allowed" :
        "hover:bg-muted/40"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium text-sm">{title}</div>
        <Badge variant="outline" className="text-[10px] tabular-nums">{count} papers</Badge>
      </div>
      <div className="text-xs text-muted-foreground mt-1">{description}</div>
    </button>
  );
}

function StrategyOption({
  value, selected, onSelect, title, description,
}: {
  value: AssignmentStrategy;
  selected: AssignmentStrategy;
  onSelect: (v: AssignmentStrategy) => void;
  title: string;
  description: string;
}) {
  const active = selected === value;
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      className={`w-full text-left p-3 border rounded transition-colors ${active ? "border-primary bg-primary/5" : "hover:bg-muted/40"}`}
    >
      <div className="font-medium text-sm">{title}</div>
      <div className="text-xs text-muted-foreground mt-1">{description}</div>
    </button>
  );
}

// ---- Source materialisation (workspace -> ProjectPaper[]) ----------------

function useMemoSources(s: ReturnType<typeof useStore>) {
  // Pull whatever the workspace currently has and present each layer as an
  // option. The "kept" count reflects effective decisions (post-override).
  const qualityKept = s.qualityReports
    ? s.qualityReports.filter(r => !s.excludedByQuality.has(r.paper_id)).length
    : 0;
  const screened = s.results?.length || 0;
  const includedAfterScreening = s.results
    ? s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE").length
    : 0;

  return {
    quality: {
      title: "Quality-assessed (post-QA)",
      count: qualityKept,
      description: `Unique papers that passed your Quality Assessment exclusion rule. The cleanest hand-off into shared screening.`,
      materialise: () => {
        if (!s.qualityReports || !s.uniquePapers) return [] as any[];
        const keep = new Set(s.qualityReports
          .filter(r => !s.excludedByQuality.has(r.paper_id))
          .map(r => r.paper_id));
        return s.uniquePapers.filter(p => keep.has(p.id)).map(p => ({
          paper_id: p.id, title: p.title, abstract: p.abstract || "", source: p.source,
          url: (p as any).url || "", doi: (p as any).doi || "",
        }));
      },
    },
    unique: {
      title: "Aggregated unique papers (post-dedup)",
      count: s.uniquePapers?.length || 0,
      description: "All unique papers retrieved across your active databases, after deduplication. Use this if Quality Assessment hasn't been run.",
      materialise: () => (s.uniquePapers || []).map(p => ({
        paper_id: p.id, title: p.title, abstract: p.abstract || "", source: p.source,
        url: (p as any).url || "", doi: (p as any).doi || "",
      })),
    },
    raw: {
      title: "Raw aggregated papers (pre-dedup)",
      count: s.rawPapers?.length || 0,
      description: "Every record fetched from databases, including duplicates. Rarely the right choice — only use when you specifically want pre-dedup state.",
      materialise: () => (s.rawPapers || []).map(p => ({
        paper_id: p.id, title: p.title, abstract: p.abstract || "", source: p.source,
        url: (p as any).url || "", doi: (p as any).doi || "",
      })),
    },
    screened: {
      title: "AI-screened papers (all)",
      count: screened,
      description: "Every paper that went through Abstract Screening (both AI-included and AI-excluded). Useful if you want reviewers to re-evaluate AI exclusions.",
      materialise: () => (s.results || []).map(r => ({
        paper_id: r.paper_id, title: r.Title, abstract: r.Abstract || "", source: r.Source,
        url: r.URL || "", doi: "",
      })),
    },
    included: {
      title: "AI-included only (effective decisions)",
      count: includedAfterScreening,
      description: "Papers that came out of Abstract Screening as INCLUDE (honouring reviewer overrides). Use this to escalate a single-user pass to a multi-reviewer second pass.",
      materialise: () => (s.results || [])
        .filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE")
        .map(r => ({
          paper_id: r.paper_id, title: r.Title, abstract: r.Abstract || "", source: r.Source,
          url: r.URL || "", doi: "",
        })),
    },
    custom: {
      title: "Custom CSV",
      count: 0,
      description: "Paste a CSV (title,abstract,source,url,doi) below. Use this when the project is independent from your current workspace.",
      materialise: () => [] as any[],
    },
  };
}

function parseCustomCsv(csv: string) {
  const lines = csv.split("\n").filter(l => l.trim());
  if (lines.length < 2) return [] as any[];
  const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
  const out: any[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCsvLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((h, j) => { row[h] = (cols[j] || "").trim(); });
    out.push({
      paper_id: row.id || row.doi || `csv_${i}`,
      title: row.title || "(no title)",
      abstract: row.abstract || "",
      source: row.source || "CSV",
      url: row.url || "",
      doi: row.doi || "",
    });
  }
  return out;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; continue; }
    if (ch === '"') { inQ = !inQ; continue; }
    if (ch === "," && !inQ) { out.push(cur); cur = ""; continue; }
    cur += ch;
  }
  out.push(cur);
  return out;
}

// ---- Detail / member-management card --------------------------------------

function ProjectDetailCard({ projectId, onChanged }: { projectId: string; onChanged: () => void }) {
  const s = useStore();
  const [project, setProject] = useState<Project | null>(null);
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [inviteRole, setInviteRole] = useState<ProjectRole>("reviewer");
  const [lastInvite, setLastInvite] = useState<Invite | null>(null);
  const [busy, setBusy] = useState(false);
  const [distributeStrategy, setDistributeStrategy] = useState<AssignmentStrategy>("full_overlap");
  const [reviewersPerPaper, setReviewersPerPaper] = useState(2);

  async function load() {
    try {
      const r = await getProject(projectId);
      setProject(r.project);
      setMembers(r.members);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load project");
    }
  }

  useEffect(() => { load(); }, [projectId]);

  async function handleInvite() {
    setBusy(true);
    try {
      const inv = await createInvite(projectId, inviteRole);
      setLastInvite(inv);
      toast.success(`Invite link created (${inviteRole})`);
    } catch (e: any) {
      toast.error(e?.message || "Failed to create invite");
    } finally {
      setBusy(false);
    }
  }

  async function handleRoleChange(userId: string, role: ProjectRole) {
    try {
      await setMemberRole(projectId, userId, role);
      toast.success(`Role updated`);
      load();
    } catch (e: any) {
      toast.error(e?.message || "Failed to change role");
    }
  }

  async function handleLock() {
    if (!confirm("Lock this project? Reviewers will no longer be able to add or change decisions. This is for final analysis.")) return;
    try {
      await lockProject(projectId);
      toast.success("Project locked for analysis");
      load();
      onChanged();
    } catch (e: any) {
      toast.error(e?.message || "Failed to lock project");
    }
  }

  async function handleDistribute() {
    setBusy(true);
    try {
      const r = await assignPapers(projectId, {
        strategy: distributeStrategy,
        reviewers_per_paper: reviewersPerPaper,
      });
      toast.success(
        `Distributed ${r.papers} papers across ${r.reviewers} reviewers (${r.assigned} assignments, ${distributeStrategy})`,
        { duration: 5000 },
      );
    } catch (e: any) {
      toast.error(e?.message || "Failed to distribute papers");
    } finally {
      setBusy(false);
    }
  }

  if (!project) return null;

  const isLead = project.my_role === "lead";
  const inviteUrl = lastInvite ? `${window.location.origin}/?invite=${lastInvite.token}` : "";

  return (
    <Card className="p-4 space-y-3 border-primary/40">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-medium">{project.name}</div>
          <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2 flex-wrap">
            <ModeBadge mode={project.screening_mode} />
            <Badge variant="outline" className="text-[10px]">My role: {project.my_role}</Badge>
            {project.locked_at && <Badge variant="outline" className="text-[10px] bg-amber-50 text-amber-700 border-amber-200"><Lock className="size-3 mr-0.5 inline" />Locked</Badge>}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          {isLead && !project.locked_at && (
            <Button variant="outline" size="sm" onClick={handleLock}>
              <Lock className="size-4 mr-1" />Lock for analysis
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => {
            s.setCurrentProjectId(null);
            s.setCurrentProjectName("");
            s.setCurrentProjectRole(null);
            s.setCurrentProjectMode(null);
            toast.info("Exited project");
            onChanged();
          }}>
            Exit project
          </Button>
        </div>
      </div>

      <div>
        <div className="text-xs font-medium mb-2">Members ({members.length})</div>
        <div className="space-y-1">
          {members.map(m => (
            <div key={m.user_id} className="flex items-center justify-between text-sm gap-3 p-2 border rounded">
              <div className="min-w-0">
                <div className="truncate font-mono text-[11px]">{m.user_id}</div>
                <div className="text-xs text-muted-foreground">Joined {new Date(m.joined_at).toLocaleDateString()}</div>
              </div>
              {isLead && m.user_id !== project.owner_user_id ? (
                <Select value={m.role} onValueChange={(v) => handleRoleChange(m.user_id, v as ProjectRole)}>
                  <SelectTrigger className="w-32 h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="lead">Lead</SelectItem>
                    <SelectItem value="reviewer">Reviewer</SelectItem>
                    <SelectItem value="adjudicator">Adjudicator</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                  </SelectContent>
                </Select>
              ) : (
                <Badge variant="outline" className="text-[10px]">{m.role}</Badge>
              )}
            </div>
          ))}
        </div>
      </div>

      {isLead && !project.locked_at && (
        <div className="space-y-2 pt-2 border-t">
          <div className="text-xs font-medium">Invite reviewers</div>
          <div className="flex gap-2">
            <Select value={inviteRole} onValueChange={(v) => setInviteRole(v as ProjectRole)}>
              <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="reviewer">Reviewer</SelectItem>
                <SelectItem value="adjudicator">Adjudicator</SelectItem>
                <SelectItem value="viewer">Viewer</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={handleInvite} disabled={busy}><LinkIcon className="size-4 mr-1" />Create invite link</Button>
          </div>
          {lastInvite && (
            <div className="p-2 border rounded bg-muted/30 text-xs space-y-1">
              <div className="flex items-center gap-2">
                <ShieldCheck className="size-3" />
                <span>One-time invite link ({lastInvite.role}):</span>
              </div>
              <div className="flex gap-2">
                <Input value={inviteUrl} readOnly className="font-mono text-[11px]" />
                <Button size="sm" variant="outline" onClick={() => {
                  navigator.clipboard.writeText(inviteUrl);
                  toast.success("Invite link copied");
                }}><Copy className="size-3" /></Button>
              </div>
            </div>
          )}
        </div>
      )}

      {isLead && !project.locked_at && (
        <div className="space-y-2 pt-2 border-t">
          <div className="text-xs font-medium">Distribute papers across reviewers</div>
          <div className="text-[11px] text-muted-foreground">
            Re-runs whenever the member list changes. Defaults to full overlap (every reviewer screens every paper); switch to <em>split</em> to share the load round-robin.
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Select value={distributeStrategy} onValueChange={(v) => setDistributeStrategy(v as AssignmentStrategy)}>
              <SelectTrigger className="w-56"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="full_overlap">Full overlap (Cochrane standard)</SelectItem>
                <SelectItem value="split">Round-robin split</SelectItem>
              </SelectContent>
            </Select>
            {distributeStrategy === "split" && (
              <div className="flex items-center gap-1">
                <Label className="text-[11px] whitespace-nowrap">reviewers/paper</Label>
                <Input
                  type="number"
                  min={1}
                  max={Math.max(1, members.length)}
                  value={reviewersPerPaper}
                  onChange={(e) => setReviewersPerPaper(Math.max(1, parseInt(e.target.value || "1", 10)))}
                  className="w-16 h-9"
                />
              </div>
            )}
            <Button onClick={handleDistribute} disabled={busy} size="sm">
              {busy ? "Distributing…" : "Distribute"}
            </Button>
          </div>
        </div>
      )}

      {!isLead && (
        <Alert>
          <AlertTriangle className="size-4 inline mr-1" />
          <AlertDescription>
            Only the project lead can invite new reviewers, change roles, or lock the project.
          </AlertDescription>
        </Alert>
      )}
    </Card>
  );
}

function ModeBadge({ mode }: { mode: ScreeningMode }) {
  const label = mode === "single" ? "Single" : mode === "dual" ? "Dual" : "Dual blinded";
  const cls = mode === "dual_blinded" ? "bg-emerald-50 text-emerald-700 border-emerald-200"
            : mode === "dual" ? "bg-blue-50 text-blue-700 border-blue-200"
            : "bg-slate-50 text-slate-700 border-slate-200";
  return <Badge variant="outline" className={`text-[10px] ${cls}`}>{label}</Badge>;
}
