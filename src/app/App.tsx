import { Component, ReactNode, Suspense, lazy, useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { PrismaFlow } from "./components/PrismaFlow";
import { PrismaChecklist } from "./components/PrismaChecklist";
import { InterraterReliability } from "./components/InterraterReliability";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { bucketAbstractExclusions, bucketFullTextExclusions } from "./lib/exclusionBucketing";
import { Toaster } from "./components/ui/sonner";
import { StoreProvider, useStore } from "./lib/store";
import { AuthProvider } from "./lib/auth";
import { BackendReadyProvider, useEngineStatus } from "./lib/backendReady";
import { UserMenu } from "./components/UserMenu";
// QualityPage stays static because RiskOfBiasSummary (used inline on the
// Diagramming page) is a named export from the same module. The rest are
// code-split so the initial bundle is small and each stage loads on demand.
import { QualityPage, RiskOfBiasSummary } from "./pages/QualityPage";
import { Skeleton } from "./components/ui/skeleton";
const HomePage = lazy(() => import("./pages/HomePage").then(m => ({ default: m.HomePage })));
const SimulationPage = lazy(() => import("./pages/SimulationPage").then(m => ({ default: m.SimulationPage })));
const AbstractPage = lazy(() => import("./pages/AbstractPage").then(m => ({ default: m.AbstractPage })));
const FullTextPage = lazy(() => import("./pages/FullTextPage").then(m => ({ default: m.FullTextPage })));
const SnowballPage = lazy(() => import("./pages/SnowballPage").then(m => ({ default: m.SnowballPage })));
const ExtractionPage = lazy(() => import("./pages/ExtractionPage").then(m => ({ default: m.ExtractionPage })));
const AcquisitionPage = lazy(() => import("./pages/AcquisitionPage").then(m => ({ default: m.AcquisitionPage })));
const TextExtractionPage = lazy(() => import("./pages/TextExtractionPage").then(m => ({ default: m.TextExtractionPage })));
const MetaAnalysisPage = lazy(() => import("./pages/MetaAnalysisPage").then(m => ({ default: m.MetaAnalysisPage })));
const ProjectsPage = lazy(() => import("./pages/ProjectsPage").then(m => ({ default: m.ProjectsPage })));
const WritingPage = lazy(() => import("./pages/WritingPage").then(m => ({ default: m.WritingPage })));
import { CommandPalette } from "./components/CommandPalette";
import { FlaskConical, Home, BarChart3, FileSearch, Network, Table2, GitBranch, ShieldCheck, FileDown, ScanText, Sigma, Users, PenLine, SlidersHorizontal, Loader2, Check, RotateCcw } from "lucide-react";

const PAGE_META: Record<string, { title: string; subtitle: string; icon: any }> = {
  home: { title: "Research Strategy", subtitle: "PICO-driven question framing and search design", icon: Home },
  simulation: { title: "Search Planning", subtitle: "Per-database query tuning with agentic optimization", icon: BarChart3 },
  quality: { title: "Quality Assessment", subtitle: "Deduplicate and surface issues in acquired papers", icon: ShieldCheck },
  abstract: { title: "Abstract Screening", subtitle: "Multi-agent title and abstract review", icon: FileSearch },
  acquisition: { title: "Full-Text Acquisition", subtitle: "Collect the full text for each included paper", icon: FileDown },
  fulltext: { title: "Full-Text Evidence", subtitle: "Per-criterion full-text evaluation", icon: FlaskConical },
  snowball: { title: "Citation Snowballing", subtitle: "Backward and forward citation discovery", icon: Network },
  extraction: { title: "Table Extraction", subtitle: "Pull tables from included papers", icon: Table2 },
  textextraction: { title: "Text Extraction", subtitle: "Ask questions in natural language and pull values from the text", icon: ScanText },
  prisma: { title: "Diagramming", subtitle: "PRISMA flow, risk-of-bias heatmap, and inter-rater reliability for the review", icon: GitBranch },
  meta: { title: "Meta-analysis", subtitle: "Automated effect-size extraction and pooling (fixed + random effects)", icon: Sigma },
  projects: { title: "Projects", subtitle: "Multi-reviewer projects with blinded screening and adjudication", icon: Users },
  writing: { title: "Writing Assistant", subtitle: "Citation export, BibTeX/RIS, and AI-generated methods summary", icon: PenLine },
};

function Shell() {
  const s = useStore();
  const engine = useEngineStatus();
  const meta = PAGE_META[s.page];
  const Icon = meta.icon;
  // If we landed on a /?invite=TOKEN URL, route to the Projects page so the
  // user sees the accept-invite banner. ProjectsPage owns the actual accept
  // flow (token preview + accept button + URL cleanup).
  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.has("invite") && s.page !== "projects") {
      s.setPage("projects");
    }
    // We only want this to run once on mount; the projects page handles
    // subsequent state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Toaster richColors position="top-right" />
      <CommandPalette />
      <Sidebar />
      <main className="flex-1 overflow-x-clip">
        <header className="border-b bg-card/50 backdrop-blur sticky top-0 z-20 px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center gap-3">
            <Icon className="size-6 text-primary" />
            <div className="flex-1">
              <h1>{meta.title}</h1>
              <p className="text-sm text-muted-foreground">{meta.subtitle}</p>
            </div>
            {s.page === "home" && s.history.length > 0 && (
              <button
                onClick={() => s.setReviewOpen(!s.reviewOpen)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-xs font-medium transition-colors ${s.reviewOpen ? "bg-primary text-primary-foreground border-primary" : "bg-primary/10 border-primary/30 text-foreground hover:bg-primary/20"}`}
                title="Edit PICO, criteria & search string"
              >
                <SlidersHorizontal className="size-3.5" />Strategy Review
              </button>
            )}
            {!engine.done && (
              <span
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-400 text-xs font-medium"
                title="The local engine (backend + AI model) is still starting. The app is usable now; screening and search unlock once it's ready."
              >
                <Loader2 className="size-3.5 animate-spin" />
                {engine.message || "Starting engine…"}
              </span>
            )}
            {s.currentProjectId && (
              <button
                onClick={() => s.setPage("projects")}
                className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-md bg-primary/10 border border-primary/30 text-xs hover:bg-primary/20 transition-colors"
                title="Open project settings"
              >
                <Users className="size-3.5 text-primary" />
                <span className="font-medium truncate max-w-[200px]">{s.currentProjectName || "Project"}</span>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground">{s.currentProjectMode}</span>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground">{s.currentProjectRole}</span>
              </button>
            )}
            {s.lastSavedAt && (
              <span
                className="hidden sm:inline-flex items-center gap-1 text-[11px] text-muted-foreground"
                title={`Saved locally at ${new Date(s.lastSavedAt).toLocaleTimeString()}`}
              >
                <Check className="size-3 text-emerald-500" />Saved
              </span>
            )}
            <UserMenu />
          </div>
        </header>
        <div key={s.page} className="max-w-6xl mx-auto p-6 animate-in fade-in duration-200 motion-reduce:animate-none">
          <Suspense fallback={<div className="space-y-3"><Skeleton className="h-8 w-64" /><Skeleton className="h-32 w-full" /><Skeleton className="h-32 w-full" /></div>}>
          {s.page === "home" && <HomePage />}
          {s.page === "simulation" && <SimulationPage />}
          {s.page === "projects" && <ProjectsPage />}
          {s.page === "quality" && <QualityPage />}
          {s.page === "abstract" && <AbstractPage />}
          {s.page === "acquisition" && <AcquisitionPage />}
          {s.page === "fulltext" && <FullTextPage />}
          {s.page === "snowball" && <SnowballPage />}
          {s.page === "extraction" && <ExtractionPage />}
          {s.page === "textextraction" && <TextExtractionPage />}
          {s.page === "prisma" && (() => {
            // Re-derive every funnel count from the live store so the diagram
            // is always self-consistent and shows where each paper left.
            // Sources of truth (in order):
            //   - rawPapers       → records identified
            //   - uniquePapers    → records after duplicates removed
            //   - rerankResults   → records after the LEADS relevance pre-filter
            //   - excludedByQuality → records dropped at the quality screen
            //   - results         → records that actually entered T/A screening
            //   - fullTextResults → records assessed at full-text
            //   - {abstract,fullText}Overrides → reviewer "Keep/Exclude" patches
            const identified = s.rawPapers?.length ?? s.prisma.identified;
            const sourceCounts: Record<string, number> = {};
            (s.rawPapers || []).forEach(p => { sourceCounts[p.source] = (sourceCounts[p.source] || 0) + 1; });

            const afterDuplicates = s.uniquePapers?.length ?? Math.max(0, identified - s.prisma.duplicates_removed);
            const duplicatesRemoved = Math.max(0, identified - afterDuplicates);

            const rerank = s.rerankResults;
            const rerankDropped = rerank ? Math.max(0, rerank.total_scored - rerank.total_kept) : 0;
            const afterRerank = rerank ? rerank.total_kept : afterDuplicates;
            const rerankFloor = rerank
              ? (typeof rerank.effective_floor === "number" ? rerank.effective_floor : rerank.threshold)
              : undefined;

            const qualityExcluded = s.excludedByQuality.size;
            const afterQuality = Math.max(0, afterRerank - qualityExcluded);

            const exclusionBd = s.results
              ? bucketAbstractExclusions(s.results, s.abstractOverrides, s.inclusion, s.exclusion)
              : s.prisma.exclusion_breakdown;
            const screenedCount = s.results?.length ?? s.prisma.screened;
            const excludedAtScreening = Object.values(exclusionBd).reduce((a, b) => a + b, 0);

            const ftExclusionBd = s.fullTextResults
              ? bucketFullTextExclusions(s.fullTextResults, s.fullTextOverrides, s.inclusion, s.exclusion)
              : s.prisma.ft_exclusion_breakdown;

            const finalCount = s.fullTextResults
              ? s.fullTextResults.filter(r => (s.fullTextOverrides[r.paper_id] ?? r.Decision) === "Include").length
              : (s.results
                  ? s.results.filter(r => (s.abstractOverrides[r.paper_id] ?? r.Decision) === "INCLUDE").length
                  : (s.prisma.included_final ?? 0));

            // Heatmap only shows papers that survived through full-text inclusion
            // (effective decision = Include, honouring reviewer overrides). If
            // full-text screening hasn't been run yet, fall back to abstract-
            // included papers so the tab is still useful after the screening
            // stage. If neither stage has produced inclusions, the tab shows an
            // empty-state hint.
            const includedFullTextIds = new Set(
              (s.fullTextResults || [])
                .filter(r => (s.fullTextOverrides[r.paper_id] ?? r.Decision) === "Include")
                .map(r => r.paper_id),
            );
            const includedAbstractIds = new Set(
              (s.results || [])
                .filter(r => (s.abstractOverrides[r.paper_id] ?? r.Decision) === "INCLUDE")
                .map(r => r.paper_id),
            );
            const heatmapStage: "fulltext" | "abstract" | "none" =
              includedFullTextIds.size > 0 ? "fulltext"
              : includedAbstractIds.size > 0 ? "abstract"
              : "none";
            const heatmapIds = heatmapStage === "fulltext" ? includedFullTextIds : includedAbstractIds;
            const heatmapReports = (s.qualityReports || []).filter(r => heatmapIds.has(r.paper_id));
            const hasHeatmap = heatmapReports.length > 0;

            return (
              <Tabs defaultValue="prisma" className="space-y-4">
                <TabsList>
                  <TabsTrigger value="prisma">PRISMA flow</TabsTrigger>
                  <TabsTrigger value="heatmap" disabled={!hasHeatmap}>
                    Risk-of-bias heatmap
                  </TabsTrigger>
                  <TabsTrigger value="irr" disabled={!s.currentProjectId}>
                    Inter-rater reliability
                  </TabsTrigger>
                  <TabsTrigger value="checklist">PRISMA checklist</TabsTrigger>
                </TabsList>
                <TabsContent value="prisma">
                  <PrismaFlow
                    counts={{
                      identified,
                      source_counts: sourceCounts,
                      duplicates_removed: duplicatesRemoved,
                      after_duplicates: afterDuplicates,
                      rerank_dropped: rerankDropped,
                      after_rerank: afterRerank,
                      rerank_floor: rerankFloor,
                      quality_excluded: qualityExcluded,
                      after_quality: afterQuality,
                      screened: screenedCount,
                      excluded_total: excludedAtScreening,
                      exclusion_breakdown: exclusionBd,
                      ft_exclusion_breakdown: ftExclusionBd,
                      included_final: finalCount,
                    }}
                    abstractResults={s.results}
                    abstractOverrides={s.abstractOverrides}
                    fullTextResults={s.fullTextResults}
                    fullTextOverrides={s.fullTextOverrides}
                    inclusion={s.inclusion}
                    exclusion={s.exclusion}
                  />
                </TabsContent>
                <TabsContent value="heatmap" className="space-y-2">
                  {hasHeatmap ? (
                    <>
                      <div className="text-xs text-muted-foreground">
                        Showing risk-of-bias appraisal for {heatmapReports.length}{" "}
                        {heatmapStage === "fulltext" ? "full-text-included" : "abstract-included"} paper{heatmapReports.length === 1 ? "" : "s"}.
                      </div>
                      <RiskOfBiasSummary
                        reports={heatmapReports}
                        overrides={s.qualityOverrides}
                        onOverride={(o) => s.addQualityOverride(o)}
                      />
                    </>
                  ) : (
                    <div className="text-sm text-muted-foreground p-4 border rounded-md">
                      No included papers yet. Run Quality Assessment plus screening (abstract or full-text) to populate the heatmap.
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="irr" className="space-y-2">
                  {s.currentProjectId ? (
                    <InterraterReliability projectId={s.currentProjectId} />
                  ) : (
                    <div className="text-sm text-muted-foreground p-4 border rounded-md">
                      Inter-rater reliability is only meaningful inside a multi-reviewer project. Open a project from the <strong>Projects</strong> tab to populate this view.
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="checklist">
                  <PrismaChecklist />
                </TabsContent>
              </Tabs>
            );
          })()}
          {s.page === "meta" && <MetaAnalysisPage />}
          {s.page === "writing" && <WritingPage />}
          </Suspense>
        </div>
      </main>
    </div>
  );
}

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: any) { console.error("App error:", error, info); }
  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-background text-foreground">
          <div className="max-w-md w-full text-center space-y-3 rounded-xl border bg-card p-8 shadow-sm">
            <div className="grid place-items-center size-12 rounded-full bg-destructive/10 text-destructive mx-auto">
              <RotateCcw className="size-6" />
            </div>
            <h2 className="font-medium">Something went wrong</h2>
            <p className="text-sm text-muted-foreground">Your work is autosaved. Reloading usually fixes this and returns you where you left off.</p>
            <pre className="text-[11px] text-left bg-muted p-3 rounded whitespace-pre-wrap max-h-40 overflow-auto">{this.state.error.message}</pre>
            <button onClick={() => window.location.reload()} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90">
              <RotateCcw className="size-4" />Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <BackendReadyProvider>
        <AuthProvider>
          <StoreProvider>
            <Shell />
          </StoreProvider>
        </AuthProvider>
      </BackendReadyProvider>
    </ErrorBoundary>
  );
}
