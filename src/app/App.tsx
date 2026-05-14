import { Component, ReactNode } from "react";
import { Sidebar } from "./components/Sidebar";
import { PrismaFlow } from "./components/PrismaFlow";
import { Toaster } from "./components/ui/sonner";
import { StoreProvider, useStore } from "./lib/store";
import { AuthProvider } from "./lib/auth";
import { UserMenu } from "./components/UserMenu";
import { HomePage } from "./pages/HomePage";
import { SimulationPage } from "./pages/SimulationPage";
import { QualityPage } from "./pages/QualityPage";
import { AbstractPage } from "./pages/AbstractPage";
import { FullTextPage } from "./pages/FullTextPage";
import { SnowballPage } from "./pages/SnowballPage";
import { ExtractionPage } from "./pages/ExtractionPage";
import { AcquisitionPage } from "./pages/AcquisitionPage";
import { TextExtractionPage } from "./pages/TextExtractionPage";
import { MetaAnalysisPage } from "./pages/MetaAnalysisPage";
import { FlaskConical, Home, BarChart3, FileSearch, Network, Table2, GitBranch, ShieldCheck, FileDown, ScanText, Sigma } from "lucide-react";

const PAGE_META: Record<string, { title: string; subtitle: string; icon: any }> = {
  home: { title: "Research Strategy", subtitle: "PICO-driven question framing and search design", icon: Home },
  simulation: { title: "Search Simulation", subtitle: "Per-database query tuning with agentic optimization", icon: BarChart3 },
  quality: { title: "Quality Assessment", subtitle: "Deduplicate and surface issues in acquired papers", icon: ShieldCheck },
  abstract: { title: "Abstract Screening", subtitle: "Multi-agent title and abstract review", icon: FileSearch },
  acquisition: { title: "Full-Text Acquisition", subtitle: "Collect the full text for each included paper", icon: FileDown },
  fulltext: { title: "Full-Text Evidence", subtitle: "Per-criterion full-text evaluation", icon: FlaskConical },
  snowball: { title: "Citation Snowballing", subtitle: "Backward and forward citation discovery", icon: Network },
  extraction: { title: "Table Extraction", subtitle: "Pull tables from included papers", icon: Table2 },
  textextraction: { title: "Text Extraction", subtitle: "Ask questions in natural language and pull values from the text", icon: ScanText },
  prisma: { title: "PRISMA Flow", subtitle: "Reporting diagram for the review", icon: GitBranch },
  meta: { title: "Meta-analysis", subtitle: "Automated effect-size extraction and pooling (fixed + random effects)", icon: Sigma },
};

function Shell() {
  const s = useStore();
  const meta = PAGE_META[s.page];
  const Icon = meta.icon;
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Toaster richColors position="top-right" />
      <Sidebar />
      <main className="flex-1 overflow-x-hidden">
        <header className="border-b bg-card/50 backdrop-blur sticky top-0 z-20 px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center gap-3">
            <Icon className="size-6 text-primary" />
            <div className="flex-1">
              <h1>{meta.title}</h1>
              <p className="text-sm text-muted-foreground">{meta.subtitle}</p>
            </div>
            <UserMenu />
          </div>
        </header>
        <div className="max-w-6xl mx-auto p-6">
          {s.page === "home" && <HomePage />}
          {s.page === "simulation" && <SimulationPage />}
          {s.page === "quality" && <QualityPage />}
          {s.page === "abstract" && <AbstractPage />}
          {s.page === "acquisition" && <AcquisitionPage />}
          {s.page === "fulltext" && <FullTextPage />}
          {s.page === "snowball" && <SnowballPage />}
          {s.page === "extraction" && <ExtractionPage />}
          {s.page === "textextraction" && <TextExtractionPage />}
          {s.page === "prisma" && <PrismaFlow counts={s.prisma} />}
          {s.page === "meta" && <MetaAnalysisPage />}
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
          <div className="max-w-md space-y-2">
            <h2>Something went wrong</h2>
            <pre className="text-xs bg-muted p-3 rounded whitespace-pre-wrap">{this.state.error.message}</pre>
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
      <AuthProvider>
        <StoreProvider>
          <Shell />
        </StoreProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
}
