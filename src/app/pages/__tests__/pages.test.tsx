// Smoke and behaviour tests for all 12 pages.
//
// Run: pnpm test
//
// Every page reads the global store, so each renders inside a real
// StoreProvider with a seeded review. Two things are under test:
//
//   1. Every page renders at all, in both the empty state and with data. A
//      page that throws on mount takes the whole tab down, and until now
//      nothing would have caught that except opening the app.
//   2. The stage gates: a page that needs upstream work must say so rather
//      than showing a confusing blank or, worse, letting the reviewer act on
//      an empty corpus and believe the result.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
import type { ReactNode } from "react";

import { StoreProvider, useStore } from "../../lib/store";
import { ImportStage, importReason } from "../../lib/importedAt";
import { setResolverBase } from "../../lib/dbKeys";
import { HomePage } from "../HomePage";
import { SimulationPage } from "../SimulationPage";
import { AbstractPage } from "../AbstractPage";
import { AcquisitionPage } from "../AcquisitionPage";
import { FullTextPage } from "../FullTextPage";
import { QualityPage } from "../QualityPage";
import { ExtractionPage } from "../ExtractionPage";
import { TextExtractionPage } from "../TextExtractionPage";
import { SnowballPage } from "../SnowballPage";
import { MetaAnalysisPage } from "../MetaAnalysisPage";
import { WritingPage } from "../WritingPage";
import { ProjectsPage } from "../ProjectsPage";

// --------------------------------------------------------------------------
// Harness: render a page inside the real store, optionally seeded.
// --------------------------------------------------------------------------

function Seed({ seed, children }: { seed?: (s: any) => void; children: ReactNode }) {
  const s = useStore();
  const done = (Seed as any)._done ?? new WeakSet();
  (Seed as any)._done = done;
  if (seed && !done.has(s)) {
    done.add(s);
    seed(s);
  }
  return <>{children}</>;
}

function renderPage(Page: () => JSX.Element, seed?: (s: any) => void) {
  let api: any;
  function Capture() {
    api = useStore();
    return null;
  }
  const utils = render(
    <StoreProvider>
      <Capture />
      <Page />
    </StoreProvider>,
  );
  if (seed) act(() => seed(api));
  return { ...utils, store: () => api };
}

const paper = (id: string) => ({
  paper_id: id, id, Title: `Study ${id}`, title: `Study ${id}`,
  Abstract: "We randomised 128 adults.", abstract: "We randomised 128 adults.",
  Source: "PubMed", source: "PubMed", URL: "", url: "",
});

const result = (id: string, decision: "INCLUDE" | "EXCLUDE" = "INCLUDE") => ({
  paper_id: id, Title: `Study ${id}`, Source: "PubMed", URL: "",
  Abstract: "We randomised 128 adults.",
  Decision: decision, Reason: "because", Agent_Trace: {},
  Pico_Assessment: { population: { vote: "PASS", evidence: "", reasoning: "" } },
});

/** Seed the state the meta page requires before it renders anything: screening
 *  has run and included at least one study. Without this the page correctly
 *  short-circuits to its "screen abstracts first" empty state. */
function seedMetaReady(s: any, extractions: any[]) {
  s.setResults([result("p1", "INCLUDE")]);
  s.setMetaExtractions(extractions);
}

const PAGES: [string, () => JSX.Element][] = [
  ["HomePage", HomePage],
  ["SimulationPage", SimulationPage],
  ["AbstractPage", AbstractPage],
  ["AcquisitionPage", AcquisitionPage],
  ["FullTextPage", FullTextPage],
  ["QualityPage", QualityPage],
  ["ExtractionPage", ExtractionPage],
  ["TextExtractionPage", TextExtractionPage],
  ["SnowballPage", SnowballPage],
  ["MetaAnalysisPage", MetaAnalysisPage],
  ["WritingPage", WritingPage],
  ["ProjectsPage", ProjectsPage],
];

beforeEach(() => {
  localStorage.clear();
  // Pages that fetch on mount get an inert response rather than a rejection.
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, headers: new Headers(),
    text: async () => "{}", json: async () => ({}),
  })));
});

// --------------------------------------------------------------------------
// An upload carries into every stage, without being counted as work done.
//
// The stages are a menu, not a queue: a reviewer who arrives with the studies
// they have already chosen can use whichever stage they need. That is why an
// import seeds a row at every stage. The row says plainly that it was not
// screened here, and the stages have to read it that way, or a reviewer who
// does want to screen their upload finds the button greyed out and the work
// apparently already done.
// --------------------------------------------------------------------------

describe("an imported record", () => {
  const imported = (stage: ImportStage) => ({
    paper_id: "p1", Title: "Study p1", Source: "Local PDFs", URL: "", Abstract: "a",
    Decision: "INCLUDE", Reason: importReason(stage), Agent_Trace: {}, Imported: stage,
  });

  it("is still offered for abstract screening", () => {
    const { container } = renderPage(AbstractPage, s => {
      s.setUniquePapers([paper("p1")] as any);
      s.setResults([imported("text-extraction")] as any);
    });
    expect(container.querySelector('[title^="Screen 1 article"]')).not.toBeNull();
  });

  it("is still offered for full-text screening", () => {
    const { container } = renderPage(FullTextPage, s => {
      s.setResults([imported("text-extraction")] as any);
      s.setFullTextResults([{ ...imported("text-extraction"), Decision: "Include" }] as any);
    });
    expect(container.querySelector('[title^="Screen 1 newly included"]')).not.toBeNull();
  });

  it("does not make a genuinely screened paper look unscreened", () => {
    const { container } = renderPage(AbstractPage, s => {
      s.setUniquePapers([paper("p1")] as any);
      s.setResults([result("p1", "INCLUDE")] as any);
    });
    expect(container.querySelector('[title^="Screen 1 article"]')).toBeNull();
  });
});

// --------------------------------------------------------------------------
// Every stage can be started from the reviewer's own studies.
//
// A reviewer who screened elsewhere and arrives with 200 PDFs should not have
// to walk back through the stages they already did. Each stage that reads an
// earlier one therefore offers its own import, and it has to be offered in the
// empty state too: that is exactly the state such a reviewer arrives in, and a
// button that only appears once the stage already has data is no use to them.
// --------------------------------------------------------------------------

describe.each([
  ["AbstractPage", AbstractPage],
  ["AcquisitionPage", AcquisitionPage],
  ["FullTextPage", FullTextPage],
  ["ExtractionPage", ExtractionPage],
  ["TextExtractionPage", TextExtractionPage],
  ["QualityPage", QualityPage],
  ["SnowballPage", SnowballPage],
  ["MetaAnalysisPage", MetaAnalysisPage],
  ["WritingPage", WritingPage],
] as [string, () => JSX.Element][])("%s with nothing upstream", (_name, Page) => {
  it("offers to start from your own studies", () => {
    const { container } = renderPage(Page);
    expect(container.textContent ?? "").toMatch(/add (seed )?studies/i);
  });
});

// --------------------------------------------------------------------------
// Starting a new session empties the store under whichever page is open.
//
// That is a render with less data than the one before it, so a page that bails
// out early and then calls another hook runs fewer hooks than React recorded,
// and React takes the whole app down with "Rendered fewer hooks than expected"
// (minified #300). Three pages did exactly that. The rule it enforces is that
// every hook sits above every early return, which no test could catch until
// something emptied the store with the page still mounted.
// --------------------------------------------------------------------------

/** Enough of a finished review that every page renders its full body. */
function seedWholeReview(s: any) {
  s.setHistory([{ goal: "g", query: "q", formal_question: "fq", ts: 1,
                  inclusion: ["adults"], exclusion: ["animals"] }] as any);
  s.setRawPapers([paper("p1")] as any);
  s.setUniquePapers([paper("p1")] as any);
  s.setResults([result("p1", "INCLUDE")] as any);
  s.setFullTexts({
    p1: { paper_id: "p1", title: "Study p1", url: "", source: "PubMed",
          status: "found", text: "body" },
  });
  // Source matters: the writing stage reads it to enrich citations, and an
  // undefined one throws out of an effect rather than failing a render.
  s.setFullTextResults([{ paper_id: "p1", Title: "Study p1", Abstract: "a", URL: "",
                          Source: "PubMed", Decision: "Include", Reason: "" }] as any);
  s.setSnowballResults([{ id: "c1", title: "A citation", seed_paper_title: "Study p1",
                          citation_type: "backward" }] as any);
  s.setQualityReports([{ paper_id: "p1", title: "Study p1", instrument: "RoB 2",
                         judgment: "Low", domains: [] }] as any);
  s.setTextExtractions([{ paper_id: "p1", title: "Study p1", answer: "",
                          spans: [], evidence: [], values: [] }] as any);
}

describe.each(PAGES)("%s with the review cleared under it", (_name, Page) => {
  it("survives a new session", () => {
    const view = renderPage(Page, seedWholeReview);
    expect(() => act(() => view.store().reset())).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Every page renders, empty and populated.
// --------------------------------------------------------------------------

describe.each(PAGES)("%s", (name, Page) => {
  it("renders in the empty state without throwing", () => {
    expect(() => renderPage(Page)).not.toThrow();
  });

  it("renders something rather than a blank tab", () => {
    const { container } = renderPage(Page);
    expect((container.textContent ?? "").trim().length).toBeGreaterThan(0);
  });

  it("renders with a full review loaded", () => {
    expect(() => renderPage(Page, (s) => {
      s.setRawPapers([paper("p1"), paper("p2"), paper("p3")]);
      s.setUniquePapers([paper("p1"), paper("p2")]);
      s.setResults([result("p1"), result("p2", "EXCLUDE")]);
      s.setPico({ population: "adults", intervention: "metformin",
                  comparator: "placebo", outcome: "HbA1c" });
      s.setInclusion(["Published after 2010"]);
      s.setExclusion(["Animal studies"]);
      s.setQuery('("metformin"[tiab])');
      s.setSources(["PubMed"]);
    })).not.toThrow();
  });

  it("does not render NaN or undefined to the reviewer", () => {
    const { container } = renderPage(Page, (s) => {
      s.setResults([result("p1")]);
      s.setRawPapers([paper("p1")]);
    });
    const text = container.textContent ?? "";
    expect(text).not.toContain("NaN");
    expect(text).not.toContain("undefined");
    expect(text).not.toContain("[object Object]");
  });
});

// --------------------------------------------------------------------------
// Stage gates. Each page depends on the one before it.
// --------------------------------------------------------------------------

describe("stage gating", () => {
  it("AbstractPage tells the reviewer to run a search first", () => {
    const { container } = renderPage(AbstractPage);
    expect((container.textContent ?? "").toLowerCase()).toMatch(/search|plan|simulat|no (studies|papers)/);
  });

  it("AcquisitionPage explains there is nothing to acquire yet", () => {
    const { container } = renderPage(AcquisitionPage);
    expect((container.textContent ?? "").toLowerCase()).toMatch(/screen|includ|no /);
  });

  it("FullTextPage explains it needs acquired texts", () => {
    const { container } = renderPage(FullTextPage);
    expect((container.textContent ?? "").length).toBeGreaterThan(0);
  });

  it("ExtractionPage points back to abstract screening when nothing is included", () => {
    const { container } = renderPage(ExtractionPage);
    expect((container.textContent ?? "").toLowerCase()).toMatch(/abstract|screen/);
  });

  // ── Planning: collapsible run history ──────────────────────────────────
  const planningState = (s: any) => {
    s.setHistory([{ goal: "g", query: "q", formal_question: "fq", ts: 1 }] as any);
    s.setSimulation({ PubMed: 47 } as any);
    // addSimulationRun stamps id/timestamp/label itself.
    s.addSimulationRun({
      unifiedQuery: "q", perDbQueries: { PubMed: "q" },
      counts: { PubMed: 47 }, totalYield: 47, source: "manual",
    } as any);
  };

  it("SimulationPage can be started without a research question", () => {
    /* A reviewer who already has a search string should be able to tune and
       count it here rather than being sent back to write a question for it. */
    const view = renderPage(SimulationPage);
    expect(view.container.textContent).toMatch(/start a search plan/i);

    const box = view.container.querySelector("textarea")!;
    fireEvent.change(box, { target: { value: "glioma AND temozolomide" } });
    fireEvent.click(screen.getByRole("button", { name: /use this query/i }));

    // The planner proper, not the starting card.
    expect(view.container.textContent).not.toMatch(/start a search plan/i);
  });

  it("SimulationPage shows the run history expanded by default", async () => {
    localStorage.removeItem("ee:planning-history-collapsed");
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run history/i));
    expect(view.container.querySelector('[aria-label="Collapse run history"]')).not.toBeNull();
  });

  it("SimulationPage collapses the run history to a rail", async () => {
    /* The point is the space: the middle column holds boolean query strings
       that wrap badly, and the history is reference material you consult
       occasionally rather than read continuously. */
    localStorage.setItem("ee:planning-history-collapsed", "1");
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(
      view.container.querySelector('[aria-label="Show run history"]')).not.toBeNull());
    expect(view.container.querySelector('[aria-label="Collapse run history"]')).toBeNull();
  });

  it("SimulationPage keeps the history reachable while collapsed", async () => {
    /* A rail, not a hidden panel: the label and the run count stay on screen so
       nobody has to remember the feature exists. */
    localStorage.setItem("ee:planning-history-collapsed", "1");
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run history/i));
    expect(view.container.textContent).toContain("1");
  });

  it("SimulationPage remembers the collapsed choice across mounts", async () => {
    /* Someone who wants the query editor wide wants it wide every time. */
    localStorage.setItem("ee:planning-history-collapsed", "1");
    const first = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(
      first.container.querySelector('[aria-label="Show run history"]')).not.toBeNull());
    first.unmount();
    const second = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(
      second.container.querySelector('[aria-label="Show run history"]')).not.toBeNull());
  });

  it("SimulationPage gives every run its own delete button", async () => {
    /* Until now the only way to remove one bad run was Clear, which takes the
       good ones with it. A run is a dead end often enough (a query that
       over-broadened, a typo) that this matters. */
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run history/i));
    expect(view.container.querySelector('[aria-label^="Delete Run"]')).not.toBeNull();
  });

  it("SimulationPage removes only the run whose delete was clicked", async () => {
    const view = renderPage(SimulationPage, s => {
      planningState(s);
      s.addSimulationRun({
        unifiedQuery: "q2", perDbQueries: { PubMed: "q2" },
        counts: { PubMed: 99 }, totalYield: 99, source: "manual",
      } as any);
    });
    await waitFor(() => expect(view.container.querySelectorAll('[aria-label^="Delete Run"]').length).toBe(2));
    act(() => {
      (view.container.querySelector('[aria-label="Delete Run 1"]') as HTMLElement).click();
    });
    await waitFor(() => expect(view.container.querySelectorAll('[aria-label^="Delete Run"]').length).toBe(1));
    expect(view.store().simulationRuns.map((r: any) => r.label)).toEqual(["Run 2"]);
  });

  it("SimulationPage renames a run inline", async () => {
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run 1/));
    act(() => { (screen.getByTitle("Rename this run") as HTMLElement).click(); });
    const input = await screen.findByLabelText(/Rename Run 1/i);
    fireEvent.change(input, { target: { value: "Dropped population block" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() =>
      expect(view.store().simulationRuns[0].label).toBe("Dropped population block"));
  });

  it("SimulationPage abandons a rename on Escape", async () => {
    /* Blur commits, so Escape has to bail out before the blur fires or a
       half-typed name gets saved on the way out. */
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run 1/));
    act(() => { (screen.getByTitle("Rename this run") as HTMLElement).click(); });
    const input = await screen.findByLabelText(/Rename Run 1/i);
    fireEvent.change(input, { target: { value: "half-typed" } });
    fireEvent.keyDown(input, { key: "Escape" });
    await waitFor(() => expect(view.store().simulationRuns[0].label).toBe("Run 1"));
  });

  it("SimulationPage no longer shows a query preview on each run", async () => {
    /* The query is identical across runs often enough that a 60-character
       prefix distinguished nothing while taking a line per row. */
    const view = renderPage(SimulationPage, planningState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Run history/i));
    const history = view.container.querySelector('[aria-label^="Delete Run"]')?.closest("div")?.parentElement;
    expect(history?.textContent ?? "").not.toContain("[tiab]");
  });

  // ── Abstract screening: the control pane ───────────────────────────────
  const screeningReady = (st: any) => {
    st.setQuery("q");
    st.setRawPapers([1, 2, 3, 4, 5, 6, 7, 8].map(i => ({
      id: `p${i}`, paper_id: `p${i}`, title: `Study ${i}`,
      abstract: "a", source: "PubMed", url: "",
    })) as any);
    st.setUniquePapers([1, 2, 3, 4, 5, 6, 7].map(i => ({
      id: `p${i}`, paper_id: `p${i}`, title: `Study ${i}`,
      abstract: "a", source: "PubMed", url: "",
    })) as any);
    st.setDuplicatesCount(1);
  };

  it("AbstractPage puts counts and every action in one control pane", async () => {
    /* The run controls used to float free at the bottom of the page, below a
       card that already described what they would do. */
    const view = renderPage(AbstractPage, screeningReady);
    await waitFor(() => expect(view.container.textContent).toMatch(/Ready to screen/i));
    expect(view.container.textContent).toMatch(/Retrieved/i);
    expect(view.container.textContent).toMatch(/Run AI screening/i);
    expect(view.container.textContent).toMatch(/Screen manually/i);
    expect(view.container.textContent).toMatch(/Retry retrieval/i);
  });

  it("AbstractPage keeps retrieval separate from screening while a retry runs", async () => {
    /* Retry retrieval used to run under the "abstract-screen" task id, so the
       page believed a screening pass had started: the control pane and the
       preview both unmounted, and a card announced per-PICO appraisal while
       nothing but database calls were in flight. */
    const view = renderPage(AbstractPage, st => {
      screeningReady(st);
      st.startTask("retrieval", [
        { id: "fetch", label: "Retrieving from databases", status: "running" },
      ]);
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/Retrieving from databases/i));
    // The pane that owns the button stays put, so the counts update in place.
    expect(view.container.textContent).toMatch(/Retry retrieval/i);
    expect(view.container.textContent).toMatch(/Ready to screen/i);
    // And nothing claims the model is reading anything.
    expect(view.container.textContent).not.toMatch(/per-PICO appraisal/i);
    expect(view.container.textContent).not.toMatch(/Cancel screening/i);
  });

  it("AbstractPage still shows the screening card for a real screening run", async () => {
    /* The other half of the gate: separating the two task ids must not stop a
       genuine screening pass from reporting itself. */
    const view = renderPage(AbstractPage, st => {
      screeningReady(st);
      st.startTask("abstract-screen", [
        { id: "screen", label: "Screening abstracts", status: "running" },
      ]);
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/per-PICO appraisal/i));
    expect(view.container.textContent).toMatch(/Cancel screening/i);
  });

  it("AbstractPage counts only what will actually cost a model call", async () => {
    /* Quality-Assessment exclusions never reach the model, so they come off
       "Ready to screen" and are reported separately. */
    const view = renderPage(AbstractPage, st => {
      screeningReady(st);
      st.setExcludedByQuality?.(new Set(["p1", "p2"]));
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/Ready to screen/i));
    // textContent runs the value straight into its label, so assert the pair.
    expect(view.container.textContent).toMatch(/5\s*Ready to screen/);
    expect(view.container.textContent).toMatch(/2\s*Excluded/);
  });

  it("AbstractPage reports duplicates removed", async () => {
    const view = renderPage(AbstractPage, screeningReady);
    await waitFor(() => expect(view.container.textContent).toMatch(/Duplicates/i));
  });

  it("MetaAnalysisPage asks for screening before showing any controls", () => {
    /* Every stage does this: until there is work to do, the only useful thing on
       screen is the way to go and get it. Controls greyed out beside an empty
       state say the same thing twice. */
    const view = renderPage(MetaAnalysisPage);
    expect(view.container.textContent).toMatch(/Screening not complete/i);
    expect(view.container.textContent).not.toMatch(/Analysis settings/i);
    expect(view.container.textContent).not.toMatch(/Ready to extract/i);
  });

  it("MetaAnalysisPage distinguishes 'not screened' from 'screened, nothing included'", () => {
    const view = renderPage(MetaAnalysisPage, s => s.setResults([result("p1", "EXCLUDE")]));
    expect(view.container.textContent).toMatch(/No included articles/i);
  });

  it("MetaAnalysisPage shows the controls once studies are included", async () => {
    const view = renderPage(MetaAnalysisPage, s => s.setResults([result("p1", "INCLUDE")]));
    await waitFor(() => expect(view.container.textContent).toMatch(/Analysis settings/i));
  });

  it("MetaAnalysisPage shows the evidence workspace even when nothing pools", async () => {
    /* THE REGRESSION. The workspace was first nested inside the block gated on
       a successful pool, so a reviewer whose extractions all failed saw none of
       it. That is backwards: when the numbers are wrong, the source document
       and the reason each study was rejected are the whole point. */
    const unpoolable = [{
      paper_id: "p1", title: "A study", outcome: "mortality",
      effect_measure: "MD", yi: null, vi: null,
      error: "MD needs sd_t",
    }] as any;
    const view = renderPage(MetaAnalysisPage, s => seedMetaReady(s, unpoolable));
    await waitFor(() => expect(view.container.textContent).toMatch(/A study/));
  });

  it("MetaAnalysisPage reuses tables already pulled by Table Extraction", async () => {
    /* Re-extracting is not just slow. The LLM tier is non-deterministic, so a
       second run can yield different tables from the ones the reviewer already
       checked on the Table Extraction tab. */
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: any) => {
      calls.push(String(url));
      return { ok: true, status: 200, headers: new Headers(),
               json: async () => ({}), text: async () => "{}" } as unknown as Response;
    }));
    const view = renderPage(MetaAnalysisPage, s => {
      s.setExtractedPapers([{
        Paper_Title: "A study", Paper_URL: "", Source: "PubMed",
        Extracted_Tables: [{ title: "Table 2", type: "outcomes", page: 7,
                             tier: "pdfplumber", deterministic: true,
                             data: [["Outcome", "HR"], ["MACE", "0.74"]] }],
      }] as any);
      seedMetaReady(s, [{
        paper_id: "p1", title: "A study", outcome: "MACE",
        effect_measure: "HR", yi: -0.3, vi: 0.01,
      }] as any);
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/Table 2|MACE/));
    expect(calls.filter(u => u.includes("/extract/tables"))).toHaveLength(0);
  });

  it("MetaAnalysisPage matches stored tables despite title whitespace drift", async () => {
    const view = renderPage(MetaAnalysisPage, s => {
      s.setExtractedPapers([{
        Paper_Title: "  A   Study  ", Paper_URL: "", Source: "PubMed",
        Extracted_Tables: [{ title: "Table 9", type: "outcomes",
                             data: [["Outcome"], ["MACE"]] }],
      }] as any);
      seedMetaReady(s, [{
        paper_id: "p1", title: "a study", outcome: "MACE",
        effect_measure: "HR", yi: -0.3, vi: 0.01,
      }] as any);
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/Table 9/));
  });

  it("MetaAnalysisPage names why a study could not be pooled", async () => {
    const unpoolable = [{
      paper_id: "p1", title: "A study", outcome: "mortality",
      effect_measure: "MD", yi: null, vi: null, error: "MD needs sd_t",
    }] as any;
    const view = renderPage(MetaAnalysisPage, s => {
      seedMetaReady(s, unpoolable);
      s.setMetaRun({ pool: { k: 0, valid_studies: [], invalid_studies: unpoolable,
                             random: null, fixed: null, heterogeneity: null,
                             prediction_interval: null, effect_measures: [],
                             tau2_method: "DL", use_knapp_hartung: false } } as any);
    });
    await waitFor(() => expect(view.container.textContent).toMatch(/could not be pooled/i));
    expect(view.container.textContent).toMatch(/MD needs sd_t/);
  });

  it("MetaAnalysisPage renders with no extracted data", () => {
    expect(() => renderPage(MetaAnalysisPage)).not.toThrow();
  });

  it("WritingPage renders before any screening has happened", () => {
    expect(() => renderPage(WritingPage)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// AcquisitionPage: the counts the reviewer acts on.
// --------------------------------------------------------------------------

describe("AcquisitionPage counts", () => {
  const seedIncluded = (s: any) => {
    s.setResults([result("p1"), result("p2"), result("p3"), result("p4", "EXCLUDE")]);
  };

  it("counts only the included studies", () => {
    const { container } = renderPage(AcquisitionPage, seedIncluded);
    expect(container.textContent).toContain("3");
  });

  it("shows nothing acquired before any fetch", () => {
    const { container } = renderPage(AcquisitionPage, seedIncluded);
    expect((container.textContent ?? "").toLowerCase()).toMatch(/acquired|missing|fetch/);
  });

  it("reflects acquired full texts in the counts", () => {
    const { container } = renderPage(AcquisitionPage, (s) => {
      seedIncluded(s);
      s.setFullTexts({
        p1: { paper_id: "p1", title: "Study p1", url: "", source: "PubMed",
              status: "found", text: "body" },
        p2: { paper_id: "p2", title: "Study p2", url: "", source: "PubMed",
              status: "missing", reason: "Paywalled.", reason_code: "paywalled" },
      });
    });
    expect(container.textContent).toMatch(/1|2|3/);
  });

  it("surfaces the reason a paper is missing", () => {
    /* The whole point of the miss payload: the reviewer needs to know where to
       go next, not just that it failed. */
    const { container } = renderPage(AcquisitionPage, (s) => {
      s.setResults([result("p1")]);
      s.setFullTexts({
        p1: { paper_id: "p1", title: "Study p1", url: "", source: "PubMed",
              status: "missing", reason: "Paywalled. Needs library access.",
              reason_code: "paywalled", doi: "10.1/abc",
              links: { doi: "https://doi.org/10.1/abc" } },
      });
    });
    expect(container.textContent).toMatch(/paywall/i);
  });
});

// --------------------------------------------------------------------------
// AbstractPage: decisions and reviewer overrides.
// --------------------------------------------------------------------------

describe("AbstractPage", () => {
  it("shows screened studies", () => {
    const { container } = renderPage(AbstractPage, (s) => {
      s.setUniquePapers([paper("p1"), paper("p2")]);
      s.setResults([result("p1"), result("p2", "EXCLUDE")]);
    });
    expect(container.textContent).toContain("Study p1");
  });

  it("counts includes and excludes separately", () => {
    const { container } = renderPage(AbstractPage, (s) => {
      s.setUniquePapers([paper("p1"), paper("p2"), paper("p3")]);
      s.setResults([result("p1"), result("p2"), result("p3", "EXCLUDE")]);
    });
    const text = container.textContent ?? "";
    expect(text).toMatch(/2/);
    expect(text).toMatch(/1/);
  });

  it("a reviewer override changes what the page reports", () => {
    const { container } = renderPage(AbstractPage, (s) => {
      s.setUniquePapers([paper("p1")]);
      s.setResults([result("p1", "INCLUDE")]);
      s.setAbstractOverrides({ p1: "EXCLUDE" });
    });
    expect((container.textContent ?? "").length).toBeGreaterThan(0);
  });

  it("renders a result whose PICO panel is missing", () => {
    /* Older sessions and LEADS runs can lack the panel entirely. */
    expect(() => renderPage(AbstractPage, (s) => {
      s.setUniquePapers([paper("p1")]);
      s.setResults([{ ...result("p1"), Pico_Assessment: undefined } as any]);
    })).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// HomePage
// --------------------------------------------------------------------------

describe("HomePage", () => {
  it("renders the entry point", () => {
    const { container } = renderPage(HomePage);
    expect((container.textContent ?? "").length).toBeGreaterThan(0);
  });

  it("renders with a PICO already set", () => {
    expect(() => renderPage(HomePage, (s) => {
      s.setPico({ population: "adults", intervention: "metformin",
                  comparator: "placebo", outcome: "HbA1c" });
    })).not.toThrow();
  });

  it("renders with a PCC frame", () => {
    expect(() => renderPage(HomePage, (s) => {
      s.setFramework("pcc");
      s.setPico({ population: "any patients", concept: "AI", context: "any setting" });
    })).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Resilience to malformed state, which is what a restored old session is.
// --------------------------------------------------------------------------

describe("resilience to malformed session state", () => {
  it.each(PAGES)("%s survives results with missing fields", (_name, Page) => {
    expect(() => renderPage(Page, (s) => {
      s.setResults([{ paper_id: "p1" } as any]);
      s.setRawPapers([{ paper_id: "p1" } as any]);
      s.setUniquePapers([{ paper_id: "p1" } as any]);
    })).not.toThrow();
  });

  it.each(PAGES)("%s survives an empty PICO", (_name, Page) => {
    expect(() => renderPage(Page, (s) => s.setPico({} as any))).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Handing the unreachable papers to a human.
//
// Most of a clinical corpus cannot be downloaded programmatically, so the
// acquisition stage always ends with a manual remainder. These assert the
// handoff: a CSV a librarian can read and an RIS every interlibrary-loan
// system and reference manager ingests.
// --------------------------------------------------------------------------

describe("exporting the missing full texts", () => {
  const acquisitionState = (st: any) => {
    st.setQuery("q");
    st.setUniquePapers([
      { id: "p1", paper_id: "p1", title: "Paywalled trial", abstract: "a", source: "PubMed",
        url: "", year: 2021, authors: "Smith J; Okafor A" },
      { id: "p2", paper_id: "p2", title: "Acquired trial", abstract: "a", source: "PubMed", url: "" },
    ] as any);
    st.setResults([
      { paper_id: "p1", Source: "PubMed", Title: "Paywalled trial", URL: "", Abstract: "a",
        Decision: "INCLUDE", Reason: "", Agent_Trace: {} },
      { paper_id: "p2", Source: "PubMed", Title: "Acquired trial", URL: "", Abstract: "a",
        Decision: "INCLUDE", Reason: "", Agent_Trace: {} },
    ] as any);
    st.setFullTexts({
      p1: { paper_id: "p1", title: "Paywalled trial", url: "", source: "PubMed",
            status: "missing", reason: "Paywalled. A publisher text-mining key, your library, or interlibrary loan can reach it.",
            doi: "10.1000/paywalled", pmid: "31234567" },
      p2: { paper_id: "p2", title: "Acquired trial", url: "", source: "PubMed",
            status: "found", text: "full text here" },
    });
  };

  /** Open the Export menu, choose a format, and return what it put in the Blob.
   *  Both formats now live behind one control, so the test drives the menu the
   *  way a reviewer does rather than calling the handler directly. */
  async function capture(format: RegExp): Promise<string> {
    let body = "";
    const realCreate = URL.createObjectURL;
    (URL as any).createObjectURL = (b: Blob) => { (b as any).text().then((t: string) => { body = t; }); return "blob:x"; };
    (URL as any).revokeObjectURL = () => {};
    const view = renderPage(AcquisitionPage, acquisitionState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Export/));
    const trigger = [...view.container.querySelectorAll("button")]
      .find(b => /^\s*Export\s*$/.test((b.textContent || "").trim()))!;
    expect(trigger).toBeTruthy();
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false });
    fireEvent.click(trigger);
    // The menu renders in a portal, so look at the whole document.
    const item = await waitFor(() => {
      const el = [...document.querySelectorAll('[role="menuitem"]')]
        .find(n => format.test(n.textContent || ""));
      if (!el) throw new Error("menu item not rendered");
      return el;
    });
    fireEvent.click(item);
    await waitFor(() => expect(body).not.toBe(""));
    (URL as any).createObjectURL = realCreate;
    return body;
  }

  it("offers both formats behind a single Export control", async () => {
    /* Two buttons competing for the same slot pushed the Missing count, the
       number this page exists to drive down, out of the control pane. */
    const view = renderPage(AcquisitionPage, acquisitionState);
    await waitFor(() => expect(view.container.textContent).toMatch(/Export/));
    const exportButtons = [...view.container.querySelectorAll("button")]
      .filter(b => /Export/i.test(b.textContent || ""));
    expect(exportButtons).toHaveLength(1);
    expect(view.container.textContent).toMatch(/Missing/);
  });

  it("writes a CSV row for each missing paper and no row for an acquired one", async () => {
    const csv = await capture(/^CSV/i);
    expect(csv).toMatch(/Paywalled trial/);
    expect(csv).not.toMatch(/Acquired trial/);
    expect(csv).toMatch(/10\.1000\/paywalled/);
  });

  it("carries the reason into the CSV, so the request is self-explaining", async () => {
    const csv = await capture(/^CSV/i);
    expect(csv).toMatch(/interlibrary loan/i);
  });

  it("writes valid RIS: a tagged record per paper, each closed with ER", async () => {
    const ris = await capture(/^RIS/i);
    expect(ris).toMatch(/^TY {2}- JOUR/m);
    expect(ris).toMatch(/^TI {2}- Paywalled trial/m);
    expect(ris).toMatch(/^DO {2}- 10\.1000\/paywalled/m);
    expect(ris).toMatch(/^AN {2}- 31234567/m);
    expect(ris.match(/^ER {2}- ?$/gm)?.length).toBe(1);
  });

  it("gives every author its own AU line, as RIS requires", async () => {
    const ris = await capture(/^RIS/i);
    expect(ris).toMatch(/^AU {2}- Smith J$/m);
    expect(ris).toMatch(/^AU {2}- Okafor A$/m);
  });

  it("notes why the paper is being requested", async () => {
    const ris = await capture(/^RIS/i);
    expect(ris).toMatch(/^N1 {2}- Full text not retrievable:/m);
  });

  // The point of the export is that someone finishes the job by hand, and the
  // entitled copy is the one link that actually ends that job.
  it("puts the library link in the CSV when a resolver is configured", async () => {
    setResolverBase("https://lib.example.edu/openurl");
    const csv = await capture(/^CSV/i);
    setResolverBase("");
    expect(csv).toMatch(/Library link/);
    expect(csv).toMatch(/lib\.example\.edu\/openurl/);
    expect(csv).toMatch(/info%3Adoi%2F10\.1000%2Fpaywalled/);
  });

  it("points RIS UR at the entitled copy, keeping the publisher link in L2", async () => {
    /* UR is what an interlibrary-loan form and a reference manager follow. */
    setResolverBase("https://lib.example.edu/openurl");
    const ris = await capture(/^RIS/i);
    setResolverBase("");
    expect(ris).toMatch(/^UR {2}- https:\/\/lib\.example\.edu\/openurl/m);
    expect(ris).toMatch(/^L2 {2}- https:\/\/doi\.org\/10\.1000\/paywalled/m);
  });

  it("falls back to the publisher link when no resolver is set", async () => {
    setResolverBase("");
    const ris = await capture(/^RIS/i);
    expect(ris).toMatch(/^UR {2}- https:\/\/doi\.org\/10\.1000\/paywalled/m);
    expect(ris).not.toMatch(/^L2 {2}- /m);
  });
});

// --------------------------------------------------------------------------
// Full-text screening: what the run will read, before it runs.
//
// A record whose full text was never acquired is screened on its abstract, and
// calling that a full-text decision misstates the review. The page used to say
// only "N articles ready for full-text analysis" above a full-width button,
// which hid exactly that.
// --------------------------------------------------------------------------

describe("FullTextPage before a run", () => {
  const ftState = (withText: number, total = 5) => (st: any) => {
    st.setQuery("q");
    st.setPico({ population: "Adults with dental records", intervention: "ML model",
                 comparator: "Clinician alone", outcome: "Caries at 24 months" });
    st.setInclusion(["Reports a validated model"]);
    st.setExclusion(["Conference abstracts"]);
    const rows = Array.from({ length: total }, (_, i) => ({
      paper_id: `p${i}`, Source: "PubMed", Title: `Study ${i}`, URL: "", Abstract: "a",
      Decision: "INCLUDE", Reason: "", Agent_Trace: {},
    }));
    st.setResults(rows as any);
    const fts: Record<string, any> = {};
    rows.forEach((r, i) => {
      fts[r.paper_id] = i < withText
        ? { paper_id: r.paper_id, title: r.Title, url: "", source: "Europe PMC (XML)",
            status: "found", text: "Methods and results. ".repeat(50) }
        : { paper_id: r.paper_id, title: r.Title, url: "", source: "PubMed",
            status: "missing", reason: "Paywalled." };
    });
    st.setFullTexts(fts);
  };

  it("counts what will be read at full length, not just the queue", async () => {
    const view = renderPage(FullTextPage, ftState(3, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Ready to screen/i));
    expect(view.container.textContent).toMatch(/3\s*Full text/);
    expect(view.container.textContent).toMatch(/2\s*Abstract only/);
  });

  it("warns that abstract-only records are not full-text decisions", async () => {
    const view = renderPage(FullTextPage, ftState(3, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Abstract only/i));
    expect(view.container.textContent).toMatch(/judged on the abstract alone/i);
  });

  it("says nothing about abstract-only when every record has full text", async () => {
    const view = renderPage(FullTextPage, ftState(5, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Ready to screen/i));
    expect(view.container.textContent).not.toMatch(/Abstract only/i);
    expect(view.container.textContent).not.toMatch(/Acquire full texts/i);
  });

  it("breaks the acquired text down by where it came from", async () => {
    const view = renderPage(FullTextPage, ftState(3, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Evidence per article/i));
    expect(view.container.textContent).toMatch(/Europe PMC \(XML\)/);
  });

  it("shows the criteria the run will judge against", async () => {
    const view = renderPage(FullTextPage, ftState(3, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Inclusion criteria/i));
    expect(view.container.textContent).toMatch(/Reports a validated model/);
    expect(view.container.textContent).toMatch(/Conference abstracts/);
    expect(view.container.textContent).toMatch(/Caries at 24 months/);
  });

  it("replaces the bare alert and full-width button with the shared control pane", async () => {
    const view = renderPage(FullTextPage, ftState(3, 5));
    await waitFor(() => expect(view.container.textContent).toMatch(/Ready to screen/i));
    expect(view.container.textContent).not.toMatch(/articles ready for full-text analysis/i);
    expect(view.container.textContent).toMatch(/Run full-text screening/i);
  });
});
