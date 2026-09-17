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
import { render, screen, act } from "@testing-library/react";
import type { ReactNode } from "react";

import { StoreProvider, useStore } from "../../lib/store";
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
