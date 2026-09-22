// Tests for the pre-run screening preview.
//
// Run: pnpm test
//
// Screening is one model call per article against criteria that were mostly
// inferred rather than typed. The panel it replaced said "34 of 34 unique
// articles ready for screening", which let nobody check anything: the useful
// questions are WHICH articles and against WHAT, and both are answerable for
// free right up until the run starts.
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { ScreeningPreview } from "../AbstractPage";
import type { ScreeningPlan } from "../../lib/store";

const paper = (i: number) => ({
  id: `p${i}`, paper_id: `p${i}`, title: `Study number ${i}`,
  abstract: "abstract", source: "PubMed", url: "",
}) as any;

const props = (over: Record<string, any> = {}) => ({
  papers: [1, 2, 3, 4, 5, 6, 7].map(paper),
  excludedIds: new Set<string>(),
  framework: "pico" as const,
  pico: { population: "adults with type 2 diabetes", intervention: "metformin",
          comparator: "placebo", outcome: "HbA1c" },
  inclusion: ["RCTs only", "English language"],
  exclusion: ["Animal studies"],
  ...over,
});

describe("what will run", () => {
  it("does not print a raw model id", () => {
    /* A HuggingFace repo path with a quantisation suffix
       ("hf.co/mradermacher/leads-mistral-7b-v1-GGUF:Q4_K_M") is an
       implementation detail, and it was the widest thing in the header. The
       sidebar already names the active model in readable form. */
    render(<ScreeningPreview {...props()} />);
    expect(document.body.textContent).not.toContain("GGUF");
  });

  it("does not list an excluded article among those to screen", () => {
    render(<ScreeningPreview {...props({ excludedIds: new Set(["p1"]) })} />);
    expect(screen.queryByText("Study number 1")).toBeNull();
  });

  it("renders nothing when there is nothing left to screen", () => {
    const { container } = render(
      <ScreeningPreview {...props({ excludedIds: new Set(["p1","p2","p3","p4","p5","p6","p7"]) })} />);
    expect(container.textContent).toBe("");
  });
});

describe("what it screens against", () => {
  it("shows the populated question elements", () => {
    render(<ScreeningPreview {...props()} />);
    expect(document.body.textContent).toContain("adults with type 2 diabetes");
    expect(document.body.textContent).toContain("metformin");
  });

  it("uses the labels of the active framework", () => {
    render(<ScreeningPreview {...props({
      framework: "pcc", pico: { population: "", concept: "AI tools", context: "primary care" },
    })} />);
    expect(document.body.textContent).toContain("Concept");
    expect(document.body.textContent).toContain("Context");
    expect(document.body.textContent).not.toContain("Intervention");
  });

  it("omits empty elements rather than showing blank rows", () => {
    render(<ScreeningPreview {...props({
      pico: { population: "adults", intervention: "", comparator: "", outcome: "" },
    })} />);
    expect(document.body.textContent).toContain("Population");
    expect(document.body.textContent).not.toContain("Comparator");
  });

  it("lists the inclusion criteria", () => {
    render(<ScreeningPreview {...props()} />);
    expect(document.body.textContent).toMatch(/Inclusion criteria/);
    expect(document.body.textContent).toContain("RCTs only");
    expect(document.body.textContent).toContain("English language");
  });

  it("lists the exclusion criteria", () => {
    render(<ScreeningPreview {...props()} />);
    expect(document.body.textContent).toMatch(/Exclusion criteria/);
    expect(document.body.textContent).toContain("Animal studies");
  });

  it("says so when a criteria list is empty instead of showing a bare heading", () => {
    render(<ScreeningPreview {...props({ exclusion: [] })} />);
    expect(document.body.textContent).toMatch(/None set/);
  });

  it("colours include and exclude differently", () => {
    /* The two lists are read at a glance before an irreversible run; colour is
       what separates them without reading. Both also carry a text heading, so
       the distinction never rests on colour alone. */
    const { container } = render(<ScreeningPreview {...props()} />);
    expect(container.querySelector(".bg-emerald-600")).not.toBeNull();
    expect(container.querySelector(".bg-rose-600")).not.toBeNull();
  });

  it("warns when there are no criteria at all", () => {
    /* With nothing to judge against, every model call is guesswork. Better to
       say so before the run than to return 34 confident verdicts. */
    render(<ScreeningPreview {...props({
      pico: {}, inclusion: [], exclusion: [],
    })} />);
    expect(document.body.textContent).toMatch(/No criteria set/i);
  });
});

describe("the article list is gone", () => {
  it("does not list article titles", () => {
    /* Replaced by the criteria. A title tells the reader nothing they cannot
       see after the run; a wrong criterion mis-sorts the whole corpus. */
    render(<ScreeningPreview {...props()} />);
    expect(document.body.textContent).not.toContain("Study number 1");
    expect(document.body.textContent).not.toMatch(/Articles to screen/i);
  });
});

// --------------------------------------------------------------------------
// Search provenance, folded into the same card.
//
// It used to be its own bordered box above the preview. Both answer the same
// question -- is this the set I meant to screen? -- and splitting them meant
// reading two panels to get one answer.
// --------------------------------------------------------------------------

const plan = (over: Partial<ScreeningPlan> = {}): ScreeningPlan => ({
  sources: ["PubMed", "Europe PMC"],
  perDbQueries: { PubMed: "q", "Europe PMC": "q" },
  counts: { PubMed: 33, "Europe PMC": 1379 },
  retrieved: { PubMed: 33, "Europe PMC": 7 },
  ranAt: "2026-09-20T02:28:27Z",
  ...over,
});

describe("search provenance", () => {
  it("lists planned against retrieved for each source", () => {
    render(<ScreeningPreview {...props({ plan: plan() })} />);
    expect(document.body.textContent).toMatch(/PubMed/);
    expect(document.body.textContent).toMatch(/planned 33/);
    expect(document.body.textContent).toMatch(/retrieved/);
  });

  it("warns when a source returned fewer than planned", () => {
    render(<ScreeningPreview {...props({ plan: plan() })} />);
    expect(document.body.textContent).toMatch(/Fewer than planned|returned 7 of/i);
  });

  it("does not warn when every source met its plan", () => {
    render(<ScreeningPreview {...props({
      plan: plan({ counts: { PubMed: 33 }, retrieved: { PubMed: 33 }, sources: ["PubMed"] }),
    })} />);
    expect(document.body.textContent).not.toMatch(/Fewer than planned/i);
  });

  it("does not repeat the totals the control pane already carries", () => {
    /* Retrieved, duplicates and ready-to-screen are stats in the pane above.
       Restating them as a reconciliation row here was the same arithmetic
       twice on one screen. */
    render(<ScreeningPreview {...props({ plan: plan() })} />);
    expect(document.body.textContent).not.toMatch(/Total retrieved/i);
    expect(document.body.textContent).not.toMatch(/duplicates =/i);
  });

  it("labels each row with the database name and nothing else", () => {
    /* Every source now carries the in-use planning run's own query, so tagging
       one "(custom query)" marked the normal case as an exception. */
    render(<ScreeningPreview {...props({
      plan: plan({ perDbQueries: { PubMed: "different", "Europe PMC": "q" } }),
      activeQuery: "q",
    })} />);
    expect(document.body.textContent).not.toMatch(/custom query/i);
    expect(document.body.textContent).toMatch(/PubMed/);
    expect(document.body.textContent).toMatch(/Europe PMC/);
  });

  it("renders the rest of the preview when there was no planning run", () => {
    render(<ScreeningPreview {...props({ plan: null })} />);
    expect(document.body.textContent).not.toMatch(/Retrieval per database/i);
    expect(document.body.textContent).toMatch(/Inclusion criteria/i);
  });
});

// --------------------------------------------------------------------------
// Attributing a planned-vs-retrieved gap.
// --------------------------------------------------------------------------

describe("why a source fell short", () => {
  it("blames a stale plan when the counted query is not the one being run", () => {
    /* Plans recorded before per-source provenance existed only know the unified
       query, so that comparison stays as the fallback. */
    render(<ScreeningPreview {...props({
      plan: plan(),
      countedQuery: "an older query", activeQuery: "the current query",
    })} />);
    expect(document.body.textContent).toMatch(/measured on a different query/i);
    expect(document.body.textContent).toMatch(/Re-run Planning/i);
  });

  it("compares the PER-SOURCE query, not the unified one", () => {
    /* Retrieval is per source. A unified-query comparison could not see a
       per-database edit, so the counts and the retrieval silently described
       two different searches and the database took the blame. */
    const p = plan();
    p.countedPerDb = { ...p.perDbQueries, "Europe PMC": "the query the count was measured on" };
    render(<ScreeningPreview {...props({
      // Unified queries agree, so only the per-source check can catch this.
      plan: p, countedQuery: "same query", activeQuery: "same query",
    })} />);
    expect(document.body.textContent).toMatch(/measured on a different query/i);
  });

  it("does not call a complete retrieval capped", () => {
    /* Asking a source for its full planned yield and receiving it is a complete
       result. Reporting that as a cap told the reviewer to raise a limit that
       was never binding. */
    const p = plan();
    p.counts = { ...(p.counts || {}), PubMed: 33 };
    p.retrieved = { ...p.retrieved, PubMed: 33 };
    p.sourceStatus = {
      ...(p.sourceStatus || {}),
      PubMed: { status: "ok", count: 33, requested: 33, capped: true, ms: 900 },
    };
    render(<ScreeningPreview {...props({ plan: p })} />);
    expect(document.body.textContent).not.toMatch(/capped/i);
    expect(document.body.textContent).not.toMatch(/raise its limit/i);
  });

  it("never offers the reviewer a list of possible causes", () => {
    /* "The source may have errored, or it reads this query differently" was
       the app declining to report something it had just observed. */
    render(<ScreeningPreview {...props({
      plan: plan(), countedQuery: "same", activeQuery: "same",
    })} />);
    expect(document.body.textContent).not.toMatch(/may have errored/i);
  });

  it("does not nag about a trivial shortfall", () => {
    /* Any strict inequality fired on 1,379 vs 1,378. */
    render(<ScreeningPreview {...props({
      plan: plan({ sources: ["PubMed"], counts: { PubMed: 1379 }, retrieved: { PubMed: 1378 } }),
    })} />);
    expect(document.body.textContent).not.toMatch(/Fewer than planned/i);
  });

  it("still flags a real collapse", () => {
    render(<ScreeningPreview {...props({
      plan: plan({ sources: ["PubMed"], counts: { PubMed: 1379 }, retrieved: { PubMed: 7 } }),
    })} />);
    expect(document.body.textContent).toMatch(/Fewer than planned|returned 7 of/i);
  });

  it("falls back gracefully on a plan recorded before diagnostics existed", () => {
    render(<ScreeningPreview {...props({ plan: plan({ sourceStatus: undefined }) })} />);
    expect(document.body.textContent).toMatch(/Retrieval per database/i);
  });
});
