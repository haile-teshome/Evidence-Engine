// Tests for the analysis progress card.
//
// Run: pnpm test
//
// This card is the only feedback during a pipeline that can run for minutes,
// and it failed in the quietest possible way: the search-building step was
// marked "done" when it finished but never marked "running" when it started.
// So it sat on "pending" for its whole duration, nothing spun, and because NO
// stage was running the header fell back to "Preparing analysis…" — for 1m40s,
// on the slowest step in the pipeline, with no indication anything was
// happening at all.
//
// Nothing errored. The work completed correctly. The only symptom was silence,
// which is exactly the kind of defect a test has to catch instead of a user.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { AnalysisProgress, type Stage } from "../AnalysisProgress";

const stages = (running?: string, done: string[] = []): Stage[] =>
  ["pico", "question", "query", "adversarial"].map(id => ({
    id,
    label: `Step ${id}`,
    status: done.includes(id) ? "done" : id === running ? "running" : "pending",
  }));

const view = (st: Stage[]) =>
  render(<AnalysisProgress stages={st} startedAt={Date.now()} />);

describe("running step", () => {
  it("names the running step in the header", () => {
    /* Not just "2 of 4": on a step that takes minutes, the reader needs to know
       WHICH step is taking them. */
    view(stages("query", ["pico", "question"]));
    expect(document.body.textContent).toContain("Step query");
  });

  it("shows how far through the pipeline it is", () => {
    view(stages("query", ["pico", "question"]));
    expect(document.body.textContent).toMatch(/2 of 4 complete/);
  });

  it("spins a loader for the running step", () => {
    const { container } = view(stages("query", ["pico", "question"]));
    expect(container.querySelector(".animate-spin")).not.toBeNull();
  });

  it("does not spin when nothing is running", () => {
    const { container } = view(stages(undefined, ["pico", "question", "query", "adversarial"]));
    expect(container.querySelector(".animate-spin")).toBeNull();
  });

  it("says the analysis is complete when every step is done", () => {
    view(stages(undefined, ["pico", "question", "query", "adversarial"]));
    expect(document.body.textContent).toContain("Analysis complete");
  });

  it("does not claim to be 'preparing' once steps have finished", () => {
    /* THE REGRESSION. Two steps done, none marked running, and the header read
       "Preparing analysis…" as though nothing had started. */
    view(stages(undefined, ["pico", "question"]));
    expect(document.body.textContent).not.toMatch(/Preparing analysis/);
    expect(document.body.textContent).toMatch(/0 of 4|2 of 4|Starting/);
  });

  it("renders every step's label so the pipeline is legible up front", () => {
    view(stages("query", ["pico"]));
    for (const id of ["pico", "question", "query", "adversarial"]) {
      expect(document.body.textContent).toContain(`Step ${id}`);
    }
  });

  it("marks an errored step without hiding the rest", () => {
    const st = stages("adversarial", ["pico"]);
    st[1] = { ...st[1], status: "error", detail: "timed out" };
    view(st);
    expect(document.body.textContent).toContain("Step question");
  });

  it("handles an empty stage list without throwing", () => {
    expect(() => render(<AnalysisProgress stages={[]} startedAt={Date.now()} />)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// The invariant that was actually broken.
// --------------------------------------------------------------------------

describe("every declared stage is driven", () => {
  it("passes each INITIAL_STAGES id to runStage", () => {
    /* The card can only show a step as running if the caller marks it running.
       The "query" stage was declared in the list and rendered in the UI, but
       the code that did its work called runGroundedBuild directly and only ever
       marked it done — so it could never spin.

       Checked against the source because the invariant lives in the pairing of
       two lists that nothing else forces to agree. A stage added to the list
       without a runStage call renders as a step that silently never starts. */
    const src = readFileSync(resolve(__dirname, "../../pages/HomePage.tsx"), "utf8");

    const listBlock = src.slice(src.indexOf("const INITIAL_STAGES"));
    const declared = [...listBlock.slice(0, listBlock.indexOf("];")).matchAll(/id:\s*"([^"]+)"/g)]
      .map(m => m[1]);
    const driven = new Set([...src.matchAll(/runStage\(\s*"([^"]+)"/g)].map(m => m[1]));

    expect(declared.length).toBeGreaterThan(5);
    expect(declared.filter(id => !driven.has(id))).toEqual([]);
  });
});
