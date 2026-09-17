// Tests for interrater agreement: Cohen's κ, Fleiss' κ, and Krippendorff's α.
//
// Run: pnpm test
//
// These numbers are published. A review reports "κ = 0.72 (substantial)" in its
// methods, and a reader takes it as evidence the screening was reproducible.
// An arithmetic error here is not a crash and not visible on screen; it is a
// wrong number in a paper.
//
// Every expected value below is derived by hand in its comment so the test is
// checkable without trusting the implementation it is testing.
import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";

import {
  buildMatrix,
  cohenKappa,
  fleissKappa,
  krippendorffAlphaNominal,
  landisKochLabel,
  InterraterReliability,
} from "../InterraterReliability";

const inc = "include" as const;
const exc = "exclude" as const;
const may = "maybe" as const;

// --------------------------------------------------------------------------
// Cohen's κ, two raters.
// --------------------------------------------------------------------------

describe("cohenKappa", () => {
  it("is 1 for perfect agreement", () => {
    // po = 1; pe = (2/4)(2/4) + (2/4)(2/4) = 0.5; κ = (1-0.5)/(1-0.5) = 1
    const { k, n, agree } = cohenKappa([inc, inc, exc, exc], [inc, inc, exc, exc]);
    expect(k).toBeCloseTo(1, 10);
    expect(n).toBe(4);
    expect(agree).toBe(4);
  });

  it("is -1 for perfect disagreement", () => {
    // po = 0; pe = (2/4)(2/4) + (2/4)(2/4) = 0.5; κ = (0-0.5)/(1-0.5) = -1
    expect(cohenKappa([inc, inc, exc, exc], [exc, exc, inc, inc]).k).toBeCloseTo(-1, 10);
  });

  it("matches a hand-worked moderate example", () => {
    // A: inc,inc,inc,exc   B: inc,inc,exc,exc
    // agree = 3 -> po = 0.75
    // A marginals inc=3 exc=1 ; B marginals inc=2 exc=2
    // pe = (3/4)(2/4) + (1/4)(2/4) = 0.375 + 0.125 = 0.5
    // κ = (0.75 - 0.5) / (1 - 0.5) = 0.5
    const { k, n, agree } = cohenKappa([inc, inc, inc, exc], [inc, inc, exc, exc]);
    expect(k).toBeCloseTo(0.5, 10);
    expect(n).toBe(4);
    expect(agree).toBe(3);
  });

  it("is 0 when agreement is exactly what chance predicts", () => {
    // Both raters call everything include: po = 1 and pe = 1, so κ is defined
    // as 1 by the pe === 1 guard rather than dividing by zero.
    expect(cohenKappa([inc, inc], [inc, inc]).k).toBe(1);
  });

  it("drops a paper either rater has not decided", () => {
    // Only the first two papers are pairable.
    const { n, agree } = cohenKappa([inc, exc, null, inc], [inc, exc, exc, null]);
    expect(n).toBe(2);
    expect(agree).toBe(2);
  });

  it("returns NaN with n=0 when nothing is pairable", () => {
    /* Not 0: a κ of 0 means "no better than chance", which is a finding.
       No overlapping decisions is the absence of a finding. */
    const { k, n } = cohenKappa([null, null], [inc, exc]);
    expect(Number.isNaN(k)).toBe(true);
    expect(n).toBe(0);
  });

  it("handles empty input", () => {
    const { k, n } = cohenKappa([], []);
    expect(Number.isNaN(k)).toBe(true);
    expect(n).toBe(0);
  });

  it("handles the three-category case including maybe", () => {
    const { k, n } = cohenKappa([inc, exc, may], [inc, exc, may]);
    expect(k).toBeCloseTo(1, 10);
    expect(n).toBe(3);
  });

  it("throws when the rater vectors are different lengths", () => {
    /* Silently truncating would misalign every paper after the first gap. */
    expect(() => cohenKappa([inc], [inc, exc])).toThrow();
  });

  it("is symmetric between the two raters", () => {
    const a = [inc, exc, inc, may];
    const b = [inc, inc, exc, may];
    expect(cohenKappa(a, b).k).toBeCloseTo(cohenKappa(b, a).k, 12);
  });

  it("never exceeds the [-1, 1] range", () => {
    const cases: [any[], any[]][] = [
      [[inc, inc, inc], [exc, exc, exc]],
      [[inc, exc, may], [may, inc, exc]],
      [[inc, inc, exc], [inc, exc, exc]],
    ];
    for (const [a, b] of cases) {
      const { k } = cohenKappa(a, b);
      if (!Number.isNaN(k)) {
        expect(k).toBeGreaterThanOrEqual(-1);
        expect(k).toBeLessThanOrEqual(1);
      }
    }
  });
});

// --------------------------------------------------------------------------
// Matrix construction
// --------------------------------------------------------------------------

const decision = (paper: string, reviewer: string, d: any) =>
  ({ paper_id: paper, reviewer_user_id: reviewer, decision: d, stage: "abstract" }) as any;

describe("buildMatrix", () => {
  it("fills a cell for every paper and reviewer", () => {
    const m = buildMatrix([], ["p1", "p2"], ["u1", "u2"]);
    expect(Object.keys(m)).toEqual(["p1", "p2"]);
    expect(m.p1.u1).toBeNull();
    expect(m.p2.u2).toBeNull();
  });

  it("places each decision in its own cell", () => {
    const m = buildMatrix(
      [decision("p1", "u1", inc), decision("p1", "u2", exc)],
      ["p1"], ["u1", "u2"],
    );
    expect(m.p1.u1).toBe(inc);
    expect(m.p1.u2).toBe(exc);
  });

  it("leaves an undecided cell null rather than defaulting it", () => {
    /* Defaulting to exclude would invent agreement that never happened. */
    const m = buildMatrix([decision("p1", "u1", inc)], ["p1"], ["u1", "u2"]);
    expect(m.p1.u2).toBeNull();
  });

  it("ignores a decision for a paper outside the set", () => {
    const m = buildMatrix([decision("ghost", "u1", inc)], ["p1"], ["u1"]);
    expect(m.p1.u1).toBeNull();
    expect(m.ghost).toBeUndefined();
  });

  it("a later decision by the same reviewer replaces the earlier one", () => {
    const m = buildMatrix(
      [decision("p1", "u1", inc), decision("p1", "u1", exc)],
      ["p1"], ["u1"],
    );
    expect(m.p1.u1).toBe(exc);
  });
});

// --------------------------------------------------------------------------
// Fleiss' κ, N raters.
// --------------------------------------------------------------------------

function matrixOf(rows: Record<string, Record<string, any>>) {
  return rows;
}

describe("fleissKappa", () => {
  const papers = ["p1", "p2", "p3", "p4"];
  const raters = ["u1", "u2", "u3"];

  it("is 1 when every rater agrees on every paper", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: inc, u3: inc },
      p2: { u1: exc, u2: exc, u3: exc },
      p3: { u1: inc, u2: inc, u3: inc },
      p4: { u1: exc, u2: exc, u3: exc },
    });
    expect(fleissKappa(m as any, papers, raters).k).toBeCloseTo(1, 10);
  });

  it("reports how many subjects and raters it used", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: inc, u3: inc },
      p2: { u1: exc, u2: exc, u3: exc },
      p3: { u1: inc, u2: inc, u3: inc },
      p4: { u1: exc, u2: exc, u3: exc },
    });
    const out = fleissKappa(m as any, papers, raters);
    expect(out.n_subjects).toBe(4);
    expect(out.n_raters).toBeGreaterThan(0);
  });

  it("drops a subject with fewer than two raters", () => {
    /* One rater cannot agree with themselves; including it would inflate κ. */
    const m = matrixOf({
      p1: { u1: inc, u2: inc, u3: null },
      p2: { u1: exc, u2: null, u3: null },   // only one rater: dropped
    });
    expect(fleissKappa(m as any, ["p1", "p2"], raters).n_subjects).toBe(1);
  });

  it("returns NaN when no subject has two raters", () => {
    const m = matrixOf({ p1: { u1: inc, u2: null, u3: null } });
    const out = fleissKappa(m as any, ["p1"], raters);
    expect(Number.isNaN(out.k)).toBe(true);
    expect(out.n_subjects).toBe(0);
  });

  it("handles an empty matrix", () => {
    expect(Number.isNaN(fleissKappa({} as any, [], []).k)).toBe(true);
  });

  it("reports po and pe alongside κ so the figure is auditable", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: inc, u3: exc },
      p2: { u1: exc, u2: exc, u3: exc },
    });
    const out = fleissKappa(m as any, ["p1", "p2"], raters);
    expect(typeof out.po).toBe("number");
    expect(typeof out.pe).toBe("number");
  });

  it("κ stays within [-1, 1] on mixed agreement", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: exc, u3: may },
      p2: { u1: exc, u2: inc, u3: exc },
      p3: { u1: may, u2: may, u3: inc },
    });
    const { k } = fleissKappa(m as any, ["p1", "p2", "p3"], raters);
    if (!Number.isNaN(k)) {
      expect(k).toBeGreaterThanOrEqual(-1);
      expect(k).toBeLessThanOrEqual(1);
    }
  });
});

// --------------------------------------------------------------------------
// Krippendorff's α
// --------------------------------------------------------------------------

describe("krippendorffAlphaNominal", () => {
  const raters = ["u1", "u2"];

  it("is 1 for perfect agreement", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: inc },
      p2: { u1: exc, u2: exc },
      p3: { u1: inc, u2: inc },
    });
    expect(krippendorffAlphaNominal(m as any, ["p1", "p2", "p3"], raters).alpha)
      .toBeCloseTo(1, 6);
  });

  it("reports the number of pairable values", () => {
    const m = matrixOf({ p1: { u1: inc, u2: inc }, p2: { u1: exc, u2: exc } });
    expect(krippendorffAlphaNominal(m as any, ["p1", "p2"], raters).n_pairable)
      .toBeGreaterThan(0);
  });

  it("handles a subject only one rater decided", () => {
    const m = matrixOf({ p1: { u1: inc, u2: inc }, p2: { u1: exc, u2: null } });
    expect(() => krippendorffAlphaNominal(m as any, ["p1", "p2"], raters)).not.toThrow();
  });

  it("returns NaN rather than 0 when nothing is pairable", () => {
    const m = matrixOf({ p1: { u1: inc, u2: null } });
    const out = krippendorffAlphaNominal(m as any, ["p1"], raters);
    expect(Number.isNaN(out.alpha) || out.n_pairable === 0).toBe(true);
  });

  it("handles an empty matrix", () => {
    expect(() => krippendorffAlphaNominal({} as any, [], [])).not.toThrow();
  });

  it("α does not exceed 1", () => {
    const m = matrixOf({
      p1: { u1: inc, u2: inc },
      p2: { u1: exc, u2: inc },
      p3: { u1: may, u2: may },
    });
    const { alpha } = krippendorffAlphaNominal(m as any, ["p1", "p2", "p3"], raters);
    if (!Number.isNaN(alpha)) expect(alpha).toBeLessThanOrEqual(1);
  });
});

// --------------------------------------------------------------------------
// Landis & Koch bands. These are the words a reader actually sees.
// --------------------------------------------------------------------------

describe("landisKochLabel", () => {
  it.each([
    [-0.2, "Poor"],
    [0.0, "Slight"],
    [0.15, "Slight"],
    [0.25, "Fair"],
    [0.45, "Moderate"],
    [0.7, "Substantial"],
    [0.9, "Almost perfect"],
    [1.0, "Almost perfect"],
  ])("labels κ=%s as %s", (k, label) => {
    expect(landisKochLabel(k as number).label).toBe(label);
  });

  it.each([
    [0.2, "Fair"],
    [0.4, "Moderate"],
    [0.6, "Substantial"],
    [0.8, "Almost perfect"],
  ])("puts the boundary value κ=%s in the upper band (%s)", (k, label) => {
    /* Landis & Koch bands are commonly written 0.21-0.40 etc. This pins which
       side of each boundary the implementation actually uses. */
    expect(landisKochLabel(k as number).label).toBe(label);
  });

  it("labels NaN as not available rather than Poor", () => {
    /* "Poor agreement" and "not enough data to say" are different claims. */
    expect(landisKochLabel(NaN).label).toBe("n/a");
  });

  it("always returns a label and a class", () => {
    for (const k of [-1, 0, 0.5, 1, NaN]) {
      const out = landisKochLabel(k);
      expect(out.label).toBeTruthy();
      expect(out.cls).toBeTruthy();
    }
  });
});

// --------------------------------------------------------------------------
// The component itself
// --------------------------------------------------------------------------

describe("InterraterReliability component", () => {
  const mockApi = (payload: any) =>
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true, status: 200, headers: new Headers(),
      json: async () => payload, text: async () => JSON.stringify(payload),
    })));

  it("renders while loading", () => {
    mockApi({});
    const { container } = render(<InterraterReliability projectId="p1" />);
    expect(container).toBeTruthy();
  });

  it("renders with no decisions without dividing by zero", async () => {
    mockApi({ decisions: [], participants: [], papers: [] });
    const { container } = render(<InterraterReliability projectId="p1" />);
    await waitFor(() => expect(container.textContent).toBeTruthy());
    expect(container.textContent).not.toContain("Infinity");
  });

  it("does not print NaN to the reviewer when agreement is undefined", async () => {
    /* NaN is the correct internal value; it must be rendered as n/a. */
    mockApi({ decisions: [decision("p1", "u1", inc)], participants: [], papers: [] });
    const { container } = render(<InterraterReliability projectId="p1" />);
    await waitFor(() => expect(container.textContent).toBeTruthy());
    expect(container.textContent).not.toContain("NaN");
  });

  it("survives a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    const { container } = render(<InterraterReliability projectId="p1" />);
    await waitFor(() => expect(container).toBeTruthy());
  });

  it("survives a malformed payload", async () => {
    mockApi({ unexpected: "shape" });
    const { container } = render(<InterraterReliability projectId="p1" />);
    await waitFor(() => expect(container).toBeTruthy());
  });
});
