// Pure frontend logic under test. No React, no DOM, no network.
//
// Run: npx vitest run
//
// These cover the decision and credential logic that runs on every study, and
// the reviewer-override path, where a bug silently changes what ends up in a
// published review rather than throwing.
import { describe, it, expect, beforeEach, afterEach } from "vitest";

// Minimal localStorage so these run under the plain node environment, with no
// jsdom dependency. dbKeys wraps every access in try/catch precisely because
// this API can be absent or throw, and that behaviour is asserted below.
if (typeof (globalThis as any).localStorage?.clear !== "function") {
  const store = new Map<string, string>();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
      setItem: (k: string, v: string) => void store.set(k, String(v)),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
      key: (i: number) => Array.from(store.keys())[i] ?? null,
      get length() { return store.size; },
    },
  });
}

import {
  effectiveAbstractDecision,
  effectiveFullTextDecision,
  categoriseAbstractExclusion,
} from "../exclusionBucketing";
import {
  isValidContactEmail,
  getContactEmail,
  setContactEmail,
  setDbKey,
  getDbKey,
  hasDbKey,
  dbKeyHeaders,
} from "../dbKeys";

// --------------------------------------------------------------------------
// Reviewer overrides. A human decision must always win over the AI's, and the
// override must be scoped to the paper it was made on.
// --------------------------------------------------------------------------

const result = (over: Partial<any> = {}) =>
  ({ paper_id: "p1", Decision: "INCLUDE", Agent_Trace: {}, ...over }) as any;

describe("effectiveAbstractDecision", () => {
  it("returns the AI decision when there is no override", () => {
    expect(effectiveAbstractDecision(result(), {})).toBe("INCLUDE");
  });

  it("lets a reviewer override win", () => {
    expect(effectiveAbstractDecision(result(), { p1: "EXCLUDE" })).toBe("EXCLUDE");
  });

  it("applies an override only to the paper it was made on", () => {
    expect(effectiveAbstractDecision(result(), { other: "EXCLUDE" })).toBe("INCLUDE");
  });

  it("treats an override back to the AI's own value as a no-op", () => {
    expect(effectiveAbstractDecision(result(), { p1: "INCLUDE" })).toBe("INCLUDE");
  });
});

describe("effectiveFullTextDecision", () => {
  it("respects the override", () => {
    const r = { paper_id: "p2", Decision: "Include" } as any;
    expect(effectiveFullTextDecision(r, { p2: "Exclude" })).toBe("Exclude");
    expect(effectiveFullTextDecision(r, {})).toBe("Include");
  });
});

// --------------------------------------------------------------------------
// Exclusion bucketing drives the PRISMA "reasons for exclusion" counts, which
// are published. A crash or a silent miscategorisation both matter.
// --------------------------------------------------------------------------

describe("categoriseAbstractExclusion", () => {
  it("returns a non-empty reason even with no criteria and no assessment", () => {
    expect(categoriseAbstractExclusion(result(), [], []).length).toBeGreaterThan(0);
  });

  it("does not throw on a missing Pico_Assessment", () => {
    expect(() =>
      categoriseAbstractExclusion(result({ Pico_Assessment: undefined }), [], []),
    ).not.toThrow();
  });

  it("falls back to the frame bucket when no named criterion failed", () => {
    const r = result({
      Pico_Assessment: { concept: { vote: "FAIL" }, population: { vote: "PASS" } },
    });
    expect(categoriseAbstractExclusion(r, [], []).length).toBeGreaterThan(0);
  });

  it("keeps reasons short enough to aggregate in a PRISMA table", () => {
    const long = "x".repeat(500);
    const r = result({ Pico_Assessment: { concept: { vote: "FAIL", reasoning: long } } });
    expect(categoriseAbstractExclusion(r, [], []).length).toBeLessThanOrEqual(120);
  });
});

// --------------------------------------------------------------------------
// Contact email. Scholarly APIs reject placeholder addresses, so sending a
// fabricated one is strictly worse than sending nothing at all.
// --------------------------------------------------------------------------

describe("isValidContactEmail", () => {
  it.each([
    "j.smith@ucsf.edu",
    "R.Patel@lab.uni-koeln.de",
    "a.b+tag@sub.domain.org",
  ])("accepts the real address %s", (e) => {
    expect(isValidContactEmail(e)).toBe(true);
  });

  it.each([
    "",
    "   ",
    "notanemail",
    "a@b",
    "@nodomain.com",
    "no-at-sign.com",
    "someone@example.com",
    "someone@example.org",
    "test@localhost",
  ])("rejects the unusable address %s", (e) => {
    expect(isValidContactEmail(e)).toBe(false);
  });
});

describe("credential headers", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("sends no email header when none is set", () => {
    expect(dbKeyHeaders()["X-User-Contact-Email"]).toBeUndefined();
  });

  it("sends no email header for a placeholder address", () => {
    setContactEmail("someone@example.com");
    expect(dbKeyHeaders()["X-User-Contact-Email"]).toBeUndefined();
  });

  it("sends the header for a real address", () => {
    setContactEmail("j.smith@ucsf.edu");
    expect(dbKeyHeaders()["X-User-Contact-Email"]).toBe("j.smith@ucsf.edu");
  });

  it("round-trips and clears the stored address", () => {
    setContactEmail("j.smith@ucsf.edu");
    expect(getContactEmail()).toBe("j.smith@ucsf.edu");
    setContactEmail("");
    expect(getContactEmail()).toBe("");
  });

  it("omits headers for unset database keys", () => {
    expect(dbKeyHeaders()["X-DB-Core-Key"]).toBeUndefined();
  });

  it("sends a database key once set", () => {
    setDbKey("core", "abc123");
    expect(hasDbKey("core")).toBe(true);
    expect(dbKeyHeaders()["X-DB-Core-Key"]).toBe("abc123");
  });

  it("trims whitespace and treats a blank key as unset", () => {
    setDbKey("core", "   ");
    expect(getDbKey("core")).toBe("");
    expect(dbKeyHeaders()["X-DB-Core-Key"]).toBeUndefined();
  });
});
