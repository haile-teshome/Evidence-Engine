// Audit trail, session persistence, and client identity. No network, no DOM.
//
// Run: npx vitest run
//
// The audit log is the PRISMA-AI / RAISE disclosure: it states which AI made
// which decision with which model. If a decision is missing from it, or a row
// shifts because a reasoning string contained a comma, the published statement
// no longer matches what actually happened.
import { describe, it, expect, beforeEach, afterEach } from "vitest";

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

import { compileAuditLog, auditToJson, auditToCsv } from "../auditLog";
import {
  getReviewerId,
  getAuthToken,
  setAuthSession,
  DEFAULT_REVIEWER_ID,
} from "../backendClient";
import {
  registerPdfBlobs,
  removePdfBlob,
  getPdfBlob,
  hasPdfBlob,
  getDocHtml,
} from "../pdfBlobs";

// --------------------------------------------------------------------------
// Audit log
// --------------------------------------------------------------------------

const decision = (over: Partial<any> = {}) => ({
  stage: "Abstract screening",
  item: "A study of periodontal disease",
  id: "p1",
  verdict: "INCLUDE",
  score: 0.9,
  reasoning: "Matches all elements",
  criteria: [{ name: "Population", verdict: "PASS", reasoning: "adults" }],
  ...over,
});

describe("compileAuditLog", () => {
  it("returns the full log shape", () => {
    const log = compileAuditLog({});
    expect(log).toHaveProperty("generated_at");
    expect(log).toHaveProperty("manifest");
    expect(log).toHaveProperty("decisions");
    expect(log).toHaveProperty("note");
  });

  it("records the model used, which the disclosure has to state", () => {
    const log = compileAuditLog({ model: "qwen2.5:7b" });
    expect(JSON.stringify(log.manifest)).toContain("qwen2.5:7b");
  });

  it("honours a supplied timestamp", () => {
    const log = compileAuditLog({ generatedAt: "2026-09-09T00:00:00Z" });
    expect(log.generated_at).toBe("2026-09-09T00:00:00Z");
  });

  it("produces a timestamp when none is supplied", () => {
    expect(compileAuditLog({}).generated_at).toBeTruthy();
  });

  it("handles null inputs without throwing", () => {
    expect(() =>
      compileAuditLog({ qualityReports: null, rerankResults: null, manifest: null }),
    ).not.toThrow();
  });

  it("always carries a note explaining the log", () => {
    expect(typeof compileAuditLog({}).note).toBe("string");
  });
});

describe("auditToJson", () => {
  it("emits parseable JSON that round-trips", () => {
    const log = compileAuditLog({ model: "qwen2.5:7b" });
    expect(JSON.parse(auditToJson(log))).toEqual(JSON.parse(JSON.stringify(log)));
  });

  it("is pretty-printed for human review", () => {
    expect(auditToJson(compileAuditLog({}))).toContain("\n");
  });
});

describe("auditToCsv", () => {
  const log = (decisions: any[]) =>
    ({ ...compileAuditLog({ model: "qwen2.5:7b" }), decisions }) as any;

  it("emits a header row", () => {
    const first = auditToCsv(log([])).split("\n")[0];
    expect(first).toContain("Stage");
    expect(first).toContain("Verdict");
  });

  it("emits one row per criterion when criteria are present", () => {
    const csv = auditToCsv(log([decision({
      criteria: [
        { name: "Population", verdict: "PASS", reasoning: "adults" },
        { name: "Concept", verdict: "FAIL", reasoning: "no linkage" },
      ],
    })]));
    expect(csv.trim().split("\n").length).toBe(3); // header + 2
  });

  it("still emits one row for a decision with no criteria", () => {
    const csv = auditToCsv(log([decision({ criteria: [] })]));
    expect(csv.trim().split("\n").length).toBe(2);
  });

  it("loses no decision", () => {
    const csv = auditToCsv(log([
      decision({ id: "p1", criteria: [] }),
      decision({ id: "p2", criteria: [] }),
      decision({ id: "p3", criteria: [] }),
    ]));
    for (const id of ["p1", "p2", "p3"]) expect(csv).toContain(id);
  });

  it("quotes a field containing a comma so columns cannot shift", () => {
    const csv = auditToCsv(log([decision({
      reasoning: "Excluded, because the population was wrong",
      criteria: [],
    })]));
    expect(csv).toMatch(/"Excluded, because the population was wrong"/);
  });

  it("survives a reasoning string containing a quote character", () => {
    const csv = auditToCsv(log([decision({
      reasoning: 'The paper says "no linkage" here',
      criteria: [],
    })]));
    expect(csv.split("\n").length).toBeGreaterThan(1);
  });

  it("renders a null score as empty rather than the text null", () => {
    const csv = auditToCsv(log([decision({ score: null, criteria: [] })]));
    expect(csv).not.toContain("null");
  });

  it("handles an empty decision list", () => {
    expect(typeof auditToCsv(log([]))).toBe("string");
  });
});

// --------------------------------------------------------------------------
// Reviewer identity. Every collaboration request is attributed with this, so a
// wrong or blank id assigns a decision to the wrong person.
// --------------------------------------------------------------------------

describe("backendClient identity", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("falls back to a default reviewer id before sign-in", () => {
    expect(getReviewerId()).toBe(DEFAULT_REVIEWER_ID);
  });

  it("has no auth token before sign-in", () => {
    expect(getAuthToken()).toBeFalsy();
  });

  it("stores a session and reads both parts back", () => {
    setAuthSession("tok-123", "rev-789");
    expect(getAuthToken()).toBe("tok-123");
    expect(getReviewerId()).toBe("rev-789");
  });

  it("never returns an empty reviewer id, which would orphan decisions", () => {
    setAuthSession("tok", "");
    expect(getReviewerId()).toBeTruthy();
  });
});

// --------------------------------------------------------------------------
// PDF blob registry (in-memory object URLs)
// --------------------------------------------------------------------------

describe("pdfBlobs", () => {
  it("reports nothing registered initially", () => {
    expect(hasPdfBlob("never-registered")).toBe(false);
    expect(getPdfBlob("never-registered")).toBeUndefined();
  });

  it("registers and retrieves document html", () => {
    registerPdfBlobs([{ id: "doc1", html: "<p>Some text</p>" }]);
    expect(getDocHtml("doc1")).toContain("Some text");
  });

  it("removes a registration", () => {
    registerPdfBlobs([{ id: "doc2", html: "<p>x</p>" }]);
    removePdfBlob("doc2");
    expect(getDocHtml("doc2")).toBeUndefined();
  });

  it("handles an empty registration list", () => {
    expect(() => registerPdfBlobs([])).not.toThrow();
  });

  it("removing an unknown id is not an error", () => {
    expect(() => removePdfBlob("never-registered")).not.toThrow();
  });

  it("keeps registrations independent", () => {
    registerPdfBlobs([{ id: "a", html: "<p>A</p>" }, { id: "b", html: "<p>B</p>" }]);
    removePdfBlob("a");
    expect(getDocHtml("b")).toContain("B");
  });
});
