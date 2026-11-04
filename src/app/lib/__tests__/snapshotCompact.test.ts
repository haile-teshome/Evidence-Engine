// The saved session is one JSON document written after every change. These helpers
// take redundant bytes out on the way to disk and put them back on load. The
// property under test: a compact-then-expand round trip is invisible to the app.
import { describe, it, expect } from "vitest";
import {
  compactFullTexts, compactArchive, expandArchive, compactRawPapers, expandRawPapers, CURRENT_RESULTS,
} from "../snapshotCompact";

const rec = (id: string, extra: Record<string, any> = {}) => ({
  paper_id: id, title: `T${id}`, url: "", source: "PubMed", status: "found" as const, ...extra,
});

describe("compactFullTexts", () => {
  it("drops the body only when a cache key can bring it back", () => {
    const out = compactFullTexts({
      keyed: rec("keyed", { text: "long body", text_key: "keyed" }),
      legacy: rec("legacy", { text: "long body" }),
      missing: rec("missing", { status: "missing", reason: "paywalled" }),
    } as any);
    expect(out.keyed.text).toBeUndefined();
    expect(out.keyed.text_key).toBe("keyed");
    expect(out.legacy.text).toBe("long body");
    expect(out.missing.status).toBe("missing");
  });

  it("leaves every other field alone", () => {
    const out = compactFullTexts({ k: rec("k", { text: "b", text_key: "k", pdf_url: "/api/x", links: { doi: "d" } }) } as any);
    expect(out.k).toEqual({ paper_id: "k", title: "Tk", url: "", source: "PubMed", status: "found", text_key: "k", pdf_url: "/api/x", links: { doi: "d" } });
  });
});

describe("screening archive", () => {
  const results = [{ paper_id: "a", Decision: "INCLUDE" }, { paper_id: "b", Decision: "EXCLUDE" }] as any[];
  const older = [{ paper_id: "a", Decision: "EXCLUDE" }] as any[];
  const archive = [
    { id: "r1", ranAt: "t", mode: "full", model: "m", screenedNow: 1, totalAfter: 1, included: 0, excluded: 1, durationSec: 1, results: older },
    { id: "r2", ranAt: "t", mode: "full", model: "m", screenedNow: 2, totalAfter: 2, included: 1, excluded: 1, durationSec: 1, results },
  ] as any[];

  it("replaces the pass whose results are the live table with a marker", () => {
    const out = compactArchive(archive, results);
    expect(out[0].results).toBe(older);            // an older pass keeps its own copy
    expect(out[1].results).toEqual([]);
    expect(out[1].resultsRef).toBe(CURRENT_RESULTS);
  });

  it("expands the marker back to the very array results hydrate to", () => {
    const revived = JSON.parse(JSON.stringify({ results, archive: compactArchive(archive, results) }));
    const out = expandArchive(revived.archive, revived.results)!;
    expect(out[1].results).toBe(revived.results);   // identity, so the run-history panel finds it
    expect(out[1].resultsRef).toBeUndefined();
    expect(out[0].results).toEqual(older);
  });

  it("passes through when there is nothing to expand", () => {
    expect(expandArchive(null, results)).toBeNull();
    expect(expandArchive(undefined, results)).toBeUndefined();
    expect(expandArchive(archive, null)![1].results).toBe(results);
  });
});

describe("raw papers", () => {
  const paper = (id: string, abstract: string) => ({ id, title: `S${id}`, abstract, source: "PubMed", url: "" });
  const unique = [paper("1", "same"), paper("2", "same2")];
  const raw = [paper("1", "same"), paper("1", "same"), paper("2", "different"), paper("3", "only raw")];

  it("drops an abstract only when the unique twin carries the identical text", () => {
    const out = compactRawPapers(raw as any, unique as any)!;
    expect(out[0].abstract).toBeUndefined();
    expect(out[1].abstract).toBeUndefined();
    expect(out[2].abstract).toBe("different");
    expect(out[3].abstract).toBe("only raw");
  });

  it("round-trips to the original records through JSON", () => {
    const revived = JSON.parse(JSON.stringify({ raw: compactRawPapers(raw as any, unique as any), unique }));
    expect(expandRawPapers(revived.raw, revived.unique)).toEqual(raw);
  });

  it("is a no-op without both collections or without markers", () => {
    expect(compactRawPapers(raw as any, null)).toBe(raw);
    expect(compactRawPapers(null, unique as any)).toBeNull();
    expect(expandRawPapers(raw, unique as any)).toBe(raw);
  });
});
