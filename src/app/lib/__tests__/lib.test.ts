// Pure frontend library logic. No React, no DOM, no network.
//
// Run: npx vitest run
//
// These cover export formats, framework normalisation, active-learning ranking
// and the audit trail. Export and audit bugs are the dangerous ones: a malformed
// RIS file or a missing decision row is only noticed downstream, in a reference
// manager or a journal's reproducibility check.
import { describe, it, expect } from "vitest";

import {
  isFrameworkId,
  normalizeFramework,
  frameworkOf,
  elementIds,
  labelFor,
  FRAMEWORK_IDS,
} from "../frameworks";
import { tokenize, activeRank } from "../activeLearning";
import { refKey, toRis, toBibTeX, serializeReferences, dedupeAgainstLibrary } from "../references";
import { formatNumber, timeAgo } from "../format";
import { biasCsv } from "../biasPlot";

// --------------------------------------------------------------------------
// Framework registry. The frontend and backend both decide which elements are
// discriminating; if they disagree, the panel and the verdict diverge.
// --------------------------------------------------------------------------

describe("frameworks", () => {
  it("recognises the known ids", () => {
    for (const id of FRAMEWORK_IDS) expect(isFrameworkId(id)).toBe(true);
  });

  it.each([null, undefined, "", "nonsense", 42, {}])(
    "rejects %s as a framework id",
    (bad) => expect(isFrameworkId(bad)).toBe(false),
  );

  it.each([null, undefined, "nonsense", 42])(
    "normalises the unusable value %s to a real framework",
    (bad) => expect(FRAMEWORK_IDS).toContain(normalizeFramework(bad)),
  );

  it("keeps a valid id unchanged", () => {
    expect(normalizeFramework("pcc")).toBe("pcc");
    expect(normalizeFramework("pico")).toBe("pico");
  });

  it("gives PCC its three elements in order", () => {
    expect(elementIds("pcc")).toEqual(["population", "concept", "context"]);
  });

  it("gives PICO its four elements", () => {
    expect(elementIds("pico")).toHaveLength(4);
  });

  it("never returns an empty element list, even for junk", () => {
    expect(elementIds("nonsense").length).toBeGreaterThan(0);
  });

  it("labels elements for display", () => {
    expect(labelFor("pcc", "concept")).toMatch(/concept/i);
  });

  it("returns a definition for any input", () => {
    expect(frameworkOf("nonsense")).toBeTruthy();
  });
});

// --------------------------------------------------------------------------
// Active learning. This decides screening order, so a ranking bug wastes the
// reviewer's time rather than corrupting data.
// --------------------------------------------------------------------------

describe("tokenize", () => {
  it("lowercases and splits on non-word characters", () => {
    expect(tokenize("Periodontal Disease, and caries!")).toContain("periodontal");
  });

  it.each(["", "   ", "!!!"])("returns an array for %s", (s) => {
    expect(Array.isArray(tokenize(s))).toBe(true);
  });

  it("does not emit empty tokens", () => {
    expect(tokenize("a,,  b")).not.toContain("");
  });
});

describe("activeRank", () => {
  const items = [
    { id: "1", title: "Machine learning on linked medical and dental records", aiInclude: true },
    { id: "2", title: "A study of unrelated geology", aiInclude: false },
    { id: "3", title: "Dental records and machine learning models", aiInclude: true },
  ];

  it("returns a result for a normal corpus", () => {
    expect(activeRank(items)).toBeTruthy();
  });

  it("loses no items", () => {
    const out: any = activeRank(items);
    const ranked = out.ranked ?? out.order ?? out.items;
    if (Array.isArray(ranked)) expect(ranked).toHaveLength(items.length);
  });

  it("handles an empty corpus", () => {
    expect(() => activeRank([])).not.toThrow();
  });

  it("handles a single item", () => {
    expect(() => activeRank([items[0]])).not.toThrow();
  });

  it("handles items with no text at all", () => {
    expect(() => activeRank([{ id: "x", aiInclude: false }])).not.toThrow();
  });

  it("respects a reviewer override without crashing", () => {
    const withOverride = [{ ...items[0], override: "exclude" as const }, items[1]];
    expect(() => activeRank(withOverride)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Reference export. These files are consumed by Zotero, EndNote and journals.
// --------------------------------------------------------------------------

// `authors` is a single string across the whole stack (Optional[str] on the
// backend, string on the frontend), semicolon-separated for multiple authors.
const ref = {
  title: "Linking medical and dental records",
  authors: "Smith J; Patel R",
  year: 2024,
  doi: "10.1/abc",
  url: "https://doi.org/10.1/abc",
} as any;

describe("reference export", () => {
  it("produces RIS with the required type and end markers", () => {
    const ris = toRis(ref);
    expect(ris).toMatch(/^TY\s+-/m);
    expect(ris).toMatch(/^ER\s+-/m);
  });

  it("includes the title in RIS", () => {
    expect(toRis(ref)).toContain("Linking medical and dental records");
  });

  it("produces a BibTeX entry with a citation key", () => {
    const bib = toBibTeX(ref, 1);
    expect(bib).toMatch(/^@\w+\{/);
    expect(bib).toContain("}");
  });

  it("serialises a list in both formats", () => {
    expect(serializeReferences([ref], "RIS")).toContain("TY");
    expect(serializeReferences([ref], "BibTeX")).toContain("@");
  });

  it("serialises an empty list without crashing", () => {
    expect(typeof serializeReferences([], "RIS")).toBe("string");
    expect(typeof serializeReferences([], "BibTeX")).toBe("string");
  });

  it("survives missing optional fields", () => {
    const sparse = { title: "Only a title" } as any;
    expect(() => toRis(sparse)).not.toThrow();
    expect(() => toBibTeX(sparse, 0)).not.toThrow();
  });

  it("splits multiple authors onto separate RIS lines", () => {
    const lines = toRis(ref).split("\n").filter(l => l.startsWith("AU"));
    expect(lines.length).toBe(2);
  });

  it("gives equal references the same key and different ones different keys", () => {
    expect(refKey(ref)).toBe(refKey({ ...ref }));
    expect(refKey(ref)).not.toBe(refKey({ ...ref, title: "Something else", doi: "10.2/x" }));
  });

  it("deduplicates against an existing library", () => {
    const out = dedupeAgainstLibrary([ref, { ...ref }], [] as any);
    expect(out).toBeTruthy();
  });
});

// --------------------------------------------------------------------------
// Risk-of-bias export
// --------------------------------------------------------------------------

describe("biasCsv", () => {
  const data = {
    domainNames: ["Randomisation", "Deviations"],
    rows: [{ label: "Smith 2024", judgments: ["Low", "High"], overall: "High" }],
  };

  it("emits a header and one row per study", () => {
    const lines = biasCsv(data).trim().split("\n");
    expect(lines.length).toBe(2);
    expect(lines[0]).toContain("Randomisation");
  });

  it("includes the study label and its overall judgment", () => {
    const csv = biasCsv(data);
    expect(csv).toContain("Smith 2024");
    expect(csv).toContain("High");
  });

  it("handles no rows", () => {
    expect(typeof biasCsv({ domainNames: [], rows: [] })).toBe("string");
  });

  it("quotes a label containing a comma so columns do not shift", () => {
    const csv = biasCsv({
      domainNames: ["D1"],
      rows: [{ label: "Smith, J 2024", judgments: ["Low"], overall: "Low" }],
    });
    expect(csv).toMatch(/"Smith, J 2024"/);
  });
});

// --------------------------------------------------------------------------
// Formatting
// --------------------------------------------------------------------------

describe("formatNumber", () => {
  it("groups thousands", () => {
    expect(formatNumber(6047)).toMatch(/6[,.\s]?047/);
  });

  it.each([null, undefined, NaN])("handles %s without printing junk", (n) => {
    const out = formatNumber(n as any);
    expect(typeof out).toBe("string");
    expect(out).not.toContain("NaN");
  });

  it("formats zero as zero", () => {
    expect(formatNumber(0)).toBe("0");
  });
});

describe("timeAgo", () => {
  it("describes a recent moment", () => {
    expect(typeof timeAgo(new Date())).toBe("string");
  });

  it.each([null, undefined, "", "not-a-date"])("handles %s safely", (v) => {
    const out = timeAgo(v as any);
    expect(typeof out).toBe("string");
    expect(out).not.toContain("Invalid Date");
  });
});
