// Tests for records imported part-way through a review.
//
// Run: pnpm test
//
// The cases worth having are the reporting ones. Seeding a stage is only
// acceptable because the seeded rows stay distinguishable: if that mark is lost,
// the app silently claims it screened papers it never saw, which would show up
// in a PRISMA diagram and an AI decision log that accompany a publication.
import { describe, it, expect } from "vitest";

import {
  ImportStage, STAGE_LABEL, needsAbstractRow, needsFullTextRow,
  importReason, isImported, countImported, excludeImported, IMPORT_REASON_PREFIX,
} from "../importedAt";

const ALL: ImportStage[] = [
  "home", "abstract-screening", "full-text-acquisition",
  "full-text-screening", "table-extraction", "text-extraction",
  "quality-assessment", "citation-snowball", "meta-analysis", "writing",
];

describe("which earlier stages get seeded", () => {
  // The rule changed deliberately: the stages are a menu, not a queue. A
  // reviewer who arrives with the studies they have already chosen should be
  // able to use whichever stage they need, so an upload seeds every row and is
  // visible everywhere at once. What keeps that from becoming a claim about
  // work nobody did is the mark on each row, tested further down, and the fact
  // that no stage counts a seeded row as its own output.
  it("seeds both rows at every stage, so an upload is usable at all of them", () => {
    for (const st of ALL) {
      expect(needsAbstractRow(st)).toBe(true);
      expect(needsFullTextRow(st)).toBe(true);
    }
  });

  it("never seeds a full-text row without an abstract row", () => {
    // Otherwise extraction would hold papers the full-text stage cannot see.
    for (const st of ALL) {
      if (needsFullTextRow(st)) expect(needsAbstractRow(st)).toBe(true);
    }
  });
});

describe("the reason recorded on a seeded row", () => {
  it("names the stage and says it was not screened here", () => {
    const r = importReason("full-text-screening");
    expect(r).toContain(IMPORT_REASON_PREFIX);
    expect(r).toContain(STAGE_LABEL["full-text-screening"]);
    expect(r.toLowerCase()).toContain("not screened");
  });

  it("has a label for every stage", () => {
    for (const st of ALL) {
      expect(STAGE_LABEL[st]).toBeTruthy();
      expect(importReason(st)).toContain(STAGE_LABEL[st]);
    }
  });
});

describe("telling imported records apart", () => {
  const screened = { paper_id: "a", Decision: "INCLUDE" as const };
  const imported = { paper_id: "b", Decision: "INCLUDE" as const, Imported: "full-text-screening" as ImportStage };

  it("recognises a seeded row", () => {
    expect(isImported(imported)).toBe(true);
  });

  it("does not mistake a genuinely screened row for one", () => {
    expect(isImported(screened)).toBe(false);
    expect(isImported(null)).toBe(false);
    expect(isImported(undefined)).toBe(false);
  });

  it("counts only the imported ones for PRISMA's other-sources box", () => {
    expect(countImported([screened, imported, imported])).toBe(2);
    expect(countImported([screened, screened])).toBe(0);
    expect(countImported(null)).toBe(0);
  });

  it("keeps imported rows out of the AI decision log", () => {
    // These were decided by the reviewer before they ever reached this app.
    const kept = excludeImported([screened, imported]);
    expect(kept).toHaveLength(1);
    expect(kept[0].paper_id).toBe("a");
  });

  it("leaves a list of purely screened rows untouched", () => {
    expect(excludeImported([screened, screened])).toHaveLength(2);
    expect(excludeImported(null)).toEqual([]);
  });
});
