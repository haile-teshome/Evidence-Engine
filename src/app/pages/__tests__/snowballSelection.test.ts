// How snowballing decides which citations survive the per-seed cap.
//
// Run: pnpm test
//
// This used to have no rule at all: each API returned page one in its own
// order and the app kept the first N. For a bibliography that means the
// references printed earliest, which skew to introduction and background
// rather than to the closest work, and nothing recorded that a seed with 85
// references had contributed only 20.
import { describe, it, expect } from "vitest";
import { sortCitations } from "../SnowballPage";

const rows = [
  { title: "Alpha",   year: 2001, cited_by_count: 5 },
  { title: "Bravo",   year: 2020, cited_by_count: 1 },
  { title: "Charlie", year: 2010, cited_by_count: 99 },
];

describe("choosing which citations to keep", () => {
  it("leaves the source order untouched when that is what was asked for", () => {
    expect(sortCitations(rows, "source").map(r => r.title)).toEqual(["Alpha", "Bravo", "Charlie"]);
  });

  it("orders by citation count, most cited first", () => {
    expect(sortCitations(rows, "cited").map(r => r.title)).toEqual(["Charlie", "Alpha", "Bravo"]);
  });

  it("orders by year, most recent first", () => {
    expect(sortCitations(rows, "recent").map(r => r.title)).toEqual(["Bravo", "Charlie", "Alpha"]);
  });

  it("orders by the relevance score when one was computed", () => {
    const scores = new Map([["alpha", 0.1], ["bravo", 0.9], ["charlie", 0.5]]);
    expect(sortCitations(rows, "relevance", scores).map(r => r.title))
      .toEqual(["Bravo", "Charlie", "Alpha"]);
  });

  it("sinks a paper the ranker never scored rather than dropping it", () => {
    /* A citation missing from the ranking is still a citation; it just should
       not outrank one that was actually judged relevant. */
    const scores = new Map([["alpha", 0.2]]);
    const out = sortCitations(rows, "relevance", scores);
    expect(out[0].title).toBe("Alpha");
    expect(out).toHaveLength(3);
  });

  it("falls back to source order when the field is missing", () => {
    const missing = [{ title: "One" }, { title: "Two", cited_by_count: 3 }];
    expect(sortCitations(missing, "cited").map(r => r.title)).toEqual(["Two", "One"]);
  });

  it("is stable, so equal values keep the source order and a run repeats", () => {
    const tied = [
      { title: "First",  year: 2020 },
      { title: "Second", year: 2020 },
      { title: "Third",  year: 2020 },
    ];
    expect(sortCitations(tied, "recent").map(r => r.title)).toEqual(["First", "Second", "Third"]);
  });

  it("never invents or loses a citation", () => {
    for (const sort of ["relevance", "cited", "recent", "source"] as const) {
      expect(sortCitations(rows, sort)).toHaveLength(rows.length);
    }
  });

  it("does not mutate the list it was given", () => {
    const before = rows.map(r => r.title);
    sortCitations(rows, "cited");
    expect(rows.map(r => r.title)).toEqual(before);
  });
});
