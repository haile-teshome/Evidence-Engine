// Splitting a pasted list of studies into lookups.
//
// Run: pnpm test
//
// The box says one per line, and a title is taken whole because titles have
// commas in them. But a reviewer pastes a row of DOIs out of an email, and
// that went to the server as one long query: five real DOIs came back as a
// single "none of those could be found".
import { describe, it, expect } from "vitest";

import { listQueries } from "../useStudyImport";
import { readsFullText } from "../importedAt";

describe("listQueries", () => {
  it("splits a comma-separated row of DOIs", () => {
    expect(listQueries(
      "10.1056/NEJMoa2001017, 10.1038/nature14539, 10.1001/jama.2016.17216, "
      + "10.1038/s41591-018-0300-7, 10.1016/S0140-6736(20)30183-5")).toEqual([
      "10.1056/NEJMoa2001017",
      "10.1038/nature14539",
      "10.1001/jama.2016.17216",
      "10.1038/s41591-018-0300-7",
      "10.1016/S0140-6736(20)30183-5",
    ]);
  });

  it("keeps one per line working", () => {
    expect(listQueries("10.1001/jama.2020.1234\n32150360\nEffect of X on Y: a trial"))
      .toEqual(["10.1001/jama.2020.1234", "32150360", "Effect of X on Y: a trial"]);
  });

  it("leaves a title with commas in one piece", () => {
    // The case that makes splitting on commas alone wrong.
    const title = "Aspirin, clopidogrel, or both for stroke: a randomised trial";
    expect(listQueries(title)).toEqual([title]);
  });

  it("takes the DOI out of a citation or a link", () => {
    expect(listQueries("Zhu et al. 2020. https://doi.org/10.1056/NEJMoa2001017."))
      .toEqual(["10.1056/NEJMoa2001017"]);
  });

  it("splits a row of PubMed ids", () => {
    expect(listQueries("32150360, 31978945; 26017442")).toEqual(["32150360", "31978945", "26017442"]);
  });

  it("asks for each identifier once", () => {
    expect(listQueries("10.1038/nature14539\n10.1038/nature14539")).toEqual(["10.1038/nature14539"]);
  });

  it("ignores blank lines and stray whitespace", () => {
    expect(listQueries("\n  10.1038/nature14539  \n\n")).toEqual(["10.1038/nature14539"]);
    expect(listQueries("")).toEqual([]);
  });
});

// A record looked up by DOI carries a title and an abstract, not the paper.
// Importing at a stage that reads the article used to leave the reviewer on an
// empty table pointing at a different stage.
describe("which stages fetch the article themselves", () => {
  it("fetches at the stages that read the full text", () => {
    for (const stage of ["full-text-screening", "table-extraction", "text-extraction",
                         "quality-assessment", "meta-analysis", "writing"] as const) {
      expect(readsFullText(stage)).toBe(true);
    }
  });

  it("leaves the earlier stages, and Acquisition's own queue, alone", () => {
    // Acquisition is the page whose job this is: it has its own queue and its
    // own cancel, and a second fetch behind it would fight with it.
    for (const stage of ["home", "abstract-screening", "full-text-acquisition",
                         "citation-snowball"] as const) {
      expect(readsFullText(stage)).toBe(false);
    }
  });
});
