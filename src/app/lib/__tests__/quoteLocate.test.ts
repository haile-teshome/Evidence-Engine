// Finding an extracted value's quote back in the paper.
//
// Run: pnpm test
//
// The quote is the only thing standing between a number in a spreadsheet and a
// number a model made up, so the reviewer has to be able to see it in the
// document. Exact string matching cannot do that: a PDF's text layer breaks
// words across lines, doubles spaces, and substitutes punctuation. Nor can
// matching be so loose that a quote which is NOT in the paper appears to be,
// because that is the case the whole mechanism exists to catch.
import { describe, it, expect } from "vitest";

import { locateQuote, quoteInContext, quotePosition } from "../quoteLocate";

const paper = [
  "Patients were grouped as follows: (1) use of levetiracetam, (2) use of lacosamide.",
  "The median OS of our cohort (n = 272) was 15.64 months.",
  "Interestingly, the methylation of the MGMT gene seemed to have a protective effect.",
].join("\n\n");

describe("locating a quote", () => {
  it("finds it where it is", () => {
    const hit = locateQuote(paper, "The median OS of our cohort (n = 272) was 15.64 months.");
    expect(hit).not.toBeNull();
    expect(paper.slice(hit!.start, hit!.end)).toContain("15.64 months");
  });

  it("survives the whitespace a text layer invents", () => {
    const hit = locateQuote(paper, "The  median   OS of our\ncohort (n = 272)");
    expect(hit).not.toBeNull();
    expect(paper.slice(hit!.start, hit!.end)).toContain("median OS");
  });

  it("survives typographic punctuation", () => {
    const text = "The cohort’s median OS — 15.64 months — was unchanged.";
    const hit = locateQuote(text, "The cohort's median OS - 15.64 months - was unchanged.");
    expect(hit).not.toBeNull();
  });

  it("matches the opening when the model ran on past the sentence", () => {
    const hit = locateQuote(
      paper,
      "Interestingly, the methylation of the MGMT gene seemed to have a protective effect "
      + "and this was significant in every subgroup we examined.");
    expect(hit).not.toBeNull();
    expect(paper.slice(hit!.start, hit!.end)).toContain("methylation of the MGMT gene");
  });

  it("finds a quote in a text layer that lost its spaces", () => {
    const squashed = "Sampleharacteristics Sex:Male167(61.4%) Age(years)63(55-71)";
    const hit = locateQuote(squashed, "Sex: Male 167 (61.4%)");
    expect(hit).not.toBeNull();
    expect(squashed.slice(hit!.start, hit!.end)).toContain("167");
  });

  it("does not find a quote that is not in the paper", () => {
    // The case the whole mechanism exists for: an invented sentence.
    expect(locateQuote(paper, "Patients taking levetiracetam lived twice as long.")).toBeNull();
  });

  it("refuses to match on a fragment too short to mean anything", () => {
    expect(locateQuote(paper, "was")).toBeNull();
    expect(locateQuote(paper, "")).toBeNull();
  });
});

describe("showing it in context", () => {
  it("returns the surrounding passage with the quote in the middle", () => {
    const got = quoteInContext(paper, "was 15.64 months", 40);
    expect(got.found).toBe(true);
    expect(got.match).toBe("was 15.64 months");
    expect(got.before).toContain("median OS");
    expect(got.after.length).toBeGreaterThan(0);
  });

  it("says plainly when the quote is not there", () => {
    const got = quoteInContext(paper, "Patients taking levetiracetam lived twice as long.");
    expect(got.found).toBe(false);
    expect(got.match).toBe("Patients taking levetiracetam lived twice as long.");
    expect(got.before).toBe("");
  });

  it("marks where in the document the quote sits", () => {
    expect(quotePosition(paper, "Patients were grouped as follows")).toBe(0);
    expect(quotePosition(paper, "methylation of the MGMT gene")).toBeGreaterThan(50);
    expect(quotePosition(paper, "nowhere in this paper at all")).toBeNull();
  });
});
