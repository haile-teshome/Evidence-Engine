// The reading view may change how extracted text *looks*, never what it says
// or where it is. These tests are mostly about the second half of that: an
// evidence quote is a pair of character offsets into the raw text, and the
// reflow has to leave every one of them pointing at the same character.
import { describe, it, expect } from "vitest";

import {
  toReadingBlocks,
  markRanges,
  blockAt,
  sectionAt,
  HEADING_RE,
  type ReadingBlock,
} from "../readingText";

/** The invariant the whole file rests on: a displayed character that claims a
 *  raw offset is the character at that offset. */
function assertOffsetsHold(raw: string, blocks: ReadingBlock[]) {
  for (const b of blocks) {
    expect(b.map.length).toBe(b.text.length);
    for (let i = 0; i < b.text.length; i++) {
      const at = b.map[i];
      if (at < 0) {
        // Only ever a separator this module inserted.
        expect(b.text[i]).toBe(" ");
        continue;
      }
      expect(raw[at]).toBe(b.text[i]);
    }
  }
}

const WRAPPED = [
  "Methods",
  "We randomised 240 adults to either arm. The primary",
  "outcome was change in HbA1c at 12 weeks, measured",
  "by a central laboratory.",
  "",
  "Results",
  "Mean HbA1c fell by 0.8% (95% CI 0.6 to 1.0) in the",
  "intervention arm.",
].join("\n");

describe("toReadingBlocks", () => {
  it("returns nothing for empty or blank text", () => {
    expect(toReadingBlocks("")).toEqual([]);
    expect(toReadingBlocks("   \n\n  \n")).toEqual([]);
  });

  it("joins hard-wrapped lines into one paragraph", () => {
    const blocks = toReadingBlocks(WRAPPED);
    const paras = blocks.filter(b => b.kind === "p");
    expect(paras[0].text).toBe(
      "We randomised 240 adults to either arm. The primary outcome was change in HbA1c at 12 weeks, measured by a central laboratory.",
    );
  });

  it("sets the section names as headings", () => {
    const blocks = toReadingBlocks(WRAPPED);
    expect(blocks.filter(b => b.kind === "heading").map(b => b.text)).toEqual(["Methods", "Results"]);
  });

  it("keeps every offset pointing at the same character", () => {
    assertOffsetsHold(WRAPPED, toReadingBlocks(WRAPPED));
  });

  it("drops a trailing colon from a heading without disturbing the rest", () => {
    const raw = "Results:\nThe effect held.";
    const blocks = toReadingBlocks(raw);
    expect(blocks[0]).toMatchObject({ kind: "heading", text: "Results" });
    assertOffsetsHold(raw, blocks);
  });

  it("does not promote a line that merely looks like a title", () => {
    expect(HEADING_RE.test("Smith J, Patel R, Okafor N")).toBe(false);
    expect(HEADING_RE.test("Table 2 Baseline characteristics")).toBe(false);
    expect(HEADING_RE.test("Methods")).toBe(true);
    expect(HEADING_RE.test("2. Materials and methods")).toBe(true);
  });

  it("rejoins a word broken across the line break", () => {
    const raw = "Patients were random-\nised to treatment.";
    const blocks = toReadingBlocks(raw);
    expect(blocks[0].text).toBe("Patients were randomised to treatment.");
    assertOffsetsHold(raw, blocks);
  });

  it("leaves a real hyphen alone", () => {
    const raw = "This was a double-blind trial\nof two arms.";
    expect(toReadingBlocks(raw)[0].text).toBe("This was a double-blind trial of two arms.");
  });

  it("pulls a stranded citation group back together", () => {
    const raw = "Earlier work disagreed [\n1\n,\n3\n] on this point.";
    const blocks = toReadingBlocks(raw);
    expect(blocks[0].text).toBe("Earlier work disagreed [1,3] on this point.");
    assertOffsetsHold(raw, blocks);
  });

  it("leaves a bracketed phrase that is not a citation alone", () => {
    const raw = "As shown [see Table 1] the effect held.";
    expect(toReadingBlocks(raw)[0].text).toBe("As shown [see Table 1] the effect held.");
  });

  it("closes up punctuation and brackets a wrap left floating", () => {
    const raw = "The result\n, which was significant ( 95% CI 0.6 to 1.0 ) , held.";
    expect(toReadingBlocks(raw)[0].text).toBe(
      "The result, which was significant (95% CI 0.6 to 1.0), held.",
    );
  });

  it("collapses the gutter of a two-column layout to one space", () => {
    const raw = "The first column     and its continuation.";
    expect(toReadingBlocks(raw)[0].text).toBe("The first column and its continuation.");
  });

  it("breaks a page-long paragraph into several, losing nothing", () => {
    const sentence = "The intervention arm showed a sustained reduction in the primary outcome. ";
    const raw = sentence.repeat(12).trim();
    const blocks = toReadingBlocks(raw);
    expect(blocks.length).toBeGreaterThan(1);
    expect(blocks.every(b => b.kind === "p")).toBe(true);
    // Nothing added, nothing dropped: the whitespace a cut falls on stays with
    // the block before it, so the pieces concatenate back to the original.
    expect(blocks.map(b => b.text).join("")).toBe(raw);
    assertOffsetsHold(raw, blocks);
  });

  it("does not break on an abbreviation", () => {
    const long = "Trials run in the U.S. Food and Drug Administration era were included. ".repeat(9).trim();
    for (const b of toReadingBlocks(long)) {
      expect(b.text.startsWith("Food")).toBe(false);
    }
  });

  it("reflows a single page-long line, which is what the importer produces", () => {
    const raw = "Abstract\n" + "We assessed the effect of the programme on readmission. ".repeat(14).trim();
    const blocks = toReadingBlocks(raw);
    expect(blocks[0].kind).toBe("heading");
    expect(blocks.filter(b => b.kind === "p").length).toBeGreaterThan(1);
    assertOffsetsHold(raw, blocks);
  });

  it("survives a CRLF text layer without shifting offsets", () => {
    const raw = "Methods\r\nWe randomised 240 adults\r\nto either arm.";
    const blocks = toReadingBlocks(raw);
    expect(blocks.find(b => b.kind === "p")?.text).toBe("We randomised 240 adults to either arm.");
    assertOffsetsHold(raw, blocks);
  });

  it("records the raw range each block covers", () => {
    const raw = "Methods\nWe randomised adults.";
    const [heading, para] = toReadingBlocks(raw);
    expect(raw.slice(heading.start, heading.end)).toBe("Methods");
    expect(raw.slice(para.start, para.end)).toBe("We randomised adults.");
  });
});

describe("markRanges", () => {
  const raw = "We randomised 240 adults to either arm.\nThe outcome was HbA1c.";
  const blocks = toReadingBlocks(raw);

  it("marks nothing when there are no spans", () => {
    expect(markRanges(blocks[0], [])).toEqual([]);
  });

  it("marks nothing for a span in another block", () => {
    const other = toReadingBlocks("Methods\nSomething else entirely.");
    expect(markRanges(other[0], [{ start: 400, end: 420 }])).toEqual([]);
  });

  it("marks the display characters a raw span covers", () => {
    const quote = "240 adults";
    const start = raw.indexOf(quote);
    const [m] = markRanges(blocks[0], [{ start, end: start + quote.length }]);
    expect(blocks[0].text.slice(m.from, m.to)).toBe(quote);
  });

  it("marks a quote that runs across the rejoined line break as one range", () => {
    const start = raw.indexOf("either arm");
    const end = raw.indexOf("outcome") + "outcome".length;
    const ranges = markRanges(blocks[0], [{ start, end }]);
    expect(ranges).toHaveLength(1);
    expect(blocks[0].text.slice(ranges[0].from, ranges[0].to))
      .toBe("either arm. The outcome");
  });

  it("keeps two separate quotes separate, and says which is which", () => {
    const a = raw.indexOf("randomised");
    const b = raw.indexOf("HbA1c");
    const ranges = markRanges(blocks[0], [
      { start: a, end: a + "randomised".length },
      { start: b, end: b + "HbA1c".length },
    ]);
    expect(ranges.map(r => r.spanIndex)).toEqual([0, 1]);
    expect(blocks[0].text.slice(ranges[0].from, ranges[0].to)).toBe("randomised");
    expect(blocks[0].text.slice(ranges[1].from, ranges[1].to)).toBe("HbA1c");
  });

  it("ignores an empty or inverted span", () => {
    expect(markRanges(blocks[0], [{ start: 5, end: 5 }])).toEqual([]);
    expect(markRanges(blocks[0], [{ start: 9, end: 2 }])).toEqual([]);
  });

  it("gives an overlapping pair to the first span, without dropping either", () => {
    const start = raw.indexOf("randomised");
    const ranges = markRanges(blocks[0], [
      { start, end: start + 10 },
      { start: start + 4, end: start + 16 },
    ]);
    expect(ranges.map(r => r.spanIndex)).toEqual([0, 1]);
    expect(ranges[0].to).toBe(ranges[1].from);
  });
});

describe("blockAt and sectionAt", () => {
  const blocks = toReadingBlocks(WRAPPED);

  it("finds the block holding an offset", () => {
    const at = WRAPPED.indexOf("central laboratory");
    const i = blockAt(blocks, at);
    expect(i).toBeGreaterThanOrEqual(0);
    expect(blocks[i].text).toContain("central laboratory");
  });

  it("reports -1 for an offset in no block", () => {
    expect(blockAt(blocks, WRAPPED.length + 50)).toBe(-1);
  });

  it("names the section a block sits under", () => {
    const at = WRAPPED.indexOf("Mean HbA1c");
    expect(sectionAt(blocks, blockAt(blocks, at))).toBe("Results");
  });

  it("returns nothing above the first heading", () => {
    expect(sectionAt(toReadingBlocks("Just a sentence."), 0)).toBe("");
  });
});
