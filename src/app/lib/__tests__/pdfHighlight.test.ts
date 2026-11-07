// Marking a quoted sentence on the page of the PDF it came from.
//
// Run: pnpm test
//
// pdf.js gives a page's text as positioned fragments, split wherever the
// typesetter changed font or kerned a pair of letters, so "levetiracetam" can
// arrive as three pieces. The sentence has to be found across those pieces and
// mapped back to the ones it covered, or the highlight lands on the wrong words.
import { describe, it, expect } from "vitest";

import { pageText, matchItems, itemBox, mergeBoxes } from "../pdfHighlight";

/** A fragment at (x, y), 10pt tall, as pdf.js would hand it over. */
const frag = (str: string, x = 0, y = 700, width = str.length * 5) =>
  ({ str, transform: [10, 0, 0, 10, x, y], width, height: 10 });

describe("a page's text", () => {
  it("joins the fragments with the spaces the typesetter left out", () => {
    const { text } = pageText([frag("The median"), frag("OS was"), frag("15.64 months.")]);
    expect(text).toBe("The median OS was 15.64 months. ");
  });

  it("does not double a space a fragment already ends with", () => {
    const { text } = pageText([frag("The median "), frag("OS")]);
    expect(text).toBe("The median OS ");
  });
});

describe("matching a quote to fragments", () => {
  const items = [
    frag("Patients were grouped", 60, 700),
    frag("as follows:", 200, 700),
    frag("The median OS of our cohort", 60, 680),
    frag("(n = 272) was 15.64 months.", 60, 660),
    frag("Both drugs were tolerated.", 60, 640),
  ];

  it("covers every fragment the sentence runs through", () => {
    const hits = matchItems(items, "The median OS of our cohort (n = 272) was 15.64 months.");
    expect(hits.map(h => h.index)).toEqual([2, 3]);
  });

  it("covers a fragment holding only part of a word", () => {
    const split = [frag("leve", 60, 700), frag("tiracetam was used", 90, 700)];
    const hits = matchItems(split, "levetiracetam was used");
    expect(hits.map(h => h.index)).toEqual([0, 1]);
  });

  it("finds nothing on a page the sentence is not on", () => {
    expect(matchItems(items, "Atorvastatin reversed the effect.")).toEqual([]);
  });

  it("finds nothing for an empty quote", () => {
    expect(matchItems(items, "   ")).toEqual([]);
  });
});

describe("where to draw the mark", () => {
  // The viewport pdf.js renders with at scale 1, for a 792pt-tall page: y is
  // flipped, because PDF counts up from the bottom and a canvas counts down.
  const viewport = [1, 0, 0, -1, 0, 792];

  it("puts the box over the words, not under them", () => {
    const box = itemBox(frag("median OS", 60, 700, 50), viewport, 1);
    expect(box.left).toBe(60);
    expect(box.top).toBe(792 - 700 - 10);   // page height, minus baseline, minus the font
    expect(box.height).toBe(10);
    expect(box.width).toBe(50);
  });

  it("scales with the page", () => {
    const box = itemBox(frag("median OS", 60, 700, 50), [2, 0, 0, -2, 0, 1584], 2);
    expect(box.left).toBe(120);
    expect(box.width).toBe(100);
    expect(box.height).toBe(20);
  });
});

describe("merging the boxes", () => {
  it("joins fragments on one line into a single mark", () => {
    const merged = mergeBoxes([
      { left: 60, top: 100, width: 40, height: 10 },
      { left: 102, top: 100, width: 50, height: 10 },
    ]);
    expect(merged).toHaveLength(1);
    expect(merged[0].left).toBe(60);
    expect(merged[0].width).toBe(92);
  });

  it("keeps separate lines separate", () => {
    const merged = mergeBoxes([
      { left: 60, top: 100, width: 40, height: 10 },
      { left: 60, top: 120, width: 40, height: 10 },
    ]);
    expect(merged).toHaveLength(2);
  });

  it("keeps a gap across the page gutter separate", () => {
    // Two columns, same line: one mark spanning the gutter would cover the
    // wrong half of the page.
    const merged = mergeBoxes([
      { left: 60, top: 100, width: 40, height: 10 },
      { left: 320, top: 100, width: 40, height: 10 },
    ]);
    expect(merged).toHaveLength(2);
  });
});

describe("when a page hands back something odd", () => {
  // A text item with no transform used to throw: array destructuring calls the
  // iterator, and in Safari that reads as "undefined is not a function", which
  // then surfaced as a raw error message in the reviewer's way.
  it("gives a box for an item with no transform rather than throwing", () => {
    const box = itemBox({ str: "x", width: 20, height: 10 } as any, [1, 0, 0, -1, 0, 792], 1);
    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  });

  it("gives a box when the viewport matrix is missing", () => {
    const box = itemBox(
      { str: "x", width: 20, height: 10, transform: [10, 0, 0, 10, 60, 700] },
      undefined as any, 1);
    expect(Number.isFinite(box.left)).toBe(true);
  });
});
