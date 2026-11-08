// The shape of the workbook a tabbed extraction form exports as.
//
// Run: pnpm test
//
// The form the reviewer designs is a workbook: tabs, each with its own columns
// and its own studies. What came out instead was a worksheet per study holding
// every field, which is not that form and cannot be read down a column: to
// compare sample sizes you had to open fourteen sheets.
import { describe, it, expect } from "vitest";

import { tabSheets, fieldsForStudy, NOT_REPORTED } from "../TextExtractionPage";

const fields = [
  { id: "author", label: "First author", type: "text", options: [] },
  { id: "n", label: "Sample size", type: "number", options: [] },
  { id: "os", label: "Median OS", type: "number", options: [] },
];

const row = (paper_id: string, title: string, values: Record<string, any>, status: any = {}) =>
  ({ paper_id, title, values, status });

describe("tabSheets", () => {
  const rows = [
    row("p1", "Bianconi 2024", { author: "Bianconi", n: 272, os: 15.64 }),
    row("p2", "Bhat 2021", { author: "Bhat", n: 10, os: "" }),
  ];

  it("gives each tab its own sheet, with that tab's columns", () => {
    const sheets = tabSheets([
      { name: "Identification", fieldIds: ["author"], studies: null },
      { name: "Survival", fieldIds: ["n", "os"], studies: null },
    ], fields as any, rows as any);

    expect(sheets.map(s => s.name)).toEqual(["Identification", "Survival"]);
    expect(sheets[0].head).toEqual(["Article", "First author"]);
    expect(sheets[1].head).toEqual(["Article", "Sample size", "Median OS"]);
  });

  it("puts one row per study, not one sheet per study", () => {
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["n", "os"], studies: null }], fields as any, rows as any);

    expect(sheet.body).toEqual([
      ["Bianconi 2024", "272", "15.64"],
      ["Bhat 2021", "10", ""],
    ]);
  });

  it("holds only the studies the tab covers", () => {
    // A tab for one study design holds that design's studies, which is exactly
    // what its study list said when the reviewer set it.
    const [sheet] = tabSheets(
      [{ name: "Clinical only", fieldIds: ["n"], studies: ["p1"] }], fields as any, rows as any);

    expect(sheet.body).toEqual([["Bianconi 2024", "272"]]);
  });

  it("leaves a rejected value out rather than exporting a value nobody kept", () => {
    const rejected = [row("p1", "Bianconi 2024", { n: 272 }, { n: "rejected" })];
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["n"], studies: null }], fields as any, rejected as any);

    expect(sheet.body).toEqual([["Bianconi 2024", ""]]);
  });

  it("names a tab that never got one", () => {
    const [sheet] = tabSheets(
      [{ name: "", fieldIds: ["n"], studies: null }], fields as any, rows as any);

    expect(sheet.name).toBe("Tab 1");
  });

  it("skips a tab whose fields have all been deleted", () => {
    const sheets = tabSheets([
      { name: "Empty", fieldIds: ["gone"], studies: null },
      { name: "Survival", fieldIds: ["n"], studies: null },
    ], fields as any, rows as any);

    expect(sheets.map(s => s.name)).toEqual(["Survival"]);
  });
});

// An empty cell meant two different things, and the reviewer could not tell
// which: the paper does not report this, or nothing ever read it. The first is
// a finding the review writes down; the second is work still to do. A blank
// column in an exported workbook reads as the second whichever it was.
describe("an empty cell in the export", () => {
  const rows = [
    { paper_id: "p1", title: "Bianconi 2024", values: { n: 272, os: "", author: "" },
      status: {}, cell: { n: "reported", os: "not_reported", author: "unread" } },
  ] as any;

  it("writes down a value the paper does not report", () => {
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["os"], studies: null }], fields as any, rows);
    expect(sheet.body).toEqual([["Bianconi 2024", NOT_REPORTED]]);
  });

  it("leaves a cell nothing read blank, so it still looks like work to do", () => {
    const [sheet] = tabSheets(
      [{ name: "Identification", fieldIds: ["author"], studies: null }], fields as any, rows);
    expect(sheet.body).toEqual([["Bianconi 2024", ""]]);
  });

  it("prefers the value whenever there is one", () => {
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["n"], studies: null }], fields as any, rows);
    expect(sheet.body).toEqual([["Bianconi 2024", "272"]]);
  });

  it("does not write 'not reported' over a value the reviewer rejected", () => {
    // Rejected means "this is wrong", not "the paper is silent".
    const rejected = [{ paper_id: "p1", title: "Bianconi 2024", values: { n: 272 },
                        status: { n: "rejected" }, cell: { n: "reported" } }] as any;
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["n"], studies: null }], fields as any, rejected);
    expect(sheet.body).toEqual([["Bianconi 2024", ""]]);
  });

  it("falls back to a blank for rows extracted before the status existed", () => {
    const old = [{ paper_id: "p1", title: "Bianconi 2024", values: { os: "" }, status: {} }] as any;
    const [sheet] = tabSheets(
      [{ name: "Survival", fieldIds: ["os"], studies: null }], fields as any, old);
    expect(sheet.body).toEqual([["Bianconi 2024", ""]]);
  });
});

// Asking a mouse paper for a clinical cohort's baseline table is how the
// preclinical sheet came back holding one. A tab scoped to some studies is
// already hidden for the others, so asking achieved nothing but a wrong value.
describe("fieldsForStudy", () => {
  const tabs = [
    { name: "Identification", fieldIds: ["author"], studies: null },
    { name: "Clinical", fieldIds: ["n"], studies: ["p1"] },
    { name: "Preclinical", fieldIds: ["os"], studies: ["p2"] },
  ];

  it("asks a study only the fields of the tabs that cover it", () => {
    expect(fieldsForStudy(tabs, fields as any, "p1").map(f => f.id))
      .toEqual(["author", "n"]);
    expect(fieldsForStudy(tabs, fields as any, "p2").map(f => f.id))
      .toEqual(["author", "os"]);
  });

  it("includes every unscoped tab for every study", () => {
    expect(fieldsForStudy(tabs, fields as any, "p9").map(f => f.id)).toEqual(["author"]);
  });

  it("asks the whole form when there are no tabs", () => {
    expect(fieldsForStudy(null, fields as any, "p1")).toHaveLength(fields.length);
    expect(fieldsForStudy([], fields as any, "p1")).toHaveLength(fields.length);
  });

  it("asks the whole form for a study no tab claims at all", () => {
    // Better than an empty row with nothing to explain it.
    const scoped = [{ name: "Clinical", fieldIds: ["n"], studies: ["p1"] }];
    expect(fieldsForStudy(scoped, fields as any, "p2")).toHaveLength(fields.length);
  });

  it("ignores a tab whose fields were deleted", () => {
    const stale = [{ name: "Gone", fieldIds: ["removed"], studies: ["p1"] },
                   { name: "Clinical", fieldIds: ["n"], studies: ["p1"] }];
    expect(fieldsForStudy(stale, fields as any, "p1").map(f => f.id)).toEqual(["n"]);
  });
});
