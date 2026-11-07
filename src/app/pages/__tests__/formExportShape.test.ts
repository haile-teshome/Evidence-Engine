// The shape of the workbook a tabbed extraction form exports as.
//
// Run: pnpm test
//
// The form the reviewer designs is a workbook: tabs, each with its own columns
// and its own studies. What came out instead was a worksheet per study holding
// every field, which is not that form and cannot be read down a column: to
// compare sample sizes you had to open fourteen sheets.
import { describe, it, expect } from "vitest";

import { tabSheets } from "../TextExtractionPage";

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
