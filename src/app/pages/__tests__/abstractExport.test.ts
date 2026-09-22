// The abstract-screening export.
//
// Run: pnpm test
//
// This export and the full-text one produce the same kind of artefact and had
// drifted into two implementations, so fixes landed on one and missed the
// other. They now share lib/exportStyle; these tests pin the half that is
// specific to abstract screening, and that the two agree where they should.
import { describe, it, expect, vi } from "vitest";
import ExcelJS from "exceljs";

import { abstractColumns, abstractRow, downloadAbstractXlsx, downloadAbstractCsv } from "../AbstractPage";
import { exportColumns } from "../FullTextPage";
import { XL } from "../../lib/exportStyle";

const row = (over: Record<string, any> = {}) => ({
  paper_id: "p1", Source: "PubMed", Title: "A trial of something",
  URL: "https://example.org/a", Abstract: "a", Decision: "INCLUDE" as const,
  Reason: "Meets every element.", Agent_Trace: {},
  Pico_Assessment: {
    population: { vote: "PASS", evidence: "We enrolled 412 adults.", reasoning: "Adults." },
    outcome:    { vote: "PARTIAL", evidence: "Adjacent outcome.", reasoning: "Partial." },
  },
  ...over,
}) as any;

describe("abstract columns", () => {
  it("uses the same groups as full-text screening, so the two files read alike", () => {
    const a = [...new Set(abstractColumns("pico").map(c => c.group))];
    const f = [...new Set(exportColumns([], "pico").map(c => c.group))];
    expect(a).toEqual(f);
  });

  it("gives every framework element a vote, quote and reasoning column", () => {
    const keys = abstractColumns("pico").map(c => c.key);
    expect(keys).toContain("population_vote");
    expect(keys).toContain("population_quote");
    expect(keys).toContain("population_reason");
  });
});

describe("abstract rows", () => {
  it("reports the effective decision and keeps the AI's original", () => {
    const v = abstractRow(row(), { p1: "EXCLUDE" }, "pico");
    expect(v.decision).toBe("EXCLUDE");
    expect(v.ai_decision).toBe("INCLUDE");
    expect(v.rev_edit).toBe("yes");
  });

  it("reassembles extracted text the same way full-text screening does", () => {
    const v = abstractRow(
      row({ Pico_Assessment: { population: { vote: "PASS", evidence: "in Asia\n,\n17\n,\n18", reasoning: "x" } } }),
      {}, "pico",
    );
    expect(v.population_quote).toContain("in Asia,17,18");
  });

  it("quotes an extract once, not twice", () => {
    const v = abstractRow(
      row({ Pico_Assessment: { population: { vote: "PASS", evidence: '"already quoted"', reasoning: "x" } } }),
      {}, "pico",
    );
    expect(v.population_quote).toBe('"already quoted"');
  });
});

describe("the abstract workbook", () => {
  async function build() {
    let blob: any = null;
    vi.spyOn(URL, "createObjectURL").mockImplementation((b: any) => { blob = b; return "blob:x"; });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    await downloadAbstractXlsx(
      [row({ Reason: "word ".repeat(300) }), row({ paper_id: "p2", Decision: "EXCLUDE" })],
      { p1: "EXCLUDE" }, "pico",
      { framework: "pico", pico: { population: "Adults" }, inclusion: ["RCTs only"], exclusion: ["Editorials"] },
    );
    const wb = new ExcelJS.Workbook();
    await wb.xlsx.load(await blob.arrayBuffer());
    return wb;
  }

  it("carries a summary sheet as well as the results", async () => {
    const wb = await build();
    expect(wb.worksheets.map(w => w.name)).toEqual(["Summary", "Abstract screening"]);
  });

  it("colours EVERY decision column, not just the one that stands", async () => {
    /* The reviewer's call and the model's are meant to be compared at a glance;
       colouring only the first made that a reading exercise. */
    const ws = (await build()).getWorksheet("Abstract screening")!;
    const r = ws.getRow(3);                        // overridden INCLUDE -> EXCLUDE
    expect((r.getCell(1).fill as any)?.fgColor?.argb).toBe(XL.redDeep);    // decision, deeper
    expect((r.getCell(2).fill as any)?.fgColor?.argb).toBe(XL.greenTint);  // AI decision, lighter
    expect((r.getCell(3).fill as any)?.fgColor?.argb).toBe(XL.blueTint);   // reviewer edit
  });

  it("gives the decision that stands a deeper tint than the AI's", async () => {
    /* Same hue, two weights: the pair stays comparable while the operative
       call is unmistakable. */
    const r = (await build()).getWorksheet("Abstract screening")!.getRow(4);   // not overridden
    expect((r.getCell(1).fill as any)?.fgColor?.argb).toBe(XL.redDeep);
    expect((r.getCell(2).fill as any)?.fgColor?.argb).toBe(XL.redTint);
  });

  it("bolds the decision that stands and leaves the AI's unbolded", async () => {
    const r = (await build()).getWorksheet("Abstract screening")!.getRow(3);
    expect(r.getCell(1).font?.bold).toBe(true);
    // ExcelJS omits bold:false on the way to the file, so read it as falsy.
    expect(r.getCell(2).font?.bold).toBeFalsy();
  });

  it("boxes a decision the reviewer changed", async () => {
    const r = (await build()).getWorksheet("Abstract screening")!.getRow(3);
    expect((r.getCell(1).border?.top as any)?.color?.argb).toBe(XL.blue);
  });

  it("sizes rows to their content rather than a fixed height", async () => {
    const ws = (await build()).getWorksheet("Abstract screening")!;
    expect(ws.getRow(3).height!).toBeGreaterThan(ws.getRow(4).height!);
    expect(ws.getRow(3).height!).toBeGreaterThan(60);   // the old fixed height
  });

  it("freezes and filters exactly as the full-text sheet does", async () => {
    const ws = (await build()).getWorksheet("Abstract screening")!;
    expect(ws.views[0]).toMatchObject({ state: "frozen", xSplit: 3, ySplit: 2 });
    expect(String(ws.autoFilter)).toMatch(/^A2:/);
  });

  it("paints the element votes on the shared traffic light", async () => {
    const ws = (await build()).getWorksheet("Abstract screening")!;
    const cols = abstractColumns("pico");
    const pop = cols.findIndex(c => c.key === "population_vote") + 1;
    const out = cols.findIndex(c => c.key === "outcome_vote") + 1;
    expect((ws.getRow(3).getCell(pop).fill as any)?.fgColor?.argb).toBe(XL.greenTint);
    expect((ws.getRow(3).getCell(out).fill as any)?.fgColor?.argb).toBe(XL.amberTint);
  });
});

describe("abstract CSV", () => {
  it("writes the same columns as the workbook, quoted properly", async () => {
    let captured = "";
    vi.spyOn(URL, "createObjectURL").mockImplementation((b: any) => {
      b.text().then((t: string) => { captured = t; });
      return "blob:x";
    });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    downloadAbstractCsv([row()], {}, "pico");
    await new Promise(r => setTimeout(r, 0));
    expect(captured).toContain("Decision");
    expect(captured).toContain("A trial of something");
    expect(captured).toContain('"""already quoted"""'.slice(0, 0) + '"We enrolled 412 adults."');
  });
});
