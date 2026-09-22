// The full-text export, both formats.
//
// Run: pnpm test
//
// An export bug is only noticed downstream, in Excel or in someone's analysis
// script, long after the run it describes. The one that mattered here: every
// row was pinned to 78 points, so a long evidence quote was written to the file
// and invisible on the page. Nothing errored, and the cell looked empty.
import { describe, it, expect, vi, beforeEach } from "vitest";
import ExcelJS from "exceljs";

import {
  rowHeightFor, exportColumns, exportRow, downloadFullTextCsv, downloadFullTextXlsx,
  tidyEvidence,
} from "../FullTextPage";

const row = (over: Record<string, any> = {}) => ({
  paper_id: "p1", Title: "A trial of something", Source: "PubMed",
  URL: "https://example.org/a", Decision: "Include" as const,
  Reason: "Meets every criterion.", inclusion_score: "3/3", exclusion_violations: "0",
  picoEvidence: {
    population: { match: "yes", value: "Adults", evidence: "We enrolled 412 adults." },
  },
  criteriaEval: { "RCTs only": "INCLUDE" },
  criteriaEvidence: { "RCTs only": { evidence: "randomised, double blind", reasoning: "Stated in the methods." } },
  ...over,
}) as any;

describe("row height", () => {
  const cols = exportColumns(["RCTs only"], "pico");

  it("grows with the longest cell instead of a fixed height", () => {
    const short = rowHeightFor({ reason: "Short." }, cols);
    const long = rowHeightFor({ reason: "word ".repeat(400) }, cols);
    expect(long).toBeGreaterThan(short);
  });

  it("gives a long quote enough room that it is not cut off", () => {
    /* 1,200 characters in a 60-wide column is roughly 17 lines. At the old
       fixed 78 points only about six of them were visible. */
    const h = rowHeightFor({ reason: "x".repeat(1200) }, cols);
    expect(h).toBeGreaterThan(160);
  });

  it("counts explicit newlines, which the criteria cells are full of", () => {
    const oneLine = rowHeightFor({ reason: "a" }, cols);
    const manyLines = rowHeightFor({ reason: "a\nb\nc\nd\ne\nf\ng\nh" }, cols);
    expect(manyLines).toBeGreaterThan(oneLine * 4);
  });

  it("never exceeds Excel's maximum row height", () => {
    /* Past 409 points Excel rejects the sheet outright. */
    expect(rowHeightFor({ reason: "x".repeat(200000) }, cols)).toBeLessThanOrEqual(409);
  });

  it("keeps a minimum height for an empty row", () => {
    expect(rowHeightFor({}, cols)).toBeGreaterThanOrEqual(22);
  });
});

describe("column layout", () => {
  it("groups the columns so the sheet has a readable structure", () => {
    const groups = [...new Set(exportColumns(["RCTs only"], "pico").map(c => c.group))];
    expect(groups.slice(0, 3)).toEqual(["Decision", "Article", "Assessment"]);
    expect(groups).toContain("Criteria");
  });

  it("gives every framework element its own match, value and evidence columns", () => {
    const keys = exportColumns([], "pico").map(c => c.key);
    expect(keys).toContain("population_match");
    expect(keys).toContain("population_value");
    expect(keys).toContain("population_evidence");
  });

  it("adds one column per criterion", () => {
    const cols = exportColumns(["RCTs only", "English language"], "pico");
    expect(cols.filter(c => c.group === "Criteria").map(c => c.header))
      .toEqual(["RCTs only", "English language"]);
  });
});

describe("row values", () => {
  it("reports the effective decision and keeps the AI's original", () => {
    const v = exportRow(row(), ["RCTs only"], { p1: "Exclude" }, "pico");
    expect(v.decision).toBe("Exclude");
    expect(v.ai_decision).toBe("Include");
    expect(v.rev_edit).toBe("yes");
  });

  it("marks nothing as edited when the reviewer agreed", () => {
    const v = exportRow(row(), ["RCTs only"], { p1: "Include" }, "pico");
    expect(v.rev_edit).toBe("");
  });

  it("puts the vote, the quote and the reasoning on separate lines", () => {
    const v = exportRow(row(), ["RCTs only"], {}, "pico");
    const lines = String(v.crit_0).split("\n").filter(Boolean);
    expect(lines[0]).toBe("INCLUDE");
    expect(v.crit_0).toContain("randomised, double blind");
    expect(v.crit_0).toContain("Stated in the methods.");
  });
});

describe("CSV", () => {
  let captured = "";
  beforeEach(() => {
    captured = "";
    vi.spyOn(URL, "createObjectURL").mockImplementation((b: any) => {
      b.text().then((t: string) => { captured = t; });
      return "blob:x";
    });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  });

  const run = async () => {
    downloadFullTextCsv([row()], ["RCTs only"], {}, "pico");
    await new Promise(r => setTimeout(r, 0));
    return captured;
  };

  it("writes a header and one row per article", async () => {
    const csv = await run();
    expect(csv).toContain("Decision");
    expect(csv).toContain("A trial of something");
  });

  it("quotes cells containing newlines, which every evidence cell has", async () => {
    /* Unquoted, one article's evidence becomes several broken rows. */
    const csv = await run();
    expect(csv).toMatch(/"INCLUDE\n\n/);
  });

  it("doubles embedded quotes so the file still parses", async () => {
    const csv = await run();
    expect(csv).toContain('""randomised, double blind""');
  });

  it("starts with a byte order mark so Excel reads it as UTF-8", async () => {
    /* Read the BYTES: Blob.text() decodes as UTF-8 and strips the BOM, so the
       decoded string can never show it. Without the BOM Excel opens the file
       in the system codepage and mangles every quoted phrase. */
    let blob: Blob | null = null;
    vi.spyOn(URL, "createObjectURL").mockImplementation((b: any) => { blob = b; return "blob:x"; });
    downloadFullTextCsv([row()], ["RCTs only"], {}, "pico");
    const bytes = new Uint8Array(await blob!.arrayBuffer());
    expect([bytes[0], bytes[1], bytes[2]]).toEqual([0xef, 0xbb, 0xbf]);
  });

  it("uses the same columns as the workbook", async () => {
    const csv = await run();
    const header = csv.replace(/^﻿/, "").split("\r\n")[0];
    const expected = exportColumns(["RCTs only"], "pico").length;
    // Count top-level commas, ignoring those inside quotes.
    let depth = 0, fields = 1;
    for (const ch of header) {
      if (ch === '"') depth ^= 1;
      else if (ch === "," && !depth) fields++;
    }
    expect(fields).toBe(expected);
  });
});

// --------------------------------------------------------------------------
// The workbook itself, generated and read back.
//
// The unit tests above check the helpers; this checks the file. It is the only
// place the clipping bug was ever observable: the text was written correctly,
// and the row it sat in was too short to show it.
// --------------------------------------------------------------------------

describe("the generated workbook", () => {
  // Parenthesised deliberately: .repeat binds to the last literal otherwise,
  // which quietly makes the "long" fixture short.
  const longReason = ("The study develops and externally validates a gradient-boosted "
    + "model on routinely collected dental records, and reports discrimination "
    + "and calibration at 24 months. ").repeat(4);

  async function build() {
    let blob: any = null;
    vi.spyOn(URL, "createObjectURL").mockImplementation((b: any) => { blob = b; return "blob:x"; });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    await downloadFullTextXlsx(
      [
        row({ Reason: longReason }),
        // Genuinely short: the criteria cell is normally the tallest thing in a
        // row, so shortening only the reason would leave both rows the same.
        row({ paper_id: "p2", Reason: "Short.", criteriaEvidence: {}, picoEvidence: {} }),
      ] as any,
      ["RCTs only"], { p1: "Exclude" }, "pico",
      {
        framework: "pico",
        pico: { population: "Adults with dental records", intervention: "ML caries model",
                comparator: "Clinician alone", outcome: "Caries at 24 months" },
        inclusion: ["Reports a validated model"], exclusion: ["Conference abstracts"],
      },
    );
    const wb = new ExcelJS.Workbook();
    await wb.xlsx.load(await blob.arrayBuffer());
    return wb;
  }

  it("carries a summary sheet as well as the results", async () => {
    const wb = await build();
    expect(wb.worksheets.map(w => w.name)).toEqual(["Summary", "Full-text screening"]);
  });

  it("records the question and criteria the decisions were made against", async () => {
    /* A bare grid does not say what it was judged by, which is the first thing
       anyone opening the file months later needs. */
    const text = JSON.stringify(wsValues(await build(), "Summary"));
    expect(text).toContain("Adults with dental records");
    expect(text).toContain("Reports a validated model");
    expect(text).toContain("Conference abstracts");
  });

  it("makes a row with a long quote taller than one with a short one", async () => {
    const ws = (await build()).getWorksheet("Full-text screening")!;
    const long = ws.getRow(3).height!;
    const short = ws.getRow(4).height!;
    expect(long).toBeGreaterThan(short);
    // The old export pinned every row to 78 and cut the rest off.
    expect(long).toBeGreaterThan(78);
  });

  it("wraps every data cell, so nothing runs out of its column", async () => {
    const ws = (await build()).getWorksheet("Full-text screening")!;
    const reason = ws.getRow(3).getCell(7);
    expect(reason.alignment?.wrapText).toBe(true);
    expect(String(reason.value)).toContain("calibration at 24 months");
  });

  it("freezes the decision block and both header rows", async () => {
    const ws = (await build()).getWorksheet("Full-text screening")!;
    expect(ws.views[0]).toMatchObject({ state: "frozen", xSplit: 3, ySplit: 2 });
  });

  it("filters from the header row, not the group band above it", async () => {
    const ws = (await build()).getWorksheet("Full-text screening")!;
    expect(String(ws.autoFilter)).toMatch(/^A2:/);
  });

  it("bands alternate rows and colours the decision", async () => {
    const ws = (await build()).getWorksheet("Full-text screening")!;
    expect((ws.getRow(4).getCell(4).fill as any)?.fgColor?.argb).toBe("FFF7F8F9");
    // The operative decision takes the deeper tint; the AI's keeps the lighter.
    expect((ws.getRow(3).getCell(1).fill as any)?.fgColor?.argb).toBe("FFFBCFCF");
    expect((ws.getRow(3).getCell(2).fill as any)?.fgColor?.argb).toBe("FFE8F0EA");
  });

  it("keeps the verdicts on a traffic light and never colours by anything else", async () => {
    /* A reviewer scans a screening sheet for red. The structure around it is
       green and neutral; only a verdict is allowed to be red or amber. */
    const wb = await build();
    const allowed = new Set([
      "FFE8F0EA", "FFFEE2E2", "FFFEF3C7",   // include / exclude / partial
      "FFD4E4DA", "FFFBCFCF",               // the verdict that stands, a shade deeper
      "FFE7EDF7",                           // a reviewer's own edit
      "FF14532D", "FFEDF0F2", "FFF7F8F9",   // group band, header, row banding
    ]);
    for (const ws of wb.worksheets) {
      ws.eachRow(r => r.eachCell(c => {
        const argb = (c.fill as any)?.fgColor?.argb;
        if (argb) expect(allowed, `unexpected fill ${argb}`).toContain(argb);
      }));
    }
  });
});

/** Every cell value on a sheet, for assertions that do not care where it sits. */
function wsValues(wb: ExcelJS.Workbook, name: string): any[] {
  const ws = wb.getWorksheet(name)!;
  const out: any[] = [];
  ws.eachRow(r => r.eachCell(c => out.push(c.value)));
  return out;
}

// --------------------------------------------------------------------------
// Reassembling extracted PDF text.
//
// pypdf breaks a paragraph wherever the printed page did, so a quote arrives
// with commas, bullet glyphs and superscript citation numbers each stranded on
// a line of their own. Exported verbatim, a cell became a column of punctuation
// with the sentence lost inside it.
// --------------------------------------------------------------------------

describe("tidyEvidence", () => {
  it("pulls a stranded comma back onto the line it belongs to", () => {
    expect(tidyEvidence("conducted in Asia\n,\nincluding China"))
      .toBe("conducted in Asia,\nincluding China");
  });

  it("reassembles a run of citation numbers into one line", () => {
    /* The worst case in a real export: seventeen lines showing "17", ",",
       "18", ",", each on its own row of a cell. */
    expect(tidyEvidence("including China (n = 10),\n17\n,\n18\n,\n19\n,\n20"))
      .toBe("including China (n = 10),17,18,19,20");
  });

  it("attaches a lone bullet to the item it introduces", () => {
    expect(tidyEvidence("Questionnaires\n•\nElectronic records\n•\nClinical data"))
      .toBe("Questionnaires\n• Electronic records\n• Clinical data");
  });

  it("treats a lone dash as a bullet too", () => {
    expect(tidyEvidence("Methods\n-\nRandom Forest")).toBe("Methods\n• Random Forest");
  });

  it("joins a fragment that opens with punctuation", () => {
    expect(tidyEvidence("the United States (n = 6\n).")).toBe("the United States (n = 6).");
  });

  it("drops the blank lines the extractor leaves behind", () => {
    expect(tidyEvidence("First line\n\n\n\nSecond line")).toBe("First line\nSecond line");
  });

  it("leaves ordinary prose exactly as it was", () => {
    /* It must not rewrite a quote: the evidence has to stay quotable. */
    const prose = "Despite the clear demand for oral disease prevention, the transition\n"
      + "from a traditional, treatment-focused model is impeded by barriers.";
    expect(tidyEvidence(prose)).toBe(prose);
  });

  it("does not swallow a number that starts a real line", () => {
    /* A digit after a full stop begins a new sentence, not a footnote marker. */
    expect(tidyEvidence("Enrolment closed.\n2021 was the final year."))
      .toBe("Enrolment closed.\n2021 was the final year.");
  });

  it("survives empty and whitespace-only input", () => {
    expect(tidyEvidence("")).toBe("");
    expect(tidyEvidence("   \n \n ")).toBe("");
  });

  it("is applied to the evidence that reaches the export", () => {
    const v = exportRow(
      row({ criteriaEvidence: { "RCTs only": { evidence: "in Asia\n,\n17\n,\n18", reasoning: "Stated." } } }),
      ["RCTs only"], {}, "pico",
    );
    expect(v.crit_0).toContain("in Asia,17,18");
    expect(v.crit_0).not.toMatch(/\n,\n/);
  });
});

describe("quoting the evidence", () => {
  it("does not double the quotation marks the extract already has", () => {
    const v = exportRow(
      row({ criteriaEvidence: { "RCTs only": { evidence: '"already quoted"', reasoning: "Stated." } } }),
      ["RCTs only"], {}, "pico",
    );
    expect(v.crit_0).toContain('"already quoted"');
    expect(v.crit_0).not.toContain('""');
  });

  it("still quotes an extract that carries none", () => {
    const v = exportRow(
      row({ criteriaEvidence: { "RCTs only": { evidence: "bare text", reasoning: "" } } }),
      ["RCTs only"], {}, "pico",
    );
    expect(v.crit_0).toContain('"bare text"');
  });
});
