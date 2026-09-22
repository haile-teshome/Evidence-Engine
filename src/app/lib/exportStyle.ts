// ---------------------------------------------------------------------------
// Shared workbook styling for the screening exports.
//
// Abstract screening and full-text screening produce the same kind of artefact:
// one row per article, a verdict, and the evidence behind it. They had two
// separate implementations that had already drifted apart, so a fix to one
// (auto-fitting rows, reassembling extracted text, the palette) missed the
// other. Everything both need lives here; each page supplies only its own
// columns and values.
// ---------------------------------------------------------------------------

import type ExcelJS from "exceljs";

// A deep verdant green and formal neutrals for the structure, and the familiar
// traffic light for the verdicts. The structure used to be saturated mint and
// lime, which shouted on a page of a hundred rows; the verdicts stay red and
// amber because that is what a reviewer scans a screening sheet for.
export const XL = {
  green:      "FF14532D",   // group bands, headings, an included verdict
  greenTint:  "FFE8F0EA",   // an included cell
  greenDeep:  "FFD4E4DA",   // the included verdict that actually stands
  red:        "FF991B1B",   // an excluded verdict
  redTint:    "FFFEE2E2",   // an excluded cell
  redDeep:    "FFFBCFCF",   // the excluded verdict that actually stands
  amber:      "FF92400E",   // a partial match: neither in nor out
  amberTint:  "FFFEF3C7",
  blue:       "FF1E40AF",   // links, and a reviewer's own edit
  blueTint:   "FFE7EDF7",
  slate:      "FF475569",   // not assessed
  slateTint:  "FFF1F5F9",
  headerTint: "FFEDF0F2",   // the column header row
  band:       "FFF7F8F9",   // every other data row
  rule:       "FFE5E7EB",   // hairline grid
  ruleStrong: "FF9CA3AF",   // the line between column groups
  ink:        "FF111827",
  muted:      "FF6B7280",
};

export const HAIRLINE = {
  top:    { style: "thin" as const, color: { argb: XL.rule } },
  bottom: { style: "thin" as const, color: { argb: XL.rule } },
  left:   { style: "thin" as const, color: { argb: XL.rule } },
  right:  { style: "thin" as const, color: { argb: XL.rule } },
};

export type ExportCol = { header: string; key: string; width: number; group: string };

/** PDF text extraction breaks a paragraph wherever the page did, so a quote
 *  arrives with commas, bullets and citation numbers stranded on lines of their
 *  own. Exported verbatim that filled a cell with a column of punctuation and
 *  buried the sentence. This reassembles it without rewriting any words. */
export function tidyEvidence(raw: string): string {
  if (!raw) return "";
  const lines = raw.replace(/\r\n?/g, "\n").split("\n").map(l => l.trim());
  const out: string[] = [];
  let bullet = false;
  for (const line of lines) {
    if (!line) continue;                                  // blank lines carry nothing
    const last = out.length - 1;
    // A lone bullet belongs to the item that follows it.
    if (/^[•·▪●∙*•‣▪●]$/.test(line) || /^[-–—]$/.test(line)) {
      bullet = true;
      continue;
    }
    // A line of pure punctuation belongs to the line before it.
    if (/^[,;:.]+$/.test(line)) {
      if (last >= 0) out[last] += line;
      continue;
    }
    // So does a fragment that opens with punctuation.
    if (last >= 0 && /^[,;:.)\]]/.test(line)) {
      out[last] += line;
      continue;
    }
    // A bare citation number after a comma is a footnote marker, not a line.
    if (last >= 0 && /^\d{1,4}$/.test(line) && /[,;]$/.test(out[last])) {
      out[last] += line;
      continue;
    }
    out.push(bullet ? `• ${line}` : line);
    bullet = false;
  }
  return out.join("\n").trim();
}

/** Quote an extract once. The extracted text often already carries its own
 *  quotation marks, and wrapping it again opened the cell on "". */
export function quoteOnce(raw: string): string {
  const t = tidyEvidence(raw).replace(/^["“]+/, "").replace(/["”]+$/, "");
  return t ? `"${t}"` : "";
}

/** How tall a row has to be for nothing in it to be cut off.
 *
 *  Both exports used to pin every row to a fixed height, so any cell with more
 *  than about five wrapped lines lost the rest: the quote was in the file and
 *  invisible on the page. Excel only auto-fits a wrapped row when no explicit
 *  height is set, and not reliably for long cells, so compute it. */
export function rowHeightFor(values: Record<string, any>, cols: ExportCol[]): number {
  const LINE = 12.6;          // points per wrapped line at 10pt
  const PAD = 8;
  const MAX = 409;            // Excel's hard ceiling
  let lines = 1;
  for (const c of cols) {
    const text = String(values[c.key] ?? "");
    if (!text) continue;
    // Characters that fit on one line ≈ the column width, a touch more at 10pt
    // against Excel's 11pt default.
    const perLine = Math.max(8, Math.floor(c.width * 1.15));
    let n = 0;
    for (const para of text.split("\n")) n += Math.max(1, Math.ceil(para.length / perLine));
    if (n > lines) lines = n;
  }
  return Math.min(MAX, Math.max(22, lines * LINE + PAD));
}

/** Rows 1 and 2: merged group bands over the column headers they cover. */
export function writeGroupHeader(ws: ExcelJS.Worksheet, cols: ExportCol[]) {
  const groupRow = ws.addRow([]);
  groupRow.height = 20;
  let i = 0;
  while (i < cols.length) {
    let j = i;
    while (j + 1 < cols.length && cols[j + 1].group === cols[i].group) j++;
    if (j > i) ws.mergeCells(1, i + 1, 1, j + 1);
    const cell = groupRow.getCell(i + 1);
    cell.value = cols[i].group;
    cell.font = { bold: true, size: 10, color: { argb: "FFFFFFFF" } };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.green } };
    cell.alignment = { vertical: "middle", horizontal: "center" };
    cell.border = { right: { style: "medium", color: { argb: "FFFFFFFF" } } };
    i = j + 1;
  }

  const header = ws.addRow(cols.map(c => c.header));
  header.height = 32;
  header.eachCell((cell, col) => {
    cell.font = { bold: true, size: 10, color: { argb: XL.green } };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.headerTint } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    // A heavier rule where one group ends and the next begins.
    const endsGroup = col === cols.length || cols[col - 1].group !== cols[col].group;
    cell.border = {
      ...HAIRLINE,
      bottom: { style: "medium", color: { argb: XL.green } },
      right: endsGroup ? { style: "medium", color: { argb: XL.ruleStrong } } : HAIRLINE.right,
    };
  });
}

/** Wrap, band and rule every cell in a data row. */
export function styleDataRow(row: ExcelJS.Row, cols: ExportCol[], banded: boolean) {
  row.eachCell({ includeEmpty: true }, (cell, col) => {
    cell.alignment = { vertical: "top", wrapText: true };
    cell.font = { size: 10, color: { argb: XL.ink } };
    if (banded) cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.band } };
    const endsGroup = col === cols.length || cols[col - 1]?.group !== cols[col]?.group;
    cell.border = endsGroup
      ? { ...HAIRLINE, right: { style: "medium", color: { argb: XL.ruleStrong } } }
      : HAIRLINE;
  });
}

/** Paint a verdict cell.
 *
 *  `lead` marks the decision that actually stands. It takes a deeper tint and
 *  bold text; the model's own verdict beside it keeps the same hue a shade
 *  lighter. Two weights of one colour, rather than two colours, so the pair
 *  stays comparable at a glance while the operative call is unmistakable. */
export function paintVerdict(cell: ExcelJS.Cell, include: boolean, lead: boolean) {
  const fill = include
    ? (lead ? XL.greenDeep : XL.greenTint)
    : (lead ? XL.redDeep : XL.redTint);
  cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fill } };
  cell.font = { size: 10, bold: lead, color: { argb: include ? XL.green : XL.red } };
  cell.alignment = { vertical: "top", horizontal: "center", wrapText: true };
}

/** Mark a decision the reviewer changed, so their own edits are findable in a
 *  sheet of hundreds. Blue, not amber: amber already means "partial match". */
export function markReviewerEdit(decisionCell: ExcelJS.Cell, flagCell?: ExcelJS.Cell) {
  const edge = { style: "medium" as const, color: { argb: XL.blue } };
  decisionCell.border = { top: edge, bottom: edge, left: edge, right: edge };
  if (flagCell) {
    flagCell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.blueTint } };
    flagCell.font = { size: 10, bold: true, color: { argb: XL.blue } };
    flagCell.alignment = { vertical: "top", horizontal: "center" };
  }
}

export function stamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

export function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** RFC 4180: double the quotes, wrap anything with a comma, quote or newline.
 *  The evidence cells contain all three, so this is not optional. */
export function csvEscape(v: any): string {
  const t = v === null || v === undefined ? "" : String(v);
  return /[",\n\r]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

export function toCsv(cols: ExportCol[], rows: Record<string, any>[]): string {
  return [
    cols.map(c => csvEscape(c.header)).join(","),
    ...rows.map(r => cols.map(c => csvEscape(r[c.key])).join(",")),
  ].join("\r\n");
}

export function saveCsv(cols: ExportCol[], rows: Record<string, any>[], name: string) {
  // A BOM, so Excel reads it as UTF-8 instead of mangling every quoted phrase.
  saveBlob(new Blob(["﻿" + toCsv(cols, rows)], { type: "text/csv;charset=utf-8" }), name);
}

/** A summary sheet: what the review asked, and what this run decided.
 *
 *  A bare results grid does not say what the decisions were made against,
 *  which is the first thing anyone opening the file months later needs. */
export function addSummarySheet(
  wb: ExcelJS.Workbook,
  title: string,
  sections: { heading: string; pairs: [string, string][] }[],
) {
  const ws = wb.addWorksheet("Summary");
  ws.columns = [{ width: 26 }, { width: 86 }];

  const t = ws.addRow([title]);
  t.getCell(1).font = { bold: true, size: 16, color: { argb: XL.green } };
  t.height = 26;
  ws.mergeCells(t.number, 1, t.number, 2);

  const when = ws.addRow([`Exported ${new Date().toLocaleString()}`]);
  when.getCell(1).font = { size: 10, color: { argb: XL.muted } };
  ws.mergeCells(when.number, 1, when.number, 2);

  for (const section of sections) {
    ws.addRow([]);
    const h = ws.addRow([section.heading]);
    h.height = 20;
    for (const col of [1, 2]) {
      h.getCell(col).fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.green } };
    }
    h.getCell(1).font = { bold: true, size: 11, color: { argb: "FFFFFFFF" } };
    h.getCell(1).alignment = { vertical: "middle" };

    for (const [k, v] of section.pairs) {
      const r = ws.addRow([k, v]);
      r.getCell(1).font = { bold: true, size: 10, color: { argb: XL.green } };
      r.getCell(2).font = { size: 10, color: { argb: XL.ink } };
      r.getCell(1).alignment = { vertical: "top" };
      r.getCell(2).alignment = { vertical: "top", wrapText: true };
      r.getCell(1).border = HAIRLINE;
      r.getCell(2).border = HAIRLINE;
      r.height = rowHeightFor({ v }, [{ header: "", key: "v", width: 86, group: "" }]);
    }
  }
  return ws;
}
