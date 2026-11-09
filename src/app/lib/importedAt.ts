// Records brought in part-way through a review.
//
// A reviewer may already have done the earlier stages elsewhere: they arrive
// with 200 PDFs that passed their own abstract screening and want to start at
// full-text screening. Every stage here reads the previous stage's output
// (uniquePapers -> results -> fullTexts -> fullTextResults), so such a record is
// invisible until something writes the earlier rows.
//
// Writing them as ordinary INCLUDE decisions would be a lie the app then
// reports: PRISMA would count them as screened here, and they would appear in
// the AI decision log as judgements no model ever made. So they are marked, and
// the places that report on provenance read the mark:
//
//   - PRISMA counts them under "References from other sources", which is the
//     box PRISMA 2020 provides for exactly this.
//   - The AI decision log leaves them out, because no AI decided them.
//
// Everywhere else they behave like any other included paper, which is the point.

export type ImportStage =
  | "home"
  | "abstract-screening"
  | "full-text-acquisition"
  | "full-text-screening"
  | "table-extraction"
  | "text-extraction"
  | "quality-assessment"
  | "citation-snowball"
  | "meta-analysis"
  | "writing";

/** What the reviewer sees on a record they imported at this stage. */
export const STAGE_LABEL: Record<ImportStage, string> = {
  "home": "Home",
  "abstract-screening": "Abstract Screening",
  "full-text-acquisition": "Full-Text Acquisition",
  "full-text-screening": "Full-Text Screening",
  "table-extraction": "Table Extraction",
  "text-extraction": "Text Extraction",
  "quality-assessment": "Quality Assessment",
  "citation-snowball": "Citation Snowball",
  "meta-analysis": "Meta-analysis",
  "writing": "Writing Assistant",
};

/**
 * Both of these are now true at every stage, and the pair is kept as functions
 * because the reasoning belongs somewhere and every caller reads it here.
 *
 * Each stage reads the previous stage's output, so a record with no earlier
 * rows is invisible everywhere except the stage it was dropped into. That made
 * the pipeline compulsory: a reviewer who uploaded PDFs at Acquisition had to
 * run full-text screening before the extraction stages would show them, even
 * though they had come with the studies they had already chosen.
 *
 * So an import seeds every row, and the stage it was dropped into is recorded
 * on each. Two things keep that honest rather than a lie about work nobody did:
 * PRISMA counts these under "references from other sources" and the AI decision
 * log leaves them out (see isImported below), and no stage treats a seeded row
 * as its own work, so screening still offers the record if the reviewer wants
 * to screen it after all.
 */
export function needsAbstractRow(_stage: ImportStage): boolean {
  return true;
}

export function needsFullTextRow(_stage: ImportStage): boolean {
  return true;
}

/**
 * Stages that read the article's words rather than its record.
 *
 * A record looked up by DOI carries a title and an abstract, not the paper, so
 * a study imported at one of these stages arrives with nothing to work on and
 * the page is empty. Importing there is a statement that the reviewer wants to
 * work on these studies HERE, so the text is fetched for them rather than
 * sending them to another stage to press a button.
 *
 * Full-Text Acquisition is left out on purpose: that page's whole job is this
 * fetch, with its own queue and its own cancel, and starting a second one
 * behind it would fight with it.
 */
export function readsFullText(stage: ImportStage): boolean {
  return stage === "full-text-screening" || stage === "table-extraction"
      || stage === "text-extraction" || stage === "quality-assessment"
      || stage === "meta-analysis" || stage === "writing";
}

export const IMPORT_REASON_PREFIX = "Imported at";

/** The reason string stored on a seeded row. Says plainly what did not happen. */
export function importReason(stage: ImportStage): string {
  return `${IMPORT_REASON_PREFIX} ${STAGE_LABEL[stage]}. Not screened in this app; `
       + `the reviewer supplied it as already eligible at this stage.`;
}

// These take `unknown` rather than a shape whose every field is optional: such
// a type is "weak" in TypeScript, so passing any record that happens not to
// carry the mark is a compile error, which is exactly the common case here.

/** True for a record seeded by an import rather than decided here. */
export function isImported(row: unknown): boolean {
  return !!(row as { Imported?: unknown } | null | undefined)?.Imported;
}

/** How many records PRISMA should report under "other sources". */
export function countImported(rows: readonly unknown[] | null | undefined): number {
  return (rows || []).filter(isImported).length;
}

/** The rows an AI decision log may legitimately claim as AI decisions. */
export function excludeImported<T>(rows: readonly T[] | null | undefined): T[] {
  return (rows || []).filter(r => !isImported(r));
}
