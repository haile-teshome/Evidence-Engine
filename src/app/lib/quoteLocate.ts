// Finding a quoted sentence back in the document it came from.
//
// Every extracted value carries the sentence the model says it read it in. That
// claim is only worth something if the reviewer can see it in the paper, which
// means finding it there: not by string equality, because a PDF's text layer
// breaks words across lines, doubles spaces, and turns hyphens and quotation
// marks into whatever the typesetter used.
//
// So the search is done on a normalised copy of the text, and every normalised
// position is mapped back to its position in the original, which is what the
// caller needs in order to highlight the real words.

/** A quote located in a document: offsets into the ORIGINAL text. */
export type QuoteHit = { start: number; end: number };

const FOLD: Record<string, string> = {
  "‘": "'", "’": "'", "“": '"', "”": '"',
  "–": "-", "—": "-", "−": "-", " ": " ",
};

/**
 * Normalise for comparison, keeping a map back to the original offsets.
 * Runs of whitespace collapse to one space; letters lower-case; the punctuation
 * a typesetter substitutes is folded back.
 */
function normalise(text: string): { norm: string; at: number[] } {
  const out: string[] = [];
  const at: number[] = [];
  let lastWasSpace = true;                       // trims the leading run too
  for (let i = 0; i < text.length; i++) {
    const raw = text[i];
    const ch = FOLD[raw] ?? raw;
    if (/\s/.test(ch)) {
      if (lastWasSpace) continue;
      lastWasSpace = true;
      out.push(" ");
      at.push(i);
      continue;
    }
    lastWasSpace = false;
    out.push(ch.toLowerCase());
    at.push(i);
  }
  return { norm: out.join(""), at };
}

/** Strip to letters and digits, for the last-resort match. */
function letters(s: string): string {
  return s.replace(/[^a-z0-9]+/g, "");
}

/**
 * Where `quote` appears in `text`, or null.
 *
 * Three attempts, each weaker than the last: the whole quote, the first clause
 * of it (models often extend a quote past where the sentence ends), and letters
 * only, which survives a text layer that lost its spaces.
 */
export function locateQuote(text: string, quote: string): QuoteHit | null {
  const q = (quote || "").trim();
  if (!text || q.length < 12) return null;

  const doc = normalise(text);
  const needle = normalise(q).norm.trim();
  if (!needle) return null;

  const direct = doc.norm.indexOf(needle);
  if (direct >= 0) {
    return { start: doc.at[direct], end: doc.at[direct + needle.length - 1] + 1 };
  }

  // The opening of the quote: enough of it to be unmistakable, in case the
  // model ran two sentences together or added a trailing clause of its own.
  const head = needle.slice(0, 60);
  if (head.length >= 24) {
    const partial = doc.norm.indexOf(head);
    if (partial >= 0) {
      return { start: doc.at[partial], end: doc.at[partial + head.length - 1] + 1 };
    }
  }

  // Letters only. A text layer that dropped its spaces ("Sex:Male167(61.4%)")
  // still matches here, and nothing else would find it.
  const bare = letters(doc.norm);
  const bareNeedle = letters(needle).slice(0, 120);
  // Twelve letters and digits is about "Sex: Male 167": short, but as an exact
  // run inside the document it is not something two different sentences share.
  if (bareNeedle.length < 12) return null;
  const hit = bare.indexOf(bareNeedle);
  if (hit < 0) return null;

  // Map a letters-only offset back by counting letters through the document.
  let seen = 0;
  let from = -1;
  for (let i = 0; i < doc.norm.length; i++) {
    if (!/[a-z0-9]/.test(doc.norm[i])) continue;
    if (seen === hit) { from = i; break; }
    seen++;
  }
  if (from < 0) return null;

  let want = bareNeedle.length;
  let to = from;
  while (to < doc.norm.length && want > 0) {
    if (/[a-z0-9]/.test(doc.norm[to])) want--;
    to++;
  }
  return { start: doc.at[from], end: doc.at[Math.min(to, doc.at.length) - 1] + 1 };
}

/**
 * The quote in its surroundings: enough of the paper either side to see what
 * the sentence is part of, cut at word boundaries.
 */
export function quoteInContext(
  text: string,
  quote: string,
  pad = 420,
): { before: string; match: string; after: string; found: boolean } {
  const hit = locateQuote(text, quote);
  if (!hit) return { before: "", match: quote || "", after: "", found: false };

  let from = Math.max(0, hit.start - pad);
  let to = Math.min(text.length, hit.end + pad);
  if (from > 0) {
    const space = text.indexOf(" ", from);
    if (space > 0 && space < hit.start) from = space + 1;
  }
  if (to < text.length) {
    const space = text.lastIndexOf(" ", to);
    if (space > hit.end) to = space;
  }
  return {
    before: (from > 0 ? "… " : "") + text.slice(from, hit.start),
    match: text.slice(hit.start, hit.end),
    after: text.slice(hit.end, to) + (to < text.length ? " …" : ""),
    found: true,
  };
}

/** Roughly how far into the document the quote sits, as a percentage. */
export function quotePosition(text: string, quote: string): number | null {
  const hit = locateQuote(text, quote);
  if (!hit || !text.length) return null;
  return Math.round((hit.start / text.length) * 100);
}
