// Turning an extracted text layer into something a person can read, without
// losing the character offsets everything else is keyed on.
//
// Extracted full text arrives in one of two shapes, and neither reads as prose:
//
//   - Hard-wrapped, from a publisher's XML or a backend PDF extraction. Every
//     line break the typesetter made is still in the string, so sentences stop
//     mid-clause and citation markers land on lines of their own: "[\n1\n,\n3]".
//   - One enormous line per page, from the browser-side importer, which joins a
//     page's text fragments with spaces. A whole page as a single paragraph is
//     no more readable than a wrapped one.
//
// Both are reflowed here into headings and paragraphs of a few sentences each.
//
// The constraint that shapes this file: an evidence quote is located by
// character offsets into the *raw* text, and those offsets are what the viewer
// highlights, what "Locate" scrolls to, and what the exported audit trail
// quotes. Reflowing by rewriting the string would invalidate every one of them.
// So each block carries `map`, giving the raw offset of each character it
// displays, and every transformation here is either a deletion of whitespace or
// an insertion of a separator, both of which a map can express exactly. No
// character of the original is ever altered or reordered, which is also what
// makes the reading view safe to quote from.

/** A heading or paragraph of the reflowed text. */
export type ReadingBlock = {
  kind: "heading" | "p";
  /** What to display. */
  text: string;
  /** `map[i]` is the raw offset of `text[i]`, or -1 for an inserted separator. */
  map: number[];
  /** The raw range this block covers, for ordering and lookup. */
  start: number;
  end: number;
};

/** A character range of the raw text, which is what an evidence quote is. */
export type RawSpan = { start: number; end: number };

/** A run of display characters belonging to one span. */
export type MarkRange = { from: number; to: number; spanIndex: number };

// The section names worth setting as a heading. Deliberately a closed list: a
// guess from capitalisation alone promotes author names, table captions and
// running heads, and a paper with forty headings has none.
export const HEADING_RE =
  /^\s*(?:\d+\.?\s+)?(abstract|background|introduction|objectives?|aims?|methods?|materials and methods|study design|participants|setting|interventions?|outcomes?|search strategy|eligibility criteria|data (?:collection|extraction|analysis|synthesis)|statistical analysis|results|findings|discussion|conclusions?|limitations|references|acknowled?ge?ments?|funding|conflicts? of interest|author contributions|keywords|highlights|supplementary material)\s*:?\s*$/i;

/** Roughly how many characters make a comfortable paragraph before it is split
 *  at a sentence boundary. A page of unbroken text is technically correct and
 *  nobody reads it. */
const PARA_CHARS = 460;

/** Sentences per paragraph once a long one is split. */
const PARA_SENTENCES = 3;

type Line = { text: string; start: number; end: number };

/** Split the raw text into lines, keeping each one's offset. The string is
 *  never rewritten, so a CR is left to the whitespace handling below rather
 *  than normalised away, which would shift every offset after it. */
function lines(raw: string): Line[] {
  const out: Line[] = [];
  let at = 0;
  while (at <= raw.length) {
    const nl = raw.indexOf("\n", at);
    const end = nl === -1 ? raw.length : nl;
    out.push({ text: raw.slice(at, end), start: at, end });
    if (nl === -1) break;
    at = nl + 1;
  }
  return out;
}

/** A mapped string under construction: display text plus the raw offset of
 *  each character. */
class Mapped {
  chars: string[] = [];
  map: number[] = [];

  /** Append a raw slice verbatim, one character at a time, so every display
   *  character keeps the offset it came from. */
  pushRaw(raw: string, from: number, to: number) {
    for (let i = from; i < to; i++) {
      this.chars.push(raw[i]);
      this.map.push(i);
    }
  }

  /** Append a character this module introduced, which belongs to no raw
   *  offset. Only ever a single space between two wrapped lines. */
  pushSeparator(ch: string) {
    this.chars.push(ch);
    this.map.push(-1);
  }

  get last(): string {
    return this.chars[this.chars.length - 1] ?? "";
  }

  pop() {
    this.chars.pop();
    this.map.pop();
  }

  get length() {
    return this.chars.length;
  }
}

/** The indices of `text` to drop so that extracted prose reads correctly.
 *
 *  Every rule here deletes whitespace and nothing else, which is what lets the
 *  offset map survive: a deletion drops an entry, where a substitution would
 *  mean a display character with no raw character behind it. */
function whitespaceToDrop(text: string): Set<number> {
  const drop = new Set<number>();
  const ws = (i: number) => /\s/.test(text[i] ?? "");

  // "[\n1\n,\n3 ]" is one citation group, not five lines. Only groups that are
  // entirely digits and separators qualify, so "[see Table 1]" is left alone.
  for (const m of text.matchAll(/\[[\s\d,;.–-]*\d[\s\d,;.–-]*\]/g)) {
    const at = m.index ?? 0;
    for (let i = at; i < at + m[0].length; i++) if (ws(i)) drop.add(i);
  }

  // A wrapped line that broke before its own punctuation: "the result ,".
  for (const m of text.matchAll(/\s+(?=[.,;:!?)\]])/g)) {
    const at = m.index ?? 0;
    for (let i = at; i < at + m[0].length; i++) drop.add(i);
  }

  // ... or straight after an opening bracket: "( 95% CI".
  for (const m of text.matchAll(/(?<=[([])\s+/g)) {
    const at = m.index ?? 0;
    for (let i = at; i < at + m[0].length; i++) drop.add(i);
  }

  // Runs of spaces from a two-column layout collapse to one.
  for (const m of text.matchAll(/[^\S\n]{2,}/g)) {
    const at = m.index ?? 0;
    for (let i = at + 1; i < at + m[0].length; i++) drop.add(i);
  }

  return drop;
}

/** Apply those deletions to a mapped string. */
function compact(m: Mapped): { text: string; map: number[] } {
  const text = m.chars.join("");
  const drop = whitespaceToDrop(text);
  if (drop.size === 0) return { text, map: m.map };
  const keptChars: string[] = [];
  const keptMap: number[] = [];
  for (let i = 0; i < text.length; i++) {
    if (drop.has(i)) continue;
    keptChars.push(text[i]);
    keptMap.push(m.map[i]);
  }
  return { text: keptChars.join(""), map: keptMap };
}

/** The raw range a map covers, ignoring inserted separators. */
function rangeOf(map: number[]): { start: number; end: number } {
  let start = Infinity;
  let end = -Infinity;
  for (const o of map) {
    if (o < 0) continue;
    if (o < start) start = o;
    if (o > end) end = o;
  }
  return Number.isFinite(start)
    ? { start, end: end + 1 }
    : { start: 0, end: 0 };
}

// A full stop that ends an abbreviation rather than a sentence. Checked against
// the text up to and including the stop, so the last token before it is what
// matches. A single letter counts, which is what catches "U.S." and "e.g.": the
// boundary before it may be a stop as well as a space.
const ABBREV_END =
  /(?:^|[\s([.])(?:[A-Za-z]|al|vs|cf|ca|no|nos|fig|figs|eq|eqs|ref|refs|tab|tabs|approx|est|resp|min|max|sd|se|ci|dr|prof|mr|mrs|ms|st|vol|pp|ed|eds|inc|ltd|co|univ|dept|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.$/i;

/** Where a long paragraph may be broken: after a sentence-ending mark followed
 *  by whitespace and the start of a new one. Abbreviations are excluded, so a
 *  paragraph never breaks inside "U.S. Food" or "Fig. 2". */
function sentenceStarts(text: string): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(/(?<=[.!?])\s+(?=["“('[]?[A-Z0-9])/g)) {
    const at = m.index ?? 0;
    if (ABBREV_END.test(text.slice(0, at))) continue;
    out.push(at + m[0].length);
  }
  return out;
}

/** Slice a mapped paragraph into blocks of a few sentences each. Purely a
 *  slice: no character is added, removed or changed, so the offsets hold. */
function splitParagraph(text: string, map: number[]): ReadingBlock[] {
  if (text.length <= PARA_CHARS) {
    return [{ kind: "p", text, map, ...rangeOf(map) }];
  }
  const starts = sentenceStarts(text);
  if (starts.length < PARA_SENTENCES) {
    return [{ kind: "p", text, map, ...rangeOf(map) }];
  }
  // Cut at every third sentence, and never leave a final scrap on its own.
  const cuts: number[] = [];
  for (let i = PARA_SENTENCES - 1; i < starts.length; i += PARA_SENTENCES) cuts.push(starts[i]);
  if (cuts.length && text.length - cuts[cuts.length - 1] < 120) cuts.pop();

  const bounds = [0, ...cuts, text.length];
  const out: ReadingBlock[] = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    const from = bounds[i];
    const to = bounds[i + 1];
    const slice = text.slice(from, to);
    if (!slice.trim()) continue;
    const sliceMap = map.slice(from, to);
    out.push({ kind: "p", text: slice, map: sliceMap, ...rangeOf(sliceMap) });
  }
  return out;
}

/**
 * Reflow extracted text into headings and paragraphs that keep their offsets.
 *
 * Returns [] for empty input, so a caller can tell "nothing to read" from "one
 * empty paragraph".
 */
export function toReadingBlocks(raw: string): ReadingBlock[] {
  if (!raw || !raw.trim()) return [];

  const blocks: ReadingBlock[] = [];
  let buf = new Mapped();

  const flush = () => {
    if (!buf.length) return;
    const { text, map } = compact(buf);
    buf = new Mapped();
    if (!text.trim()) return;
    blocks.push(...splitParagraph(text, map));
  };

  for (const line of lines(raw)) {
    const trimmed = line.text.trim();
    if (!trimmed) { flush(); continue; }

    // Where the trimmed content sits in the raw string.
    const lead = line.text.length - line.text.trimStart().length;
    const from = line.start + lead;
    const to = from + trimmed.length;

    if (HEADING_RE.test(trimmed)) {
      flush();
      const h = new Mapped();
      h.pushRaw(raw, from, to);
      // Drop a trailing colon from the display, not from the map's idea of what
      // the heading covers: "Methods:" is the same heading as "Methods".
      while (/[:\s]/.test(h.last)) h.pop();
      const { text, map } = compact(h);
      blocks.push({ kind: "heading", text, map, ...rangeOf(map) });
      continue;
    }

    if (buf.length) {
      // A word broken across the line break: "random-\nised". Drop the hyphen
      // and join without a space. Only before a lowercase letter, so hyphenated
      // compounds ("double-blind") and ranges ("1990-2020") are left alone.
      const hyphenated = /[-‐]$/.test(buf.last) && /^[a-z]/.test(trimmed);
      if (hyphenated) buf.pop();
      else buf.pushSeparator(" ");
    }
    buf.pushRaw(raw, from, to);
  }
  flush();

  return blocks;
}

/**
 * Which display characters of a block belong to which span.
 *
 * An inserted separator between two marked characters is marked too, so a quote
 * spanning a line break draws as one continuous highlight rather than as two
 * marks with a gap between them.
 */
export function markRanges(block: ReadingBlock, spans: RawSpan[]): MarkRange[] {
  if (!spans.length || !block.text) return [];
  const owner = new Int32Array(block.map.length).fill(-1);
  let any = false;

  spans.forEach((span, k) => {
    if (span.end <= span.start) return;
    for (let i = 0; i < block.map.length; i++) {
      const at = block.map[i];
      if (at < 0 || owner[i] !== -1) continue;
      if (at >= span.start && at < span.end) { owner[i] = k; any = true; }
    }
  });
  if (!any) return [];

  // Bridge the separators this module inserted, inside a span.
  for (let i = 1; i < owner.length - 1; i++) {
    if (owner[i] !== -1 || block.map[i] !== -1) continue;
    if (owner[i - 1] >= 0 && owner[i - 1] === owner[i + 1]) owner[i] = owner[i - 1];
  }

  const out: MarkRange[] = [];
  let i = 0;
  while (i < owner.length) {
    if (owner[i] === -1) { i++; continue; }
    const spanIndex = owner[i];
    let j = i;
    while (j + 1 < owner.length && owner[j + 1] === spanIndex) j++;
    out.push({ from: i, to: j + 1, spanIndex });
    i = j + 1;
  }
  return out;
}

/** The index of the block a raw offset falls in, or -1. Used to scroll the
 *  reading view to a quote the reviewer asked to see. */
export function blockAt(blocks: ReadingBlock[], offset: number): number {
  for (let i = 0; i < blocks.length; i++) {
    if (offset >= blocks[i].start && offset < blocks[i].end) return i;
  }
  return -1;
}

/** The heading in force at a given block, for a sticky section label. */
export function sectionAt(blocks: ReadingBlock[], index: number): string {
  for (let i = Math.min(index, blocks.length - 1); i >= 0; i--) {
    if (blocks[i]?.kind === "heading") return blocks[i].text;
  }
  return "";
}
