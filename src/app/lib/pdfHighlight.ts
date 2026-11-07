// Finding a quoted sentence on a page of the actual PDF.
//
// The text passage in the provenance dialog proves the sentence is in the
// document. It does not show the reviewer the table it sits under, the figure
// beside it, or which page to turn to in the paper they are citing. For that
// the page itself has to be on screen with the sentence marked on it.
//
// pdf.js hands back a page's text as a list of positioned fragments, in reading
// order but broken wherever the typesetter changed font or kerned a word. This
// joins them into one string, finds the quote in it with the same tolerant
// matching used elsewhere, and reports which fragments it covered, which is
// what a highlight is drawn from.
import { locateQuote } from "./quoteLocate";

/** The part of a pdf.js text item this module needs. */
export type PdfTextItem = {
  str: string;
  /** [a, b, c, d, e, f] — pdf.js text matrix, in PDF user space. */
  transform: number[];
  width: number;
  height: number;
};

/** A fragment of the page the quote covers, in page coordinates. */
export type ItemMatch = { index: number; item: PdfTextItem };

/** The page's text as one string, with each item's span within it. */
export function pageText(items: PdfTextItem[]): { text: string; spans: [number, number][] } {
  const spans: [number, number][] = [];
  let text = "";
  for (const item of items) {
    const start = text.length;
    text += item.str;
    spans.push([start, text.length]);
    // pdf.js splits on kerning, so a space between fragments is a guess. It is
    // the right guess for matching: the normaliser collapses runs of spaces,
    // and without it two words run together into one that is in no dictionary.
    if (!/\s$/.test(item.str)) text += " ";
  }
  return { text, spans };
}

/**
 * The items a quote covers on this page, or [] when it is not on this page.
 */
export function matchItems(items: PdfTextItem[], quote: string): ItemMatch[] {
  if (!items.length || !quote.trim()) return [];
  const { text, spans } = pageText(items);
  const hit = locateQuote(text, quote);
  if (!hit) return [];
  const out: ItemMatch[] = [];
  spans.forEach(([from, to], i) => {
    // Any overlap: a fragment holding one word of the sentence is part of it.
    if (from < hit.end && to > hit.start) out.push({ index: i, item: items[i] });
  });
  return out;
}

/** A highlight box in CSS pixels, relative to the rendered page. */
export type Box = { left: number; top: number; width: number; height: number };

/**
 * Where to draw a fragment, given the page's viewport transform.
 *
 * `transform` is the viewport matrix pdf.js renders with, so the result lines
 * up with the canvas at whatever scale and rotation the page was drawn.
 */
export function itemBox(item: PdfTextItem, transform: number[], scale: number): Box {
  // Both matrices are read defensively: destructuring an undefined array calls
  // its iterator and throws, and a page with one odd text item should lose its
  // highlight, not its render.
  const v = Array.isArray(transform) && transform.length >= 6 ? transform : [1, 0, 0, -1, 0, 0];
  const t = Array.isArray(item?.transform) && item.transform.length >= 6
    ? item.transform : [scale, 0, 0, scale, 0, 0];
  const [a, b, c, d, e, f] = v;
  const [ta, tb, tc, td, te, tf] = t;
  // Matrix multiply: viewport × item.
  const m = [
    a * ta + c * tb,
    b * ta + d * tb,
    a * tc + c * td,
    b * tc + d * td,
    a * te + c * tf + e,
    b * te + d * tf + f,
  ];
  const height = Math.hypot(m[2], m[3]) || item.height * scale;
  return {
    left: m[4],
    top: m[5] - height,
    width: Math.max(item.width * scale, 1),
    height: Math.max(height, 1),
  };
}

/** Merge boxes that sit on the same line, so one sentence is one mark. */
export function mergeBoxes(boxes: Box[]): Box[] {
  if (boxes.length < 2) return boxes;
  const sorted = [...boxes].sort((p, q) => (p.top - q.top) || (p.left - q.left));
  const out: Box[] = [];
  for (const box of sorted) {
    const last = out[out.length - 1];
    const sameLine = last
      && Math.abs(last.top - box.top) <= Math.max(2, last.height * 0.4)
      && box.left - (last.left + last.width) < Math.max(6, last.height);
    if (sameLine) {
      const right = Math.max(last.left + last.width, box.left + box.width);
      last.left = Math.min(last.left, box.left);
      last.width = right - last.left;
      last.height = Math.max(last.height, box.height);
      last.top = Math.min(last.top, box.top);
    } else {
      out.push({ ...box });
    }
  }
  return out;
}
