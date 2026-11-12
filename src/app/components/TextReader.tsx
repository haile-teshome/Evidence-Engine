// The extracted text of a paper, to read or to audit.
//
// Two views of the same string, because they answer different questions.
//
//   "read"  reflows the text into headings and paragraphs at a comfortable
//           measure. This is what a reviewer reads when they want to know what
//           the paper says. The old viewer showed only the other one, and 12px
//           monospace in a line-numbered table is a log file, not a paper.
//
//   "raw"   is the text exactly as extracted, line by line, with the character
//           offsets intact. This is the auditable artefact: it is what the
//           export quotes and what the offsets in an evidence record refer to,
//           so it stays one click away rather than being replaced.
//
// Both mark every evidence span and put `data-span-start` on the mark, so the
// page that owns the scroll container can scroll to a quote in either view
// without knowing which one is showing.
import { forwardRef, useMemo } from "react";

import { markRanges, toReadingBlocks, type RawSpan } from "../lib/readingText";

/** An evidence span to mark. `section` and `label` are only for the tooltip. */
export type ReaderSpan = RawSpan & { section?: string; label?: string };

export type TextReaderView = "read" | "raw";

export type TextReaderProps = {
  text: string;
  spans?: ReaderSpan[];
  /** The one span the reviewer asked to see, drawn more strongly than the rest. */
  activeSpan?: [number, number] | null;
  view?: TextReaderView;
  className?: string;
};

// A translucent amber over a dark background, under near-white text, comes out
// dimmer than the prose around it, which is the opposite of a highlight. So in
// dark mode the mark keeps the paper metaphor: amber with dark ink on it.
const MARK =
  "rounded-[2px] px-0.5 bg-amber-200/80 dark:bg-amber-400/75 dark:text-neutral-900";
const MARK_ACTIVE =
  "rounded-[2px] px-0.5 bg-amber-300 ring-2 ring-amber-500/80 "
  + "dark:bg-amber-300 dark:text-neutral-900 dark:ring-amber-200/70";

function tip(span: ReaderSpan): string {
  const where = span.section ? `${span.section}, ` : "";
  return `${span.label ? `${span.label} · ` : ""}${where}chars ${span.start}-${span.end}`;
}

export const TextReader = forwardRef<HTMLDivElement, TextReaderProps>(function TextReader(
  { text, spans = [], activeSpan = null, view = "read", className = "" },
  ref,
) {
  const blocks = useMemo(() => (view === "read" ? toReadingBlocks(text) : []), [text, view]);
  const lines = useMemo(() => (view === "raw" ? toLines(text) : []), [text, view]);

  const isActive = (span: ReaderSpan) =>
    Boolean(activeSpan && span.start === activeSpan[0] && span.end === activeSpan[1]);

  if (!text) {
    return (
      <div ref={ref} className={`flex items-center justify-center p-6 text-sm text-muted-foreground ${className}`}>
        No full text available for this article.
      </div>
    );
  }

  // ── Raw: verbatim, numbered, auditable ────────────────────────────────────
  if (view === "raw") {
    return (
      <div ref={ref} className={`overflow-auto bg-muted/20 font-mono text-xs leading-relaxed ${className}`}>
        <table className="w-full border-collapse">
          <tbody>
            {lines.map(ln => {
              const here = spans.filter(s => s.start < ln.end && s.end > ln.start);
              return (
                <tr key={ln.idx} className="align-top">
                  <td className="w-10 select-none border-r bg-muted/30 py-0.5 pl-2 pr-2 text-right tabular-nums text-muted-foreground/60">
                    {ln.idx}
                  </td>
                  <td className="whitespace-pre-wrap break-words px-3 py-0.5">
                    {here.length === 0 ? ln.text : rawLine(ln, here, isActive)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  // ── Read: reflowed prose ──────────────────────────────────────────────────
  return (
    <div ref={ref} className={`overflow-auto bg-background ${className}`}>
      <div className="mx-auto max-w-[68ch] px-6 py-5 text-[15px] leading-7 text-foreground/90">
        {blocks.map((block, i) => {
          const ranges = markRanges(block, spans);

          // A heading sticks to the top of the scroll area until the next one
          // pushes it out, so the section being read is always named.
          if (block.kind === "heading") {
            return (
              <h4
                key={i}
                className="sticky top-0 z-10 -mx-6 mb-2 mt-7 bg-background/95 px-6 pb-1.5 pt-2
                           text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground
                           backdrop-blur-sm first:mt-0"
              >
                {block.text}
              </h4>
            );
          }

          return (
            <p key={i} className="mb-3.5 [text-wrap:pretty]">
              {ranges.length === 0 ? block.text : paint(block.text, ranges, spans, isActive)}
            </p>
          );
        })}
      </div>
    </div>
  );
});

/** Slice a reflowed paragraph into plain text and marks. */
function paint(
  text: string,
  ranges: { from: number; to: number; spanIndex: number }[],
  spans: ReaderSpan[],
  isActive: (s: ReaderSpan) => boolean,
) {
  const out: React.ReactNode[] = [];
  let at = 0;
  ranges.forEach((r, i) => {
    if (r.from > at) out.push(<span key={`t${i}`}>{text.slice(at, r.from)}</span>);
    const span = spans[r.spanIndex];
    out.push(
      <mark
        key={`m${i}`}
        data-span-start={span?.start}
        title={span ? tip(span) : undefined}
        className={span && isActive(span) ? MARK_ACTIVE : MARK}
      >
        {text.slice(r.from, r.to)}
      </mark>,
    );
    at = r.to;
  });
  if (at < text.length) out.push(<span key="tail">{text.slice(at)}</span>);
  return out;
}

type Line = { text: string; start: number; end: number; idx: number };

/** The raw text split into lines, each knowing its own character range. */
function toLines(text: string): Line[] {
  const out: Line[] = [];
  let at = 0;
  let idx = 1;
  while (at <= text.length) {
    const nl = text.indexOf("\n", at);
    const end = nl === -1 ? text.length : nl;
    // The newline counts as part of the line it ends, so the ranges tile the
    // string exactly and a span landing on a break still finds a line.
    out.push({ text: text.slice(at, end), start: at, end: nl === -1 ? end : end + 1, idx: idx++ });
    if (nl === -1) break;
    at = nl + 1;
  }
  return out;
}

/** One raw line, with the parts of it that belong to an evidence span marked. */
function rawLine(ln: Line, here: ReaderSpan[], isActive: (s: ReaderSpan) => boolean) {
  const out: React.ReactNode[] = [];
  let at = 0;
  [...here]
    .sort((a, b) => a.start - b.start)
    .forEach((span, i) => {
      const from = Math.max(0, span.start - ln.start);
      const to = Math.min(ln.text.length, span.end - ln.start);
      if (to <= at) return;                            // already covered by an earlier span
      if (from > at) out.push(<span key={`t${i}`}>{ln.text.slice(at, from)}</span>);
      out.push(
        <mark
          key={`m${i}`}
          data-span-start={span.start}
          title={tip(span)}
          className={isActive(span) ? MARK_ACTIVE : MARK}
        >
          {ln.text.slice(Math.max(from, at), to)}
        </mark>,
      );
      at = to;
    });
  if (at < ln.text.length) out.push(<span key="tail">{ln.text.slice(at)}</span>);
  return out;
}
