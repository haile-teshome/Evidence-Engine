// One evidence quote, as a card.
//
// What this card is for: a reviewer reads the sentence, decides whether it
// supports the value beside it, and then goes to look at it in context. So the
// sentence gets the room, and going to see it is the card's own click rather
// than a button in the corner.
//
// It used to be an italic blockquote with a left rule, a quote icon inline
// before the first word, and "char 1193-1247" in the header. Three markers for
// one idea, a 40-word passage set in italic, and the header given over to an
// offset nobody cites. Now: the section in the edge rule, the page of the PDF
// where the viewer found the sentence, and one typographic quote mark.
import { MapPin } from "lucide-react";

// The badge palette, as a left rule. Carrying the section in the card's edge
// leaves the first line free for the sentence.
const SECTION_RULE: Record<string, string> = {
  Abstract:     "border-l-violet-400",
  Background:   "border-l-blue-400",
  Introduction: "border-l-blue-400",
  Methods:      "border-l-sky-400",
  Results:      "border-l-emerald-400",
  Discussion:   "border-l-amber-400",
  Conclusion:   "border-l-amber-400",
  Limitations:  "border-l-orange-400",
  References:   "border-l-slate-300",
};

export function sectionRuleClass(section?: string): string {
  return (section && SECTION_RULE[section]) || "border-l-slate-300 dark:border-l-slate-600";
}

export type EvidenceQuoteCardProps = {
  /** Position in the evidence list, which is how the viewer labels its marks. */
  index: number;
  quote: string;
  section?: string;
  /** Why the model offered this passage. */
  why?: string;
  /** The character range in the extracted text, for the tooltip. */
  start: number;
  end: number;
  /** The page it was found on, once the document viewer has looked. */
  page?: number;
  onLocate: () => void;
};

export function EvidenceQuoteCard({
  index, quote, section, why, start, end, page, onLocate,
}: EvidenceQuoteCardProps) {
  return (
    <button
      type="button"
      onClick={onLocate}
      title={`Show this passage in the text (characters ${start} to ${end})`}
      className={`group block w-full rounded-lg border border-l-[3px] bg-card p-3.5 text-left
                  transition-colors hover:border-primary/40 hover:bg-muted/30
                  ${sectionRuleClass(section)}`}
    >
      <div className="mb-1.5 flex items-baseline gap-2">
        {/* The document viewer labels its marks by this number, so the card has
            to carry it: without it the chips in the viewer's toolbar refer to
            nothing the reviewer can see. */}
        <span className="text-[11px] tabular-nums text-muted-foreground/60">{index + 1}</span>
        <span className="text-[11px] font-semibold uppercase tracking-[0.07em] text-muted-foreground">
          {section || "Quote"}
        </span>
        {page != null && (
          <span className="text-[11px] tabular-nums text-muted-foreground/70">p. {page}</span>
        )}
        <span className="ml-auto inline-flex shrink-0 items-center gap-1 text-[11px] text-primary
                         opacity-0 transition-opacity group-hover:opacity-100">
          <MapPin className="size-3" />Show in text
        </span>
      </div>

      {/* Real quotation marks, inline. A hanging decorative glyph was tried and
          read as a stray tick above the first line: too small to be punctuation
          and too odd to be an ornament. The marks belong in the text flow. */}
      <div className="break-words text-sm leading-6 text-foreground/90">
        <span className="text-muted-foreground/60">&ldquo;</span>
        {quote}
        <span className="text-muted-foreground/60">&rdquo;</span>
      </div>

      {why && <div className="mt-2 text-xs text-muted-foreground">{why}</div>}
    </button>
  );
}
