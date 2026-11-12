// The extraction workspace: source PDF, the table it came from, and the pooled
// result, side by side.
//
// The arrangement is the argument. A pooled effect is a claim about other
// people's data, and the only thing that makes it checkable is a short path
// from the number back to the page it was printed on. So the value, its
// provenance, and the source sit in one view rather than in three places the
// reviewer has to hold in their head.
//
// Two rules this component exists to enforce:
//
//   1. Every displayed number names where it came from, and whether the
//      extractor that produced it was deterministic.
//   2. A citation that could not be verified is shown as unverified rather than
//      quietly rendered like any other. A fabricated page reference is the one
//      error a reader cannot catch by reading more carefully.
import { useMemo, useState } from "react";
import { FileType2, ExternalLink, Search, Loader2, AlertTriangle, Check } from "lucide-react";

import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { metaAsk, cachedPdfUrl, type ExtractedTable, type GroundedAnswer } from "../lib/apiClient";
import { PdfViewer } from "./PdfViewer";

// ---------------------------------------------------------------------------
// Provenance badge
// ---------------------------------------------------------------------------

const TIER_LABEL: Record<string, string> = {
  jats: "Europe PMC JATS",
  pdfplumber: "pdfplumber",
  unitable: "image recognition",
  llm: "language model",
};

export function TierBadge({ tier, deterministic }: { tier?: string; deterministic?: boolean }) {
  if (!tier) return null;
  // Deterministic means the same PDF yields the same table every time, which is
  // what makes an extraction reproducible by someone else. The LLM tier is not,
  // and is coloured to say so rather than sitting quietly beside the others.
  const det = deterministic !== false && tier !== "llm";
  return (
    <span
      title={det
        ? "Deterministic: re-running this extraction on the same document reproduces this table exactly."
        : "Not deterministic: this table was produced by a language model and may differ on a re-run. Check it against the source."}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
        det ? "bg-primary/10 text-primary" : "bg-amber-100 text-amber-800"
      }`}
    >
      {TIER_LABEL[tier] || tier} · {det ? "deterministic" : "unverified"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Source pane
// ---------------------------------------------------------------------------

export function SourcePane({
  pdfKey, page, url, text, title,
}: {
  pdfKey?: string; page?: number | null; url?: string; text?: string; title?: string;
}) {
  // Without the page: the cited page moves this viewer by scrolling, and a
  // url that changed with it would reload the whole document instead.
  const src = pdfKey ? cachedPdfUrl(pdfKey) : (url && /\.pdf($|\?)/i.test(url) ? url : "");

  return (
    <div className="flex flex-col h-full min-h-0 rounded-lg border bg-card overflow-hidden">
      <div className="flex items-center gap-2 border-b px-3 py-2 text-xs">
        <FileType2 className="size-3.5 text-muted-foreground shrink-0" />
        <span className="truncate font-medium">{title || "Source"}</span>
        <span className="ml-auto shrink-0 text-muted-foreground">
          {page ? `p. ${page}` : ""}
        </span>
      </div>

      {src ? (
        // The cited page is a prop rather than a url fragment, so moving to
        // another one scrolls this viewer instead of reloading the document.
        <PdfViewer url={src} initialPage={page ?? undefined} externalUrl={url} hideTitle
                   className="min-h-0 flex-1 rounded-none border-0 border-t" />
      ) : text ? (
        <div className="flex-1 overflow-auto p-4 text-[13px] leading-6 whitespace-pre-wrap">
          {text}
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
          <FileType2 className="size-8 opacity-40" />
          <div>No source document for this study.</div>
          <div className="text-xs max-w-xs">
            Acquire the full text first, or open the record at its publisher.
          </div>
          {url && (
            <a href={url} target="_blank" rel="noreferrer"
               className="text-primary hover:underline inline-flex items-center gap-1 text-xs">
              Open the source <ExternalLink className="size-3" />
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Extracted table
// ---------------------------------------------------------------------------

export function TablePane({
  tables, selected, onSelect, onCite,
}: {
  tables: ExtractedTable[];
  selected: number;
  onSelect: (i: number) => void;
  onCite?: (page: number | null | undefined) => void;
}) {
  const t = tables[selected];

  if (!t) {
    return (
      <div className="rounded-lg border bg-card p-6 text-center text-sm text-muted-foreground">
        No tables extracted for this study yet.
      </div>
    );
  }

  const [head, ...body] = t.data || [];

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        {/* Name the table even when it is the only one. "Table 2" is the unit a
            reader cites, and for a JATS-sourced table it is the ONLY locator
            there is, since structured markup carries no page number. */}
        <span className="text-xs font-medium">{t.title || "Extracted table"}</span>
        <TierBadge tier={t.tier} deterministic={t.deterministic} />
        {t.page ? (
          <button
            onClick={() => onCite?.(t.page)}
            className="text-[11px] text-primary hover:underline"
            title="Show this page in the source pane"
          >
            p. {t.page} →
          </button>
        ) : null}
        {tables.length > 1 && (
          <select
            className="ml-auto rounded border bg-background px-1.5 py-0.5 text-[11px]"
            value={selected}
            onChange={e => onSelect(Number(e.target.value))}
          >
            {tables.map((x, i) => (
              <option key={i} value={i}>{x.title || `Table ${i + 1}`}</option>
            ))}
          </select>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] font-mono">
          {head && (
            <thead className="bg-muted/50">
              <tr>{head.map((c, i) => (
                <th key={i} className="px-3 py-2 text-left font-semibold whitespace-nowrap">{c}</th>
              ))}</tr>
            </thead>
          )}
          <tbody>
            {body.slice(0, 40).map((row, ri) => (
              <tr key={ri} className="border-t">
                {row.map((c, ci) => (
                  <td key={ci} className="px-3 py-1.5 whitespace-nowrap">{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {t.caption ? (
        <div className="border-t px-3 py-2 text-[11px] text-muted-foreground">{t.caption}</div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Grounded question box
// ---------------------------------------------------------------------------

function fmt(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return "";
  return String(Math.round(n * 1000) / 1000);
}

export function AskPane({
  paper, fullText, tables, onCite,
}: {
  paper: { id?: string; title?: string; abstract?: string };
  fullText?: string;
  tables?: ExtractedTable[];
  onCite?: (page: number | null | undefined) => void;
}) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [a, setA] = useState<GroundedAnswer | null>(null);
  const [err, setErr] = useState("");

  async function ask() {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true); setErr(""); setA(null);
    try {
      setA(await metaAsk(paper, question, { fullText, tables }));
    } catch (e: any) {
      setErr(e?.message?.slice(0, 160) || "The question could not be answered.");
    } finally {
      setBusy(false);
    }
  }

  const provenance = useMemo(() => {
    if (!a) return [];
    const bits: string[] = [];
    if (a.units) bits.push(`units: ${a.units}`);
    if (a.timepoint) bits.push(`time point: ${a.timepoint}`);
    const where = [a.source_page ? `p. ${a.source_page}` : "", a.source_table || ""]
      .filter(Boolean).join(", ");
    if (where) bits.push(`source: ${where}`);
    return bits;
  }, [a]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Search className="size-3.5 text-muted-foreground shrink-0" />
        <Input
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") ask(); }}
          placeholder="Ask this paper a question, e.g. What was the primary outcome at 12 months?"
          className="h-8 text-[13px]"
        />
        <Button size="sm" className="h-8" onClick={ask} disabled={busy || !q.trim()}>
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : "Ask"}
        </Button>
      </div>

      {err && (
        <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {err}
        </div>
      )}

      {a && (
        <div className="rounded-lg border bg-muted/30 px-4 py-3 space-y-1.5">
          {a.found ? (
            <>
              <div className="text-[15px] font-semibold">
                {a.answer}
                {a.ci_low !== null && a.ci_high !== null && (
                  <span className="ml-2 font-normal text-muted-foreground text-[13px]">
                    (95% CI {fmt(a.ci_low)} to {fmt(a.ci_high)})
                  </span>
                )}
              </div>

              {provenance.length > 0 && (
                <div className="text-[11px] text-muted-foreground">
                  {provenance.join(" · ")}
                  {a.source_page ? (
                    <>
                      {" · "}
                      <button onClick={() => onCite?.(a.source_page)}
                              className="text-primary hover:underline">
                        jump to PDF
                      </button>
                    </>
                  ) : null}
                </div>
              )}

              {/* The quote is the check. Saying whether it was actually found in
                  the source is the difference between a citation and a claim. */}
              {a.quote && (
                <div className={`mt-1 border-l-2 pl-2 text-[11px] italic ${
                  a.quote_verified
                    ? "border-primary/40 text-muted-foreground"
                    : "border-amber-500 text-amber-700"
                }`}>
                  {a.quote_verified
                    ? <Check className="mr-1 inline size-3 text-primary" />
                    : <AlertTriangle className="mr-1 inline size-3" />}
                  “{a.quote}”
                  {!a.quote_verified && (
                    <div className="mt-1 not-italic font-medium">
                      This quotation could not be located in the source text. Verify it
                      against the document before using this value.
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="text-[13px] text-muted-foreground">{a.answer}</div>
          )}
        </div>
      )}
    </div>
  );
}
