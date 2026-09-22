// Tests for the extraction workspace.
//
// Run: pnpm test
//
// This surface exists to make a pooled number checkable, so the tests are about
// provenance rather than appearance. Two properties carry the weight:
//
//   1. A table's extraction tier is stated, and the non-deterministic tier is
//      visibly distinguished. A reviewer reproducing the review needs to know
//      which numbers come back identical on a re-run and which may not.
//   2. A quotation that could NOT be located in the source is marked as such.
//      That is the only error a careful reader cannot catch by reading harder:
//      a fabricated citation looks exactly like a real one.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { SourcePane, TablePane, TierBadge, AskPane } from "../EvidenceWorkspace";
import type { ExtractedTable } from "../../lib/apiClient";

const TABLE: ExtractedTable = {
  title: "Table 2",
  type: "outcomes",
  page: 7,
  tier: "pdfplumber",
  deterministic: true,
  data: [["Outcome", "Metformin", "Placebo", "HR (95% CI)"],
         ["MACE", "142 / 1204", "189 / 1198", "0.74 (0.60-0.92)"],
         ["MI", "61 / 1204", "80 / 1198", "0.76 (0.54-1.05)"]],
};

const answer = (over: Record<string, unknown> = {}) => ({
  found: true, answer: "HbA1c -0.9%", value: -0.9, units: "%",
  ci_low: -1.2, ci_high: -0.6, timepoint: "12 months",
  source_page: 7, source_table: "Table 2",
  quote: "HbA1c fell by -0.9% (95% CI -1.2 to -0.6).",
  quote_verified: true, confidence: 0.9, ...over,
});

function mockAsk(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, headers: new Headers(),
    text: async () => JSON.stringify(body),
    json: async () => body,
  } as unknown as Response)));
}

beforeEach(() => { vi.unstubAllGlobals(); localStorage.clear(); });

// --------------------------------------------------------------------------
// Tier badge
// --------------------------------------------------------------------------

describe("TierBadge", () => {
  it.each(["pdfplumber", "jats", "unitable"])("marks the %s tier deterministic", (tier) => {
    render(<TierBadge tier={tier} deterministic />);
    expect(document.body.textContent).toMatch(/deterministic/i);
  });

  it("does NOT call the language-model tier deterministic", () => {
    /* An LLM-extracted table may differ on a re-run. Presenting it beside a
       pdfplumber table with the same styling would imply a reproducibility it
       does not have. */
    render(<TierBadge tier="llm" deterministic={false} />);
    expect(document.body.textContent).toMatch(/unverified/i);
    expect(document.body.textContent).not.toMatch(/·\s*deterministic/i);
  });

  it("treats the llm tier as non-deterministic even if the flag says otherwise", () => {
    render(<TierBadge tier="llm" deterministic />);
    expect(document.body.textContent).toMatch(/unverified/i);
  });

  it("names the human-readable source, not the internal tier key", () => {
    render(<TierBadge tier="jats" deterministic />);
    expect(document.body.textContent).toMatch(/Europe PMC/i);
  });

  it("renders nothing without a tier", () => {
    const { container } = render(<TierBadge />);
    expect(container.textContent).toBe("");
  });
});

// --------------------------------------------------------------------------
// Source pane
// --------------------------------------------------------------------------

describe("SourcePane", () => {
  const frame = () => document.querySelector("iframe") as HTMLIFrameElement | null;

  it("opens the PDF at the cited page", () => {
    /* The #page fragment is what makes "jump to p. 7" work without rendering
       the document ourselves. */
    render(<SourcePane pdfKey="abc123" page={7} title="A study" />);
    expect(frame()?.getAttribute("src")).toContain("#page=7");
  });

  it("omits the fragment when no page is cited", () => {
    render(<SourcePane pdfKey="abc123" title="A study" />);
    expect(frame()?.getAttribute("src") ?? "").not.toContain("#page=");
  });

  it("points at the cached-PDF route", () => {
    render(<SourcePane pdfKey="abc123" page={2} />);
    expect(frame()?.getAttribute("src")).toContain("/fulltext/pdf/abc123");
  });

  it("re-mounts when the page changes, so the viewer actually moves", () => {
    /* Mutating only the hash of a live iframe does not re-navigate it. */
    const { rerender } = render(<SourcePane pdfKey="abc" page={2} />);
    const first = frame();
    rerender(<SourcePane pdfKey="abc" page={9} />);
    expect(frame()?.getAttribute("src")).toContain("#page=9");
    expect(frame()).not.toBe(first);
  });

  it("shows the page number in the header", () => {
    render(<SourcePane pdfKey="abc" page={7} />);
    expect(document.body.textContent).toContain("p. 7");
  });

  it("falls back to the full text when there is no PDF", () => {
    render(<SourcePane text="The full text of the article." />);
    expect(screen.getByText(/full text of the article/i)).toBeInTheDocument();
    expect(frame()).toBeNull();
  });

  it("offers the publisher link when there is nothing to render", () => {
    render(<SourcePane url="https://example.org/article" />);
    expect(screen.getByText(/Open the source/i)).toBeInTheDocument();
  });

  it("renders a direct PDF url even without a cached key", () => {
    render(<SourcePane url="https://example.org/a.pdf" page={3} />);
    expect(frame()?.getAttribute("src")).toContain("a.pdf");
  });

  it("does not treat a non-PDF url as a PDF", () => {
    render(<SourcePane url="https://example.org/article" />);
    expect(frame()).toBeNull();
  });
});

// --------------------------------------------------------------------------
// Table pane
// --------------------------------------------------------------------------

describe("TablePane", () => {
  it("renders the header row and the body", () => {
    render(<TablePane tables={[TABLE]} selected={0} onSelect={() => {}} />);
    expect(screen.getByText("HR (95% CI)")).toBeInTheDocument();
    expect(screen.getByText("0.74 (0.60-0.92)")).toBeInTheDocument();
  });

  it("states the extraction tier", () => {
    render(<TablePane tables={[TABLE]} selected={0} onSelect={() => {}} />);
    expect(document.body.textContent).toMatch(/pdfplumber/i);
  });

  it("cites the page and jumps to it when clicked", () => {
    const onCite = vi.fn();
    render(<TablePane tables={[TABLE]} selected={0} onSelect={() => {}} onCite={onCite} />);
    fireEvent.click(screen.getByText(/p\. 7/));
    expect(onCite).toHaveBeenCalledWith(7);
  });

  it("offers a picker only when there is more than one table", () => {
    const { rerender } = render(<TablePane tables={[TABLE]} selected={0} onSelect={() => {}} />);
    expect(document.querySelector("select")).toBeNull();
    rerender(<TablePane tables={[TABLE, { ...TABLE, title: "Table 3" }]} selected={0} onSelect={() => {}} />);
    expect(document.querySelector("select")).not.toBeNull();
  });

  it("says so when there are no tables rather than rendering an empty grid", () => {
    render(<TablePane tables={[]} selected={0} onSelect={() => {}} />);
    expect(document.body.textContent).toMatch(/No tables extracted/i);
  });

  it("survives an out-of-range selection", () => {
    expect(() => render(<TablePane tables={[TABLE]} selected={9} onSelect={() => {}} />)).not.toThrow();
  });

  it("shows no page link for a table with no page", () => {
    render(<TablePane tables={[{ ...TABLE, page: null }]} selected={0} onSelect={() => {}} />);
    expect(document.body.textContent).not.toMatch(/p\. \d/);
  });
});

// --------------------------------------------------------------------------
// Grounded question box
// --------------------------------------------------------------------------

describe("AskPane", () => {
  const paper = { id: "p1", title: "A trial" };
  const ask = async (q = "What was the primary outcome at 12 months?") => {
    const box = screen.getByPlaceholderText(/Ask this paper/i);
    fireEvent.change(box, { target: { value: q } });
    fireEvent.click(screen.getByText("Ask"));
  };

  it("shows the answer and its interval", async () => {
    mockAsk(answer());
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() => expect(screen.getByText(/HbA1c -0.9%/)).toBeInTheDocument());
    expect(document.body.textContent).toMatch(/95% CI -1\.2 to -0\.6/);
  });

  it("shows units, time point and source", async () => {
    mockAsk(answer());
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() => expect(document.body.textContent).toMatch(/units: %/));
    expect(document.body.textContent).toMatch(/time point: 12 months/);
    expect(document.body.textContent).toMatch(/p\. 7, Table 2/);
  });

  it("jumps to the cited page", async () => {
    const onCite = vi.fn();
    mockAsk(answer());
    render(<AskPane paper={paper} onCite={onCite} />);
    await ask();
    await waitFor(() => expect(screen.getByText(/jump to PDF/i)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/jump to PDF/i));
    expect(onCite).toHaveBeenCalledWith(7);
  });

  it("FLAGS a quotation that could not be found in the source", async () => {
    /* The assertion this component exists for. An unverifiable citation must
       never render like a verified one. */
    mockAsk(answer({ quote_verified: false }));
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() =>
      expect(document.body.textContent).toMatch(/could not be located in the source/i));
  });

  it("does not warn when the quotation was verified", async () => {
    mockAsk(answer({ quote_verified: true }));
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() => expect(screen.getAllByText(/HbA1c/).length).toBeGreaterThan(0));
    expect(document.body.textContent).not.toMatch(/could not be located/i);
  });

  it("reports a not-found answer instead of an empty card", async () => {
    mockAsk({ found: false, answer: "Not reported in the text available for this paper.",
              value: null, quote: null, quote_verified: false });
    render(<AskPane paper={paper} />);
    await ask("What was the effect on mortality?");
    await waitFor(() => expect(document.body.textContent).toMatch(/Not reported/i));
  });

  it("shows no stale value alongside a not-found answer", async () => {
    mockAsk({ found: false, answer: "Not reported.", value: null, quote: null, quote_verified: false });
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() => expect(document.body.textContent).toMatch(/Not reported/));
    expect(document.body.textContent).not.toMatch(/95% CI/);
  });

  it("surfaces a backend failure rather than failing silently", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("backend is down"))));
    render(<AskPane paper={paper} />);
    await ask();
    await waitFor(() => expect(document.body.textContent).toMatch(/backend is down/i));
  });

  it("will not send an empty question", async () => {
    const spy = vi.fn();
    vi.stubGlobal("fetch", spy);
    render(<AskPane paper={paper} />);
    fireEvent.click(screen.getByText("Ask"));
    expect(spy).not.toHaveBeenCalled();
  });

  it("sends the tables along, since they hold the numbers", async () => {
    const spy = vi.fn(async () => ({
      ok: true, status: 200, headers: new Headers(),
      text: async () => JSON.stringify(answer()), json: async () => answer(),
    } as unknown as Response));
    vi.stubGlobal("fetch", spy);
    render(<AskPane paper={paper} tables={[TABLE]} fullText="body text" />);
    await ask();
    await waitFor(() => expect(spy).toHaveBeenCalled());
    const body = JSON.parse((spy.mock.calls[0] as any)[1].body);
    expect(body.tables).toHaveLength(1);
    expect(body.full_text).toBe("body text");
  });

  it("submits on Enter", async () => {
    mockAsk(answer());
    render(<AskPane paper={paper} />);
    const box = screen.getByPlaceholderText(/Ask this paper/i);
    fireEvent.change(box, { target: { value: "HR?" } });
    fireEvent.keyDown(box, { key: "Enter" });
    await waitFor(() => expect(screen.getAllByText(/HbA1c/).length).toBeGreaterThan(0));
  });
});
