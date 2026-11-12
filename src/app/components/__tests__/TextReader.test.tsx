// The reading view may reflow the text, but two contracts have to survive it:
// every evidence span is still marked, and every mark still carries the raw
// offset the page scrolls to. "Show in text" finds its target by querying
// [data-span-start], so a mark without one is a dead link.
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { TextReader } from "../TextReader";

const TEXT = [
  "Methods",
  "We randomised 240 adults with impaired glucose tolerance to a 12-month",
  "programme or to usual care.",
  "",
  "Results",
  "Mean HbA1c fell by 0.8% in the intervention arm.",
].join("\n");

const at = (s: string) => {
  const start = TEXT.indexOf(s);
  return { start, end: start + s.length };
};

const METHODS = { ...at("We randomised 240 adults"), section: "Methods" };
const RESULTS = { ...at("Mean HbA1c fell by 0.8%"), section: "Results" };

const marks = () => Array.from(document.querySelectorAll("mark"));

describe("TextReader", () => {
  it("says so rather than rendering an empty frame", () => {
    render(<TextReader text="" />);
    expect(screen.getByText(/No full text available/i)).toBeInTheDocument();
  });

  it("reflows the text into prose, not into one line per source line", () => {
    const { container } = render(<TextReader text={TEXT} />);
    expect(container.querySelector("p")?.textContent).toBe(
      "We randomised 240 adults with impaired glucose tolerance to a 12-month programme or to usual care.",
    );
  });

  it("sets the sections as headings", () => {
    const { container } = render(<TextReader text={TEXT} />);
    expect(Array.from(container.querySelectorAll("h4")).map(h => h.textContent))
      .toEqual(["Methods", "Results"]);
  });

  it("marks every span, and tags each mark with the offset the page scrolls to", () => {
    render(<TextReader text={TEXT} spans={[METHODS, RESULTS]} />);
    expect(marks()).toHaveLength(2);
    expect(marks().map(m => m.getAttribute("data-span-start")))
      .toEqual([String(METHODS.start), String(RESULTS.start)]);
  });

  it("draws the span the reviewer asked for differently from the rest", () => {
    render(<TextReader text={TEXT} spans={[METHODS, RESULTS]} activeSpan={[RESULTS.start, RESULTS.end]} />);
    const [methods, results] = marks();
    expect(results.className).toContain("ring-2");
    expect(methods.className).not.toContain("ring-2");
  });

  it("keeps the offsets through the reflow, not through the raw line breaks", () => {
    // The span runs across a line break the reading view has just closed up,
    // which is exactly the case a naive reflow breaks.
    const across = at("12-month\nprogramme");
    render(<TextReader text={TEXT} spans={[across]} />);
    expect(marks()).toHaveLength(1);
    expect(marks()[0].textContent).toBe("12-month programme");
    expect(marks()[0].getAttribute("data-span-start")).toBe(String(across.start));
  });

  it("names the section and the offsets in the tooltip, not in the body", () => {
    render(<TextReader text={TEXT} spans={[METHODS]} />);
    expect(marks()[0].getAttribute("title"))
      .toBe(`Methods, chars ${METHODS.start}-${METHODS.end}`);
  });

  it("shows the raw text verbatim, numbered, when asked to", () => {
    const { container } = render(<TextReader text={TEXT} spans={[METHODS]} view="raw" />);
    const rows = container.querySelectorAll("tr");
    expect(rows).toHaveLength(TEXT.split("\n").length);
    // Line 2 is still broken where the source broke it.
    expect(rows[1].textContent).toContain("to a 12-month");
    expect(rows[1].textContent).not.toContain("programme or to usual care");
  });

  it("marks the raw view too, so a locate works in either", () => {
    render(<TextReader text={TEXT} spans={[METHODS, RESULTS]} view="raw" />);
    expect(marks().map(m => m.getAttribute("data-span-start")))
      .toEqual([String(METHODS.start), String(RESULTS.start)]);
  });

  it("marks nothing when there is nothing to mark", () => {
    render(<TextReader text={TEXT} />);
    expect(marks()).toHaveLength(0);
  });
});
