// Seeing where an extracted value came from.
//
// Run: pnpm test
//
// A number in a spreadsheet and a number a model invented look identical. The
// sentence it was read from is the only thing that tells them apart, and until
// now it lived in a title attribute on a 6-pixel dot: not selectable, with no
// surrounding text, and with no way to tell a sentence that is in the paper
// from one that is not.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, act, fireEvent, screen } from "@testing-library/react";

import { StoreProvider, useStore } from "../../lib/store";
import { TextExtractionPage } from "../TextExtractionPage";

const PAPER_TEXT =
  "Patients were grouped as follows: (1) use of levetiracetam, (2) use of lacosamide.\n\n"
  + "The median OS of our cohort (n = 272) was 15.64 months.\n\n"
  + "Both drugs were generally well tolerated.";

function renderPage(seed: (s: any) => void) {
  let api: any;
  function Capture() { api = useStore(); return null; }
  const utils = render(
    <StoreProvider>
      <Capture />
      <TextExtractionPage />
    </StoreProvider>,
  );
  act(() => seed(api));
  return utils;
}

const seedReview = (s: any) => {
  s.setResults([{ paper_id: "p1", Title: "Bianconi 2024", Source: "Local PDFs", URL: "",
                  Abstract: "a", Decision: "INCLUDE", Reason: "", Agent_Trace: {} }]);
  s.setFullTexts({
    p1: { paper_id: "p1", title: "Bianconi 2024", url: "", source: "Local PDFs",
          status: "found", text: PAPER_TEXT },
  });
};

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, headers: new Headers(),
    text: async () => "{}", json: async () => ({}),
  })));
});

/** One finished extraction row, as a run leaves it. */
const rowWith = (quote: string, value = "15.64") => ([{
  paper_id: "p1", title: "Bianconi 2024",
  values: { design: value },
  meta: { design: { source_quote: quote, confidence: "high", needs_review: false } },
  status: { design: "accepted" },
}]);

describe("a value's provenance", () => {
  it("shows the quote in the paper, with what surrounds it", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.")));
    renderPage(seedReview);

    // The first column is the field these rows carry.
    fireEvent.click(screen.getAllByTitle("Click for the source, double-click to edit")[0]);

    // The passage, with the sentence before it: a quote with no surroundings
    // cannot be checked against the paper. Nothing is said about the match
    // itself, because the highlighted passage is the answer.
    expect(document.body.textContent).toContain("Patients were grouped as follows");
    expect(document.body.textContent).toContain("15.64 months");
    expect(screen.queryByText(/not in the document/i)).toBeNull();
  });

  it("says plainly when the sentence is not in the paper", () => {
    // The case the whole mechanism exists to catch.
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("Patients taking levetiracetam lived twice as long.")));
    renderPage(seedReview);

    // The first column is the field these rows carry.
    fireEvent.click(screen.getAllByTitle("Click for the source, double-click to edit")[0]);

    expect(screen.getByText(/not in the document/i)).toBeTruthy();
    expect(screen.getByText(/unverified/i)).toBeTruthy();
  });

  it("explains an empty cell the model answered", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify([{
      paper_id: "p1", title: "Bianconi 2024",
      values: { design: "" },
      meta: { design: { source_quote: "", confidence: "none", needs_review: true } },
      status: { design: "accepted" },
    }]));
    renderPage(seedReview);

    // The first column is the field these rows carry.
    fireEvent.click(screen.getAllByTitle("Click for the source, double-click to edit")[0]);

    expect(screen.getByText(/found nothing for this field/i)).toBeTruthy();
  });

  it("distinguishes a field that was never asked", () => {
    // No meta at all: its group of fields failed, which is not the same as the
    // paper not reporting it, and is fixed by running again rather than by hand.
    localStorage.setItem("ee:extract-rows", JSON.stringify([{
      paper_id: "p1", title: "Bianconi 2024", values: {}, meta: {}, status: {},
    }]));
    renderPage(seedReview);

    // The first column is the field these rows carry.
    fireEvent.click(screen.getAllByTitle("Click for the source, double-click to edit")[0]);

    expect(screen.getByText(/never answered/i)).toBeTruthy();
  });
});

describe("correcting an extracted value", () => {
  const openCell = () =>
    fireEvent.click(screen.getAllByTitle("Click for the source, double-click to edit")[0]);

  it("saves what the reviewer typed, and says the model's value was changed", () => {
    // The model read 15.64 from the wrong row; the reviewer, looking at the
    // quote beside the box, corrects it. Theirs is the value that counts.
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    renderPage(seedReview);

    openCell();
    const box = document.querySelector("textarea")!;
    fireEvent.change(box, { target: { value: "17.55" } });
    fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));

    expect(document.body.textContent).toContain("17.55");
    openCell();
    // The value it restores is on the button, not in its label.
    expect(screen.getByRole("button", { name: /^Undo$/ })).toHaveAttribute(
      "title", expect.stringContaining("15.64"));
  });

  it("puts the model's value back when the edit is undone", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    renderPage(seedReview);

    openCell();
    fireEvent.change(document.querySelector("textarea")!, { target: { value: "17.55" } });
    fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));

    openCell();
    fireEvent.click(screen.getByRole("button", { name: /^Undo$/ }));

    expect(document.body.textContent).toContain("15.64");
    expect(document.body.textContent).not.toContain("17.55");
  });

  it("does not offer to save a value nobody changed", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    renderPage(seedReview);

    openCell();
    expect(screen.getByRole("button", { name: /^Save$/ })).toBeDisabled();
  });

  it("can be edited in the table itself", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    renderPage(seedReview);

    const cell = screen.getAllByTitle("Click for the source, double-click to edit")[0];
    fireEvent.doubleClick(cell);
    const box = document.querySelector("textarea")!;
    fireEvent.change(box, { target: { value: "17.55" } });
    fireEvent.blur(box);

    expect(document.body.textContent).toContain("17.55");
  });
});

describe("keeping or rejecting a value", () => {
  it("offers both on a value the model was sure of", () => {
    // Confidence is the model's, not the reviewer's: a cell it was certain
    // about can still be wrong, and there has to be a way to say so.
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    renderPage(seedReview);

    expect(screen.getAllByTitle(/keep this value|kept/i).length).toBeGreaterThan(0);
    expect(screen.getAllByTitle(/reject this value|rejected/i).length).toBeGreaterThan(0);
  });

  it("strikes a value through when it is rejected", () => {
    localStorage.setItem("ee:extract-rows", JSON.stringify(
      rowWith("The median OS of our cohort (n = 272) was 15.64 months.", "15.64")));
    const { container } = renderPage(seedReview);

    fireEvent.click(screen.getAllByTitle("Reject this value")[0]);
    expect(container.querySelector(".line-through")).not.toBeNull();
  });
});

// A blank cell said nothing about why it was blank. "The paper does not report
// this" is a finding a review records; "nothing read this field" is work left
// to do. They were the same dash.
describe("an empty cell in the table", () => {
  const seedRow = (cell: Record<string, string> | undefined) => {
    localStorage.setItem("ee:extract-rows", JSON.stringify([{
      paper_id: "p1", title: "Bianconi 2024",
      values: { design: "" },
      meta: { design: { source_quote: "", confidence: "none", needs_review: true } },
      status: { design: "accepted" },
      ...(cell ? { cell } : {}),
    }]));
  };

  // Both empty cases read as a dash, because a column of "Not reported" beside
  // a hundred others is unreadable. What tells them apart is the hover text, so
  // that is what is pinned here.
  const hover = () =>
    document.querySelector("td span[title]")?.getAttribute("title") || "";

  it("says the model read the article and found nothing", () => {
    seedRow({ design: "not_reported" });
    renderPage(seedReview);
    expect(document.body.textContent).toContain("\u2014");
    expect(hover()).toMatch(/read the article and found no value/i);
  });

  it("says plainly when nothing has read the field yet", () => {
    seedRow({ design: "unread" });
    renderPage(seedReview);
    expect(document.body.textContent).toContain("\u2014");
    expect(hover()).toMatch(/nothing read this field/i);
  });

  it("claims nothing for rows extracted before the status existed", () => {
    // Never "the paper does not report it" by default: that would put a
    // finding nobody established into the review.
    seedRow(undefined);
    renderPage(seedReview);
    expect(hover()).not.toMatch(/found no value/i);
  });
});
