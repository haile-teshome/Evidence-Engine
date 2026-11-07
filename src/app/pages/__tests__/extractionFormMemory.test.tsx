// Does the extraction form survive leaving the page?
//
// Run: pnpm test
//
// This stage is one of ten and the reviewer walks between them. The form is
// built by hand, or imported from a workbook of seventy fields, so losing it to
// a click on another stage or to a reload is losing real work. The page is
// unmounted by both, which is why none of this can live in component state
// alone. What is remembered also has to be checked on the way back in: a row
// for a paper this review does not have was somebody else's result.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";

import { StoreProvider, useStore } from "../../lib/store";
import { TextExtractionPage } from "../TextExtractionPage";

function renderPage(seed: (s: any) => void) {
  let api: any;
  function Capture() {
    api = useStore();
    return null;
  }
  const utils = render(
    <StoreProvider>
      <Capture />
      <TextExtractionPage />
    </StoreProvider>,
  );
  act(() => seed(api));
  return utils;
}

/** A review that has got as far as this stage: screened, with a full text. */
const seedReview = (s: any) => {
  s.setResults([{ paper_id: "p1", Title: "Study p1", Source: "PubMed", URL: "",
                  Abstract: "a", Decision: "INCLUDE", Reason: "", Agent_Trace: {} }]);
  s.setFullTexts({
    p1: { paper_id: "p1", title: "Study p1", url: "", source: "PubMed",
          status: "found", text: "body" },
  });
};

const openForm = () => fireEvent.click(screen.getByRole("button", { name: /^Form$/ }));

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, headers: new Headers(),
    text: async () => "{}", json: async () => ({}),
  })));
});

describe("the extraction form across a visit", () => {
  it("keeps an edited field, and stays on the form half", () => {
    const first = renderPage(seedReview);
    openForm();
    fireEvent.change(screen.getByDisplayValue("Sample size"),
                     { target: { value: "Participants analysed" } });
    first.unmount();

    renderPage(seedReview);
    expect(screen.getByDisplayValue("Participants analysed")).toBeTruthy();
  });

  it("keeps the tabs", () => {
    const first = renderPage(seedReview);
    openForm();
    fireEvent.click(screen.getByRole("button", { name: "+ Tab" }));
    first.unmount();

    renderPage(seedReview);
    // The flat form became the first tab rather than losing its fields to the
    // new empty one.
    expect(screen.getByRole("button", { name: "Extraction" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Tab 2" })).toBeTruthy();
  });

  it("does not show rows remembered from another review", () => {
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-rows", JSON.stringify([
      { paper_id: "ghost", title: "Belongs to another review",
        values: { design: "RCT" }, meta: {}, status: {} },
    ]));

    renderPage(seedReview);
    expect(screen.queryByText("Belongs to another review")).toBeNull();
  });

  it("keeps rows for papers this review does have", () => {
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-rows", JSON.stringify([
      { paper_id: "p1", title: "Study p1", values: { design: "RCT" }, meta: {}, status: {} },
    ]));

    renderPage(seedReview);
    expect(screen.getAllByText("Study p1").length).toBeGreaterThan(0);
  });

  it("restores the tabs a previous visit saved", () => {
    // The refresh case proper: nothing is in memory, only what was written.
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-form", JSON.stringify([
      { id: "design_0_0", label: "Study design", type: "text", options: [] },
      { id: "n_1_0", label: "Sample size", type: "number", options: [] },
    ]));
    localStorage.setItem("ee:extract-tabs", JSON.stringify([
      { name: "Characteristics", fieldIds: ["design_0_0"], studies: null },
      { name: "Outcomes", fieldIds: ["n_1_0"], studies: null },
    ]));

    renderPage(seedReview);
    expect(screen.getByRole("button", { name: "Characteristics" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Outcomes" })).toBeTruthy();
  });

  it("does not reopen the workbook designer by itself", () => {
    // The draft is kept, and offered. A workbook imported once must not greet
    // the reviewer on every visit to the stage: the form this stage ships with
    // is what should be in front of them.
    const field = (label: string) => ({ label, type: "text", options: [],
                                        description: "", include: true });
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-design", JSON.stringify([
      { name: "Identification", role: "extract", fields: [field("First author")] },
    ]));

    renderPage(seedReview);
    expect(screen.queryByText("Fields found in your workbook")).toBeNull();
    expect(screen.getByRole("button", { name: /resume import/i })).toBeTruthy();
  });

  it("reopens it on request, where it was left", () => {
    // Sorting seventy fields into five tabs is work; a reload in the middle of
    // it used to start the import again from the file.
    const field = (label: string) => ({ label, type: "text", options: [],
                                        description: "", include: true });
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-design", JSON.stringify([
      { name: "Identification", role: "extract", fields: [field("First author")] },
      { name: "Clinical Survival", role: "extract", fields: [field("Median OS")] },
    ]));

    renderPage(seedReview);
    fireEvent.click(screen.getByRole("button", { name: /resume import/i }));
    expect(screen.getByText("Fields found in your workbook")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Identification" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Clinical Survival" })).toBeTruthy();
  });

  it("Reset brings back the fields this stage ships with, and forgets the workbook", () => {
    // An imported workbook is a choice for one review, not a new default. Reset
    // has to undo all of it: the fields, the tabs, and the half-finished design
    // that would otherwise still be sitting there to resume.
    const field = (label: string) => ({ label, type: "text", options: [],
                                        description: "", include: true });
    localStorage.setItem("ee:extract-mode", JSON.stringify("form"));
    localStorage.setItem("ee:extract-form", JSON.stringify([
      { id: "covidence_0_0", label: "Covidence ID", type: "text", options: [] },
    ]));
    localStorage.setItem("ee:extract-tabs", JSON.stringify([
      { name: "Identification", fieldIds: ["covidence_0_0"], studies: null },
    ]));
    localStorage.setItem("ee:extract-design", JSON.stringify([
      { name: "Identification", role: "extract", fields: [field("Covidence ID")] },
    ]));

    renderPage(seedReview);
    expect(screen.getByDisplayValue("Covidence ID")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^Reset$/ }));

    expect(screen.getByDisplayValue("Sample size")).toBeTruthy();
    expect(screen.queryByDisplayValue("Covidence ID")).toBeNull();
    expect(screen.queryByRole("button", { name: /resume import/i })).toBeNull();
  });

  it("starts from the defaults when nothing was saved", () => {
    renderPage(seedReview);
    openForm();
    expect(screen.getByDisplayValue("Sample size")).toBeTruthy();
  });
});
