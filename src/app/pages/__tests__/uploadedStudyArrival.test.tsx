// A document uploaded at one stage, seen at another.
//
// Run: pnpm test
//
// The stages are a menu: a reviewer uploads PDFs wherever they happen to be and
// expects to find them wherever they go next. Two ways that failed, both silent.
// The record has to carry across, and the stage has to come up ready to work
// rather than in an empty state; and a scanned PDF,
// which imports perfectly and contains no words, simply never appeared there at
// all, under an empty state telling the reviewer to go and fetch a full text
// they had just supplied.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, act } from "@testing-library/react";

import { StoreProvider, useStore } from "../../lib/store";
import { TextExtractionPage } from "../TextExtractionPage";

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

/** The store exactly as an import at Table Extraction leaves it. */
const importedAtTableExtraction = (text: string) => (s: any) => {
  const paper = { id: "p1", title: "My uploaded paper", source: "Local PDFs", url: "", abstract: "" };
  s.setRawPapers([paper]);
  s.setUniquePapers([paper]);
  s.setResults([{
    paper_id: "p1", Title: "My uploaded paper", Source: "Local PDFs", URL: "", Abstract: "",
    Decision: "INCLUDE", Reason: "Imported at Table Extraction.", Agent_Trace: {},
    Imported: "table-extraction",
  }]);
  s.setFullTextResults([{
    paper_id: "p1", Title: "My uploaded paper", Source: "Local PDFs", URL: "", Abstract: "",
    Decision: "Include", Reason: "Imported at Table Extraction.", Imported: "table-extraction",
  }]);
  s.setFullTexts({
    p1: { paper_id: "p1", title: "My uploaded paper", url: "", source: "Local PDFs",
          status: "found", text },
  });
};

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, status: 200, headers: new Headers(),
    text: async () => "{}", json: async () => ({}),
  })));
});

describe("a PDF uploaded at another stage", () => {
  it("leaves the stage ready to work, not in an empty state", () => {
    const { container } = renderPage(importedAtTableExtraction("Body text of the paper."));
    expect(container.textContent).toMatch(/ready to extract/i);
    expect(container.textContent).not.toMatch(/No full texts yet/i);
  });

  it("says the reading was tried and failed, when a file has no text at all", () => {
    // It is not missing: every other stage has it, and the pages were already
    // read on import, image pass included. So the page says that, rather than
    // sending the reviewer to acquire what they supplied or to run OCR
    // themselves, which is work the app can do and now does.
    const { container } = renderPage(importedAtTableExtraction(""));
    expect(container.textContent).toMatch(/nothing could be read/i);
    expect(container.textContent).toMatch(/page images/i);
    expect(container.textContent).not.toMatch(/Go to Full-Text Acquisition/i);
    expect(container.textContent).not.toMatch(/run OCR/i);
  });
});
