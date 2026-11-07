// Importing a document that is already in the review.
//
// Run: pnpm test
//
// A re-upload is matched to the paper the corpus already holds, so everything
// about it has to be keyed on THAT id rather than on the id the fresh parse
// invented. The OCR step was keyed on the parse, so for a document uploaded
// twice it asked the server about a paper that did not exist, got nothing back,
// and told the reviewer their files could not be read.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";

const importStudies = vi.fn();
const readTextOnServer = vi.fn();
const sendPdfsToServer = vi.fn(async (_ids: string[]) => 0);

// The real module, with only the file-reading parts replaced: guessTitle is
// part of what is under test here, since a document that could not be read in
// the browser gets its name from the server's read instead.
vi.mock("../pdfImport", async () => ({
  ...(await vi.importActual<any>("../pdfImport")),
  importStudies: (...a: any[]) => importStudies(...a),
  isAccepted: () => true,
}));
vi.mock("../pdfHandoff", () => ({
  readTextOnServer: (...a: any[]) => readTextOnServer(...a),
  sendPdfsToServer: (ids: string[]) => sendPdfsToServer(ids),
}));
vi.mock("../pdfBlobs", () => ({
  registerPdfBlobs: () => { /* no-op */ },
  removePdfBlob: () => { /* no-op */ },
}));
vi.mock("sonner", () => {
  const t: any = (...a: any[]) => t.message(...a);
  t.loading = vi.fn(() => "toast-id");
  t.success = vi.fn();
  t.error = vi.fn();
  t.warning = vi.fn();
  t.info = vi.fn();
  t.message = vi.fn();
  return { toast: t };
});

import { StoreProvider, useStore } from "../store";
import { useStudyImport } from "../useStudyImport";

const wrapper = ({ children }: { children: ReactNode }) => (
  <StoreProvider>{children}</StoreProvider>
);

/** The hook under test plus the store it writes into. */
function mount() {
  return renderHook(() => ({ imp: useStudyImport("table-extraction"), s: useStore() }), { wrapper });
}

const scanOf = (title: string, extra: Record<string, any> = {}) => ({
  paper: { id: `fresh_${Math.random().toString(36).slice(2)}`, title,
           source: "Local PDFs", url: "", abstract: "" },
  fullText: { paper_id: "ignored", title, url: "", source: "Local PDFs",
              status: "found", text: "" },
  objectBlob: new Blob(["%PDF"], { type: "application/pdf" }),
  objectMime: "application/pdf",
  ...extra,
});

beforeEach(() => {
  localStorage.clear();
  importStudies.mockReset();
  readTextOnServer.mockReset();
  readTextOnServer.mockResolvedValue({ text: "", method: "none" });
  sendPdfsToServer.mockClear();
});

describe("re-uploading a document the review already has", () => {
  it("reads it under the id the corpus gave it, not the new parse's", async () => {
    const { result } = mount();
    act(() => {
      result.current.s.setRawPapers([{ id: "existing-1", title: "A scanned trial",
                                       source: "Local PDFs", url: "", abstract: "" } as any]);
    });
    importStudies.mockResolvedValue({ studies: [scanOf("A scanned trial")], failed: [] });

    await act(async () => {
      await result.current.imp.importFiles([new File(["x"], "scan.pdf", { type: "application/pdf" })]);
    });

    expect(readTextOnServer).toHaveBeenCalledWith("existing-1");
  });

  it("does not forget text it already has when the same paper arrives as a scan", async () => {
    const { result } = mount();
    act(() => {
      result.current.s.setRawPapers([{ id: "existing-1", title: "A scanned trial",
                                       source: "Local PDFs", url: "", abstract: "" } as any]);
      result.current.s.setFullTexts({
        "existing-1": { paper_id: "existing-1", title: "A scanned trial", url: "",
                        source: "Local PDFs", status: "found", text: "The words we already had." } as any,
      });
    });
    importStudies.mockResolvedValue({ studies: [scanOf("A scanned trial")], failed: [] });

    await act(async () => {
      await result.current.imp.importFiles([new File(["x"], "scan.pdf", { type: "application/pdf" })]);
    });

    expect(result.current.s.fullTexts["existing-1"].text).toBe("The words we already had.");
    // Nothing to recover: the paper has its words.
    expect(readTextOnServer).not.toHaveBeenCalled();
  });
});

describe("a new document with no text layer", () => {
  it("is read on the server under its own id", async () => {
    const { result } = mount();
    importStudies.mockResolvedValue({ studies: [scanOf("A brand new scan")], failed: [] });
    readTextOnServer.mockResolvedValue({ text: "Recovered words.", method: "ocr" });

    await act(async () => {
      await result.current.imp.importFiles([new File(["x"], "scan.pdf", { type: "application/pdf" })]);
    });

    const ids = Object.keys(result.current.s.fullTexts);
    expect(ids).toHaveLength(1);
    expect(readTextOnServer).toHaveBeenCalledWith(ids[0]);
    expect(result.current.s.fullTexts[ids[0]].text).toBe("Recovered words.");
  });
});

describe("a document the browser could not read at all", () => {
  it("takes its name from the paper once the server has read it", async () => {
    // Until it is read, such a file is named after itself ("djab018"), and that
    // name follows it through every stage of the review.
    const { result } = mount();
    importStudies.mockResolvedValue({
      studies: [scanOf("djab018", { titleFromFilename: true, parseError: "worker failed to start" })],
      failed: [],
    });
    readTextOnServer.mockResolvedValue({
      method: "ocr",
      text: "Dopamine Receptor Antagonists, Radiation, and Cholesterol Biosynthesis\n"
          + "Kruttika Bhat, PhD, Mohammad Saki, PhD\n",
    });

    await act(async () => {
      await result.current.imp.importFiles([new File(["x"], "djab018.pdf", { type: "application/pdf" })]);
    });

    const titles = (result.current.s.rawPapers || []).map(p => p.title);
    expect(titles[0]).toMatch(/Dopamine Receptor Antagonists/);
  });

  it("keeps a title the document itself gave", async () => {
    const { result } = mount();
    importStudies.mockResolvedValue({
      studies: [scanOf("A properly titled trial", { titleFromFilename: false })],
      failed: [],
    });
    readTextOnServer.mockResolvedValue({ text: "Some other heading entirely here.", method: "ocr" });

    await act(async () => {
      await result.current.imp.importFiles([new File(["x"], "scan.pdf", { type: "application/pdf" })]);
    });

    expect((result.current.s.rawPapers || [])[0].title).toBe("A properly titled trial");
  });
});
