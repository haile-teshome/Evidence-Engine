// Handing the reviewer's own PDF to the backend.
//
// Run: pnpm test
//
// An imported paper is parsed in the browser, so the server only ever saw the
// text. Every reader that works on the file was therefore skipped for exactly
// the papers a reviewer supplied themselves, which is how two PDFs with four
// tables between them came back with none. The rules worth pinning down: the
// file goes over once, a paper with no stored file is not an error, and a
// failed handover never takes the stage down with it, because it is one rung of
// a ladder and the others still run.
import { describe, it, expect, vi, beforeEach } from "vitest";

const getPdfFile = vi.fn();
const uploadPdf = vi.fn();
const readPdfText = vi.fn();

vi.mock("../pdfBlobs", () => ({ getPdfFile: (id: string) => getPdfFile(id) }));
vi.mock("../apiClient", () => ({
  AIService: {
    uploadPdf: (...a: any[]) => uploadPdf(...a),
    readPdfText: (...a: any[]) => readPdfText(...a),
  },
}));

import {
  sendPdfToServer, sendPdfsToServer, readTextOnServer, _resetPdfHandoff,
} from "../pdfHandoff";

const pdf = () => new Blob(["%PDF-1.7"], { type: "application/pdf" });

beforeEach(() => {
  _resetPdfHandoff();
  getPdfFile.mockReset();
  uploadPdf.mockReset();
  uploadPdf.mockResolvedValue(undefined);
  readPdfText.mockReset();
  readPdfText.mockResolvedValue({ text: "", method: "none" });
});

describe("sending a stored PDF", () => {
  it("sends the file under its paper id", async () => {
    const file = pdf();
    getPdfFile.mockResolvedValue(file);

    expect(await sendPdfToServer("p1")).toBe(true);
    expect(uploadPdf).toHaveBeenCalledWith("p1", file);
  });

  it("sends each paper only once", async () => {
    getPdfFile.mockResolvedValue(pdf());

    await sendPdfToServer("p1");
    await sendPdfToServer("p1");
    expect(uploadPdf).toHaveBeenCalledTimes(1);
  });

  it("is not an error for a paper with no stored file", async () => {
    // A title pasted into the list has no PDF behind it, which is normal.
    getPdfFile.mockResolvedValue(undefined);

    expect(await sendPdfToServer("p1")).toBe(false);
    expect(uploadPdf).not.toHaveBeenCalled();
  });

  it("swallows a failed upload rather than failing the stage", async () => {
    getPdfFile.mockResolvedValue(pdf());
    uploadPdf.mockRejectedValue(new Error("backend down"));

    expect(await sendPdfToServer("p1")).toBe(false);
  });

  it("retries a paper whose upload failed", async () => {
    getPdfFile.mockResolvedValue(pdf());
    uploadPdf.mockRejectedValueOnce(new Error("backend down"));

    expect(await sendPdfToServer("p1")).toBe(false);
    expect(await sendPdfToServer("p1")).toBe(true);
  });

  it("ignores a blank id", async () => {
    expect(await sendPdfToServer("")).toBe(false);
    expect(getPdfFile).not.toHaveBeenCalled();
  });
});

describe("sending several", () => {
  it("counts the ones that went over", async () => {
    getPdfFile.mockImplementation(async (id: string) => (id === "p2" ? undefined : pdf()));

    expect(await sendPdfsToServer(["p1", "p2", "p3"])).toBe(2);
    expect(uploadPdf).toHaveBeenCalledTimes(2);
  });

  it("goes one at a time so a folder of PDFs cannot flood the API", async () => {
    let live = 0;
    let peak = 0;
    getPdfFile.mockResolvedValue(pdf());
    uploadPdf.mockImplementation(async () => {
      live++;
      peak = Math.max(peak, live);
      await new Promise(r => setTimeout(r, 1));
      live--;
    });

    await sendPdfsToServer(["a", "b", "c", "d"]);
    expect(peak).toBe(1);
  });
});

describe("reading a document the browser could not", () => {
  it("sends the file, then asks the server to read it", async () => {
    // A scanned PDF: the browser parsed it and got nothing, so the pages have
    // to be read on the other side. The reviewer is not asked to do anything.
    getPdfFile.mockResolvedValue(pdf());
    readPdfText.mockResolvedValue({ text: "Sample Characteristics n = 272", method: "ocr" });

    const got = await readTextOnServer("p1");
    expect(got.text).toContain("272");
    expect(got.method).toBe("ocr");
    expect(uploadPdf).toHaveBeenCalled();
    expect(readPdfText).toHaveBeenCalledWith("p1");
  });

  it("distinguishes a file that never left the browser", async () => {
    // The failure the reviewer saw, and the one that says nothing at all about
    // their document: it had not been handed over when the read was asked for.
    getPdfFile.mockResolvedValue(undefined);

    expect(await readTextOnServer("p1")).toEqual({ text: "", method: "unsent" });
    expect(readPdfText).not.toHaveBeenCalled();
  });

  it("distinguishes a reader that is not answering", async () => {
    getPdfFile.mockResolvedValue(pdf());
    readPdfText.mockRejectedValue(new Error("backend down"));

    expect(await readTextOnServer("p1")).toEqual({ text: "", method: "unreachable" });
  });

  it("passes through a read that genuinely found nothing", async () => {
    getPdfFile.mockResolvedValue(pdf());
    readPdfText.mockResolvedValue({ text: "", method: "none" });

    expect(await readTextOnServer("p1")).toEqual({ text: "", method: "none" });
  });
});
