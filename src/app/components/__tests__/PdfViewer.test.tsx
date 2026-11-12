// The viewer is mostly geometry and lifecycle, so pdf.js is replaced with a
// document of known size and known text. What is worth testing here is the
// things an iframe got wrong: rasterising at device density, marking the quote,
// and saying so when the quote is not in the file.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// ── pdf.js stand-in ─────────────────────────────────────────────────────────
// One text item per page, positioned so itemBox has something to work with.
const PAGE_TEXT: Record<number, string> = {
  1: "The trial enrolled 240 adults.",
  2: "Mean HbA1c fell by 0.8% in the intervention arm.",
  3: "No serious adverse events were reported.",
};

const renderCalls: { page: number; width: number; height: number }[] = [];
// Long enough that a second zoom lands while the first render is in flight.
let RENDER_MS = 0;
const busy = new Set<any>();

function fakePage(n: number) {
  return {
    getViewport: ({ scale }: { scale: number }) => ({
      width: 612 * scale,
      height: 792 * scale,
      transform: [scale, 0, 0, -scale, 0, 792 * scale],
    }),
    getTextContent: async () => ({
      items: [{
        str: PAGE_TEXT[n] ?? "",
        transform: [1, 0, 0, 1, 72, 700],
        width: 300,
        height: 11,
      }],
    }),
    render: ({ viewport, canvas }: any) => {
      renderCalls.push({ page: n, width: viewport.width, height: viewport.height });
      // pdf.js refuses to draw two things into one canvas, and that refusal is
      // the whole point of the test below, so the stand-in has to make it too.
      if (busy.has(canvas)) {
        return {
          promise: Promise.reject(new Error("Cannot use the same canvas during multiple render() operations.")),
          cancel: () => {},
        };
      }
      busy.add(canvas);
      let cancelled = false;
      return {
        promise: new Promise<void>((resolve, reject) =>
          setTimeout(() => {
            busy.delete(canvas);
            if (!cancelled) return resolve();
            const e: any = new Error("cancelled");
            e.name = "RenderingCancelledException";
            reject(e);
          }, RENDER_MS)),
        cancel: () => { cancelled = true; },
      };
    },
  };
}

const getDocument = vi.fn(() => ({
  promise: Promise.resolve({
    numPages: 3,
    getPage: async (n: number) => fakePage(n),
    destroy: () => {},
  }),
}));

vi.mock("pdfjs-dist", () => ({
  GlobalWorkerOptions: { workerSrc: "" },
  get getDocument() { return getDocument; },
}));
vi.mock("pdfjs-dist/build/pdf.worker.min.mjs?url", () => ({ default: "worker.js" }));

const getPdfFile = vi.fn(async (_id: string) => new Blob(["%PDF-1.7"], { type: "application/pdf" }));
vi.mock("../../lib/pdfBlobs", () => ({ getPdfFile: (id: string) => getPdfFile(id) }));

import { PdfViewer } from "../PdfViewer";

const sheets = () => Array.from(document.querySelectorAll("[data-page]")) as HTMLElement[];
const marks = () => Array.from(document.querySelectorAll("[aria-hidden].absolute")) as HTMLElement[];

beforeEach(() => {
  renderCalls.length = 0;
  RENDER_MS = 0;
  busy.clear();
  getPdfFile.mockResolvedValue(new Blob(["%PDF-1.7"], { type: "application/pdf" }));
  // jsdom's stub returns null, which would stop every render before it starts.
  (HTMLCanvasElement.prototype as any).getContext = () => ({});
  Object.defineProperty(window, "devicePixelRatio", { configurable: true, value: 2 });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("PdfViewer", () => {
  it("lays out one sheet per page, sized from the page itself", async () => {
    render(<PdfViewer paperId="p1" title="Trial report" />);
    await waitFor(() => expect(sheets()).toHaveLength(3));
    // 612 x 792 at scale 1, because jsdom reports no pane width to fit to.
    expect(sheets()[0].style.width).toBe("612px");
    expect(sheets()[0].style.height).toBe("792px");
  });

  it("shows the page count and the document title", async () => {
    render(<PdfViewer paperId="p1" title="Trial report" />);
    await waitFor(() => expect(screen.getByText("1 / 3")).toBeInTheDocument());
    expect(screen.getByText("Trial report")).toBeInTheDocument();
  });

  it("rasterises at device pixel ratio, not at CSS size", async () => {
    render(<PdfViewer paperId="p1" />);
    await waitFor(() => expect(renderCalls.length).toBeGreaterThan(0));
    // The sheet is 612 CSS px wide; the bitmap behind it is twice that.
    expect(renderCalls[0]).toMatchObject({ page: 1, width: 1224, height: 1584 });
    const canvas = sheets()[0].querySelector("canvas") as HTMLCanvasElement;
    expect(canvas.width).toBe(1224);
    expect(canvas.style.width).toBe("612px");
  });

  it("marks a quote on the page it is printed on", async () => {
    render(<PdfViewer paperId="p1" quotes={["Mean HbA1c fell by 0.8%"]} />);
    await waitFor(() => expect(marks().length).toBeGreaterThan(0));
    // Page 2 holds that sentence, so the mark belongs to the second sheet.
    expect(sheets()[1].querySelectorAll("[aria-hidden]").length).toBeGreaterThan(0);
    expect(sheets()[0].querySelectorAll("[aria-hidden]").length).toBe(0);
  });

  it("says so when the sentence is not in the file", async () => {
    render(<PdfViewer paperId="p1" quotes={["a sentence from a different paper"]} />);
    await waitFor(() => expect(screen.getByText(/Not found in this file/i)).toBeInTheDocument());
    expect(marks()).toHaveLength(0);
  });

  it("offers a chip per quote when there are several, and none for one", async () => {
    const { unmount } = render(<PdfViewer paperId="p1" quotes={["The trial enrolled 240 adults", "No serious adverse events"]} />);
    await waitFor(() => expect(screen.getByTitle(/Go to quote 1, on page 1/)).toBeInTheDocument());
    expect(screen.getByTitle(/Go to quote 2, on page 3/)).toBeInTheDocument();
    unmount();

    render(<PdfViewer paperId="p1" quotes={["The trial enrolled 240 adults"]} />);
    await waitFor(() => expect(marks().length).toBeGreaterThan(0));
    expect(screen.queryByTitle(/Go to quote/)).toBeNull();
  });

  it("scales the marks with the zoom, so they stay on their words", async () => {
    render(<PdfViewer paperId="p1" quotes={["The trial enrolled 240 adults"]} />);
    await waitFor(() => expect(marks().length).toBeGreaterThan(0));
    const before = parseFloat(marks()[0].style.left);
    fireEvent.click(screen.getByTitle("Zoom in"));
    await waitFor(() => expect(screen.getByText("120%")).toBeInTheDocument());
    expect(parseFloat(marks()[0].style.left)).toBeCloseTo(before * 1.2, 1);
  });

  it("zooms the sheet itself as well", async () => {
    render(<PdfViewer paperId="p1" />);
    await waitFor(() => expect(sheets()).toHaveLength(3));
    fireEvent.click(screen.getByTitle("Zoom in"));
    await waitFor(() => expect(sheets()[0].style.width).toBe("734px"));
  });

  it("settles on the last scale asked for when zoom lands mid-render", async () => {
    // Claiming the canvas used to happen after an await, so a second render
    // could start with nothing to cancel. pdf.js refuses to draw two things
    // into one canvas, and the page came out at the wrong scale underneath
    // correctly-placed highlights.
    RENDER_MS = 25;
    render(<PdfViewer paperId="p1" />);
    await waitFor(() => expect(sheets()).toHaveLength(3));

    fireEvent.click(screen.getByTitle("Zoom in"));
    fireEvent.click(screen.getByTitle("Zoom in"));
    await waitFor(() => expect(screen.getByText("140%")).toBeInTheDocument());

    const canvas = () => sheets()[0].querySelector("canvas") as HTMLCanvasElement;
    await waitFor(() => expect(canvas().dataset.scale).toBe("1.4"));
    // The bitmap, the sheet and the scale all agree.
    expect(canvas().style.width).toBe(`${Math.floor(612 * 1.4)}px`);
    expect(sheets()[0].style.width).toBe(`${Math.floor(612 * 1.4)}px`);
  });

  it("will not page below the first page", async () => {
    render(<PdfViewer paperId="p1" />);
    await waitFor(() => expect(screen.getByText("1 / 3")).toBeInTheDocument());
    expect(screen.getByTitle("Previous page")).toBeDisabled();
    expect(screen.getByTitle("Next page")).toBeEnabled();
  });

  it("falls back to a link when there is no file", async () => {
    getPdfFile.mockResolvedValue(undefined as any);
    render(<PdfViewer paperId="missing" externalUrl="https://example.org/a.pdf" />);
    await waitFor(() =>
      expect(screen.getByText(/No document file for this study/i)).toBeInTheDocument());
    expect(screen.getByRole("link", { name: /Open it in a new tab/i }))
      .toHaveAttribute("href", "https://example.org/a.pdf");
  });

  it("falls back when the document will not parse", async () => {
    getDocument.mockReturnValueOnce({ promise: Promise.reject(new Error("corrupt")) } as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(<PdfViewer paperId="p1" />);
    await waitFor(() =>
      expect(screen.getByText(/could not be displayed here/i)).toBeInTheDocument());
  });

  it("fetches a url source, dropping the viewer's page anchor", async () => {
    const fetchMock = vi.fn(async () => new Response(new Blob(["%PDF-1.7"])));
    vi.stubGlobal("fetch", fetchMock);
    render(<PdfViewer url="/api/fulltext/pdf/abc#page=4" />);
    await waitFor(() => expect(sheets()).toHaveLength(3));
    expect(fetchMock).toHaveBeenCalledWith("/api/fulltext/pdf/abc");
  });

  it("does not ask pdf.js for anything when given neither a file nor a url", async () => {
    render(<PdfViewer />);
    await waitFor(() =>
      expect(screen.getByText(/No document file for this study/i)).toBeInTheDocument());
    expect(getDocument).not.toHaveBeenCalled();
  });
});
