// Getting an imported file back out of the browser.
//
// Run: pnpm test
//
// The bytes are written to IndexedDB through a queue, and the import asks for
// them back in the same tick so the backend can read a scanned PDF. The read
// was racing the write it depends on, and lost: the file never reached the
// reader, and the reviewer was told their documents could not be read, which
// said nothing true about their documents.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const store = new Map<string, any>();

// A write only blocks when a test opens this gate before triggering it, so the
// window the read used to fall into can be held open deliberately.
let gate: Promise<void> | null = null;
let openGate: (() => void) | null = null;

const idbGet = vi.fn(async (k: string) => store.get(k));
const idbSet = vi.fn(async (k: string, v: any) => {
  if (gate) await gate;
  store.set(k, v);
});
const idbDel = vi.fn(async (k: string) => { store.delete(k); });

vi.mock("../idb", () => ({
  idbGet: (...a: any[]) => (idbGet as any)(...a),
  idbSet: (...a: any[]) => (idbSet as any)(...a),
  idbDel: (...a: any[]) => (idbDel as any)(...a),
}));

import { registerPdfBlobs, getPdfFile } from "../pdfBlobs";

beforeEach(() => {
  store.clear();
  gate = null;
  openGate = null;
  idbGet.mockClear();
  idbSet.mockClear();
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: () => "blob:test-url",
    revokeObjectURL: () => { /* no-op */ },
  });
});

afterEach(() => openGate?.());

describe("reading back a file that was just imported", () => {
  it("waits for the write it is racing", async () => {
    gate = new Promise<void>(res => { openGate = res; });
    const blob = new Blob(["%PDF-1.7"], { type: "application/pdf" });
    registerPdfBlobs([{ id: "p1", blob, mime: "application/pdf" }]);

    // Ask for it immediately, exactly as the import does, while the write that
    // stores it is still in flight.
    const pending = getPdfFile("p1");
    openGate!();

    await expect(pending).resolves.toBe(blob);
  });

  it("is undefined for a file nobody imported", async () => {
    await expect(getPdfFile("nothing")).resolves.toBeUndefined();
  });
});
