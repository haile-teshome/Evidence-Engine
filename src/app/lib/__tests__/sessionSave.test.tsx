// What counts as work worth saving.
//
// Run: pnpm test
//
// The autosave used to ask whether Home had run. That was the same question as
// "is there anything to save" only while every review started on Home, and it
// stopped being true once studies could be imported at any stage and a search
// could be planned without a research question. The failure was silent and
// destructive in both directions: nothing was written, and the empty-history
// branch deleted whatever snapshot was already there.
import { readFileSync } from "node:fs";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";

const idbGet = vi.fn(async () => undefined);
const idbSet = vi.fn(async () => undefined);
const idbDel = vi.fn(async () => undefined);
vi.mock("../idb", () => ({
  idbGet: (...a: any[]) => (idbGet as any)(...a),
  idbSet: (...a: any[]) => (idbSet as any)(...a),
  idbDel: (...a: any[]) => (idbDel as any)(...a),
}));

import { StoreProvider, useStore, sessionHasWork } from "../store";

const wrapper = ({ children }: { children: ReactNode }) => (
  <StoreProvider>{children}</StoreProvider>
);
const mount = () => renderHook(() => useStore(), { wrapper });

const paper = (id: string) => ({
  paper_id: id, id, Title: `Study ${id}`, title: `Study ${id}`,
  Abstract: "abstract", abstract: "abstract",
  Source: "Local PDFs", source: "Local PDFs", URL: "", url: "",
}) as any;

/** Let the restore settle, then let the debounced save fire. */
async function settle() {
  await act(async () => { await Promise.resolve(); });
  await act(async () => { vi.advanceTimersByTime(2000); await Promise.resolve(); });
}

beforeEach(() => {
  localStorage.clear();
  idbGet.mockClear();
  idbSet.mockClear();
  idbDel.mockClear();
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe("the autosave gate", () => {
  it("saves a review that only ever imported PDFs", async () => {
    // The reviewer screened elsewhere and arrived with their own corpus. They
    // never touched Home, so history is empty and their work used to be
    // dropped on the floor.
    const { result } = mount();
    await settle();
    idbSet.mockClear();
    idbDel.mockClear();

    act(() => {
      result.current.setRawPapers([paper("p1")]);
      result.current.setUniquePapers([paper("p1")]);
    });
    await settle();

    expect(result.current.hasWork).toBe(true);
    expect(idbSet).toHaveBeenCalled();
    expect(idbDel).not.toHaveBeenCalled();
  });

  it("saves a search planned without a research question", async () => {
    const { result } = mount();
    await settle();
    idbSet.mockClear();

    act(() => result.current.setUnifiedSearchQuery("glioma AND temozolomide"));
    await settle();

    expect(result.current.hasWork).toBe(true);
    expect(idbSet).toHaveBeenCalled();
  });

  it("clears the snapshot when the review is reset to nothing", async () => {
    // The other half of the rule: starting a new session must not leave the old
    // one's snapshot behind to be restored on the next launch.
    const { result } = mount();
    await settle();
    act(() => result.current.setRawPapers([paper("p1")]));
    await settle();
    expect(idbSet).toHaveBeenCalled();
    idbSet.mockClear();
    idbDel.mockClear();

    act(() => result.current.reset());
    await settle();

    expect(result.current.hasWork).toBe(false);
    expect(idbDel).toHaveBeenCalled();
  });
});

describe("a late stage on its own", () => {
  it("saves when only the meta-analysis rows change", async () => {
    // The stage furthest from Home, and the one a reviewer can work in for an
    // hour without touching anything an earlier stage owns.
    const { result } = mount();
    await settle();
    idbSet.mockClear();

    act(() => result.current.setMetaExtractions([{ paper_id: "p1" } as any]));
    await settle();

    expect(idbSet).toHaveBeenCalled();
  });

  it("saves when only an acquired full text changes", async () => {
    const { result } = mount();
    await settle();
    idbSet.mockClear();

    act(() => result.current.setFullTexts({
      p1: { paper_id: "p1", title: "Study p1", url: "", source: "PubMed",
            status: "found", text: "body" } as any,
    }));
    await settle();

    expect(idbSet).toHaveBeenCalled();
  });
});

describe("what the saves watch", () => {
  // A source-level check, because the bug it guards against is a list that
  // drifts away from the snapshot it is supposed to track. Two hand-kept lists
  // had already drifted: the backend copy was missing thirty-five fields.
  // vitest runs from the repository root.
  const src = readFileSync("src/app/lib/store.tsx", "utf8");
  const snapshotBody = src.slice(src.indexOf("const snapshot = "),
                                src.indexOf("// `authoritative` = true"));
  const listStart = src.indexOf("const persisted: unknown[] = [");
  const persistedBody = src.slice(listStart, src.indexOf("];", listStart));
  const stateFields = [...src.matchAll(/const \[(\w+), set\w+\] = useState/g)].map(m => m[1]);

  it("has one list, and it covers every field the snapshot writes", () => {
    expect(listStart).toBeGreaterThan(0);
    const watched = new Set(persistedBody.match(/\b\w+\b/g) || []);
    const missing = stateFields.filter(
      f => new RegExp(`\\b${f}\\b`).test(snapshotBody) && !watched.has(f));
    expect(missing).toEqual([]);
  });

  it("bumps a version when a persisted field changes", async () => {
    const { result } = mount();
    const before = result.current.snapshotVersion;
    act(() => result.current.setResults([{ paper_id: "p1" } as any]));
    expect(result.current.snapshotVersion).toBeGreaterThan(before);
  });
});

describe("stage data that was never written down", () => {
  it("round-trips the seed papers chosen on Home", () => {
    const { result } = mount();
    act(() => result.current.setSeedIds(new Set(["p1", "p2"])));
    const snap = JSON.parse(JSON.stringify(result.current.snapshot()));

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect([...fresh.result.current.seedIds].sort()).toEqual(["p1", "p2"]);
  });

  it("round-trips the seed set the citations were fetched from", () => {
    // Without it the "your seeds have changed" warning cannot fire after a
    // reload, so stale citations quietly pass for current ones.
    const { result } = mount();
    act(() => result.current.setSnowballSeeds(["Study p1", "Study p2"]));
    const snap = JSON.parse(JSON.stringify(result.current.snapshot()));

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.snowballSeeds).toEqual(["Study p1", "Study p2"]);
  });
});

describe("sessionHasWork", () => {
  it("is false for nothing at all", () => {
    expect(sessionHasWork({})).toBe(false);
    expect(sessionHasWork({ history: [], rawPapers: [], fullTexts: {}, query: "   " })).toBe(false);
  });

  it.each([
    ["a research goal", { history: [{ goal: "g" }] }],
    ["imported papers", { rawPapers: [{ id: "p1" }] }],
    ["a deduplicated corpus", { uniquePapers: [{ id: "p1" }] }],
    ["abstract screening", { results: [{ paper_id: "p1" }] }],
    ["full-text screening", { fullTextResults: [{ paper_id: "p1" }] }],
    ["an acquired full text", { fullTexts: { p1: { status: "found" } } }],
    ["a risk-of-bias appraisal", { qualityReports: [{ paper_id: "p1" }] }],
    ["snowballed citations", { snowballResults: [{ id: "c1" }] }],
    ["extracted tables", { extractedPapers: [{ Paper_Title: "t" }] }],
    ["extracted text", { textExtractions: [{ paper_id: "p1" }] }],
    ["meta-analysis rows", { metaExtractions: [{ paper_id: "p1" }] }],
    ["a typed search string", { query: "glioma AND temozolomide" }],
    ["a tuned search string", { unifiedSearchQuery: "glioma" }],
  ])("is true for %s", (_what, state) => {
    expect(sessionHasWork(state as any)).toBe(true);
  });
});
