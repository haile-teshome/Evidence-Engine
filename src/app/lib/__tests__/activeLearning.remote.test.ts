// Tests for the active-learning ranker's tier contract.
//
// Run: pnpm test
//
// The manuscript reports WSS@95 0.679 for a BGE-large + TF-IDF ensemble and
// describes it as what the platform runs. For most of this project's life it
// wasn't: the app shipped an in-browser Naive Bayes that had never been
// benchmarked, and nothing in the UI or the code said so. The ranker now calls
// the backend, but a fallback path still exists, so the property that stops the
// same divergence recurring is narrow and specific:
//
//     whichever ranker produced the ordering, `tier` names it, and the
//     unbenchmarked one is never labelled as the benchmarked one.
//
// Everything below exists to pin that. The recall estimate drives a "safe to
// stop" prompt, so a reviewer trusting an unvalidated model's estimate while
// believing it is the validated one is how a systematic review silently loses
// included studies.
import { describe, it, expect, beforeEach, vi } from "vitest";

import { rankRecords, activeRank, type ALItem } from "../activeLearning";

type FakeRes = { ok?: boolean; status?: number; body?: string };

function mockFetch(handler: (url: string, init: any) => FakeRes | Promise<FakeRes>) {
  const calls: { url: string; init: any }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: any, init: any = {}) => {
    calls.push({ url: String(url), init });
    const res = await handler(String(url), init);
    return {
      ok: res.ok ?? true,
      status: res.status ?? 200,
      text: async () => res.body ?? "{}",
      json: async () => JSON.parse(res.body ?? "{}"),
      headers: new Headers(),
    } as unknown as Response;
  }));
  return calls;
}

const items: ALItem[] = [
  { id: "a", title: "Metformin in type 2 diabetes", text: "randomised trial", aiInclude: true, override: "include" },
  { id: "b", title: "Bridge fatigue in steel girders", text: "engineering", aiInclude: false, override: "exclude" },
  { id: "c", title: "Metformin and HbA1c outcomes", text: "cohort", aiInclude: true },
  { id: "d", title: "Concrete creep modelling", text: "structural", aiInclude: false },
];

const served = (over: Partial<any> = {}) => JSON.stringify({
  order: ["c", "d"],
  scores: { c: 1.7, d: 0.2 },
  tier: "bge+tfidf",
  trained: true,
  reviewed: 2,
  includes_found: 1,
  predicted_remaining: 1,
  est_recall: 0.5,
  batch: 1,
  detail: "benchmarked configuration on mps",
  ...over,
});

beforeEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

// --------------------------------------------------------------------------
// The benchmarked path
// --------------------------------------------------------------------------

describe("the benchmarked ranker", () => {
  it("reports the bge+tfidf tier when the backend serves it", async () => {
    mockFetch(() => ({ body: served() }));
    expect((await rankRecords(items)).tier).toBe("bge+tfidf");
  });

  it("uses the backend's ordering rather than re-ranking locally", async () => {
    mockFetch(() => ({ body: served({ order: ["d", "c"] }) }));
    expect((await rankRecords(items)).order).toEqual(["d", "c"]);
  });

  it("carries the backend's recall estimate through unchanged", async () => {
    /* This number drives the "safe to stop" prompt, so it must not be
       recomputed from a different model's scores on the way to the UI. */
    mockFetch(() => ({ body: served({ est_recall: 0.97, predicted_remaining: 0 }) }));
    const r = await rankRecords(items);
    expect(r.estRecall).toBe(0.97);
    expect(r.predictedRemaining).toBe(0);
  });

  it("posts to the rank endpoint", async () => {
    const calls = mockFetch(() => ({ body: served() }));
    await rankRecords(items);
    expect(calls[0].url).toContain("/rank");
    expect(calls[0].init.method).toBe("POST");
  });
});

// --------------------------------------------------------------------------
// Labels. The backend can only be right if it is told the truth.
// --------------------------------------------------------------------------

describe("label extraction", () => {
  const body = (calls: { init: any }[]) => JSON.parse(calls[0].init.body);

  it("sends include as 1 and exclude as 0", async () => {
    const calls = mockFetch(() => ({ body: served() }));
    await rankRecords(items);
    expect(body(calls).labels).toEqual({ a: 1, b: 0 });
  });

  it("does not invent labels for unreviewed records", async () => {
    const calls = mockFetch(() => ({ body: served() }));
    await rankRecords(items);
    expect(body(calls).labels).not.toHaveProperty("c");
    expect(body(calls).labels).not.toHaveProperty("d");
  });

  it("sends every record, not only the unlabelled pool", async () => {
    /* TF-IDF is fit over the whole corpus, so a truncated record list would
       silently change the feature space the benchmark was measured on. */
    const calls = mockFetch(() => ({ body: served() }));
    await rankRecords(items);
    expect(body(calls).records).toHaveLength(4);
  });

  it("sends an empty label map when nothing is reviewed yet", async () => {
    const calls = mockFetch(() => ({ body: served({ tier: "cold", trained: false }) }));
    await rankRecords(items.map(it => ({ ...it, override: undefined })));
    expect(body(calls).labels).toEqual({});
  });
});

// --------------------------------------------------------------------------
// THE REGRESSION GUARD. A fallback is allowed; an unlabelled fallback is not.
// --------------------------------------------------------------------------

describe("fallbacks are always named", () => {
  it("falls back to the in-browser model when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("connection refused"))));
    const r = await rankRecords(items);
    expect(r.order.length).toBeGreaterThan(0);
  });

  it("reports tier 'nb', NOT 'bge+tfidf', when it fell back", async () => {
    /* The single assertion this whole file exists for. */
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("connection refused"))));
    expect((await rankRecords(items)).tier).toBe("nb");
  });

  it("says in words that the fallback is not the benchmarked model", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    expect((await rankRecords(items)).detail).toMatch(/unbenchmarked|not.*benchmark/i);
  });

  it("reports tier 'nb' when the backend returns a 500", async () => {
    mockFetch(() => ({ ok: false, status: 500, body: '{"detail":"boom"}' }));
    expect((await rankRecords(items)).tier).toBe("nb");
  });

  it("reports the reduced tier verbatim when the backend has no embeddings", async () => {
    /* sklearn present, torch absent. Still a real ranker, still not the one
       behind the published number, so it keeps its own name. */
    mockFetch(() => ({ body: served({ tier: "tfidf", detail: "lexical only" }) }));
    expect((await rankRecords(items)).tier).toBe("tfidf");
  });

  it("never returns bge+tfidf without the backend having said so", async () => {
    for (const fail of [
      () => Promise.reject(new Error("offline")),
      async () => ({ ok: false, status: 503, text: async () => "", json: async () => ({}), headers: new Headers() }),
    ]) {
      vi.stubGlobal("fetch", vi.fn(fail as any));
      expect((await rankRecords(items)).tier).not.toBe("bge+tfidf");
    }
  });
});

// --------------------------------------------------------------------------
// Cold start
// --------------------------------------------------------------------------

describe("cold start", () => {
  it("falls back to the local AI-decision ordering when too few labels exist", async () => {
    mockFetch(() => ({ body: served({ tier: "cold", trained: false, order: [], scores: {} }) }));
    const r = await rankRecords(items);
    expect(r.tier).toBe("cold");
    expect(r.order).toEqual(activeRank(items).order);
  });

  it("puts AI-included records first while cold", async () => {
    mockFetch(() => ({ body: served({ tier: "cold", trained: false, order: [] }) }));
    const r = await rankRecords(items.map(it => ({ ...it, override: undefined })));
    expect(r.order[0]).toBe("a");
  });

  it("is not marked trained while cold", async () => {
    mockFetch(() => ({ body: served({ tier: "cold", trained: false }) }));
    expect((await rankRecords(items)).trained).toBe(false);
  });

  it("treats a trained:false response as cold even if it names a tier", async () => {
    /* Otherwise a backend that answers before it has fit anything would be
       reported as the validated ranker on the strength of its tier string. */
    mockFetch(() => ({ body: served({ tier: "bge+tfidf", trained: false }) }));
    expect((await rankRecords(items)).tier).toBe("cold");
  });
});

// --------------------------------------------------------------------------
// Aborts are cancellation, not failure
// --------------------------------------------------------------------------

describe("abort handling", () => {
  it("propagates an abort instead of reporting a fallback ranking", async () => {
    /* Re-ranking is cancelled whenever the reviewer labels faster than the
       request returns. Swallowing that as a fallback would flip the UI to
       "offline ranker" during entirely normal use. */
    const err = new Error("aborted");
    err.name = "AbortError";
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(err)));
    await expect(rankRecords(items, new AbortController().signal)).rejects.toThrow(/abort/i);
  });
});

// --------------------------------------------------------------------------
// Edge cases
// --------------------------------------------------------------------------

describe("edge cases", () => {
  it("handles an empty record list without throwing", async () => {
    mockFetch(() => ({ ok: false, status: 400, body: '{"detail":"no records to rank"}' }));
    expect((await rankRecords([])).order).toEqual([]);
  });

  it("handles records with no text at all", async () => {
    mockFetch(() => ({ body: served({ order: ["z"], scores: { z: 0.5 } }) }));
    const r = await rankRecords([{ id: "z", aiInclude: false }]);
    expect(r.order).toEqual(["z"]);
  });

  it("survives a malformed backend body by falling back and saying so", async () => {
    mockFetch(() => ({ body: "not json at all" }));
    const r = await rankRecords(items);
    expect(["nb", "cold"]).toContain(r.tier);
  });
});
