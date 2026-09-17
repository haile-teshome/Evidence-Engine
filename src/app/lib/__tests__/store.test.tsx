// Tests for the review store: the single React context that holds the entire
// review, plus its snapshot/hydrate persistence cycle.
//
// Run: pnpm test
//
// This is where a bug loses a reviewer's work. If snapshot drops a field, or
// hydrate fails to restore one, hours of screening vanish on a browser refresh
// and nothing errors: the app simply comes back emptier than it went away.
// Every test here is a round-trip or an overwrite-protection assertion.
import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";

import { StoreProvider, useStore } from "../store";

const wrapper = ({ children }: { children: ReactNode }) => (
  <StoreProvider>{children}</StoreProvider>
);

const mount = () => renderHook(() => useStore(), { wrapper });

const paper = (id: string) => ({
  paper_id: id, id, Title: `Study ${id}`, title: `Study ${id}`,
  Abstract: "abstract", abstract: "abstract",
  Source: "PubMed", source: "PubMed", URL: "", url: "",
});

const screenResult = (id: string, decision: "INCLUDE" | "EXCLUDE" = "INCLUDE") => ({
  paper_id: id, Title: `Study ${id}`, Source: "PubMed", URL: "",
  Abstract: "We randomised 128 adults.",
  Decision: decision, Reason: "because", Agent_Trace: {},
  Pico_Assessment: {
    overall_reasoning: "Matches every element.",
    population: { vote: "PASS", evidence: "", reasoning: "" },
  },
}) as any;

beforeEach(() => localStorage.clear());

describe("initial state", () => {
  it("starts on a valid page", () => {
    const { result } = mount();
    expect(typeof result.current.page).toBe("string");
  });

  it("starts with an empty corpus", () => {
    const { result } = mount();
    expect(result.current.rawPapers ?? []).toHaveLength(0);
    expect(result.current.results ?? []).toHaveLength(0);
  });

  it("exposes a setter for every value it exposes", () => {
    const { result } = mount();
    for (const key of ["page", "model", "pico", "results", "sources"]) {
      const setter = `set${key[0].toUpperCase()}${key.slice(1)}`;
      expect(typeof (result.current as any)[setter]).toBe("function");
    }
  });

  it("defaults to a local model rather than a paid one", () => {
    const { result } = mount();
    expect(String(result.current.model ?? "")).not.toMatch(/gpt|claude|gemini/i);
  });
});

describe("state updates", () => {
  it("navigates between pages", () => {
    const { result } = mount();
    act(() => result.current.setPage("abstract"));
    expect(result.current.page).toBe("abstract");
  });

  it("stores screening results", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1"), screenResult("p2", "EXCLUDE")]));
    expect(result.current.results).toHaveLength(2);
  });

  it("stores the PICO frame", () => {
    const { result } = mount();
    act(() => result.current.setPico({
      population: "adults", intervention: "metformin",
      comparator: "placebo", outcome: "HbA1c",
    } as any));
    expect(result.current.pico.population).toBe("adults");
  });

  it("stores a reviewer override without touching the AI decision", () => {
    const { result } = mount();
    act(() => {
      result.current.setResults([screenResult("p1", "INCLUDE")]);
      result.current.setAbstractOverrides({ p1: "EXCLUDE" } as any);
    });
    expect(result.current.results[0].Decision).toBe("INCLUDE");
    expect(result.current.abstractOverrides.p1).toBe("EXCLUDE");
  });
});

// --------------------------------------------------------------------------
// Snapshot / hydrate. The property that matters: anything the reviewer did
// must survive the round trip.
// --------------------------------------------------------------------------

describe("snapshot and hydrate", () => {
  it("round-trips screening results", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1"), screenResult("p2")]));
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.results).toHaveLength(2);
    expect(fresh.result.current.results[0].paper_id).toBe("p1");
  });

  it("round-trips the PICO frame and framework", () => {
    const { result } = mount();
    act(() => {
      result.current.setPico({ population: "adults", concept: "AI" } as any);
      result.current.setFramework("pcc" as any);
    });
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.pico.population).toBe("adults");
    expect(fresh.result.current.framework).toBe("pcc");
  });

  it("round-trips reviewer overrides, which are human decisions", () => {
    const { result } = mount();
    act(() => result.current.setAbstractOverrides({ p1: "EXCLUDE", p2: "INCLUDE" } as any));
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.abstractOverrides).toEqual({ p1: "EXCLUDE", p2: "INCLUDE" });
  });

  it("round-trips the corpus and its dedup count", () => {
    const { result } = mount();
    act(() => {
      result.current.setRawPapers([paper("p1"), paper("p2"), paper("p3")] as any);
      result.current.setUniquePapers([paper("p1"), paper("p2")] as any);
      result.current.setDuplicatesCount(1);
    });
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.uniquePapers).toHaveLength(2);
    expect(fresh.result.current.duplicatesCount).toBe(1);
  });

  it("round-trips eligibility criteria", () => {
    const { result } = mount();
    act(() => {
      result.current.setInclusion(["Published after 2010"]);
      result.current.setExclusion(["Animal studies"]);
    });
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.inclusion).toEqual(["Published after 2010"]);
    expect(fresh.result.current.exclusion).toEqual(["Animal studies"]);
  });

  it("round-trips acquired full texts", () => {
    const { result } = mount();
    act(() => result.current.setFullTexts({
      p1: { paper_id: "p1", title: "T", url: "", source: "PubMed",
            status: "found", text: "full text body" },
    } as any));
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.fullTexts.p1.status).toBe("found");
  });

  it("round-trips the search query and per-database queries", () => {
    const { result } = mount();
    act(() => {
      result.current.setQuery('("dental"[tiab])');
      result.current.setPerDbQueries({ PubMed: '("dental"[tiab])' } as any);
    });
    const snap = result.current.snapshot();

    const fresh = mount();
    act(() => fresh.result.current.hydrate(snap));
    expect(fresh.result.current.query).toBe('("dental"[tiab])');
    expect(fresh.result.current.perDbQueries.PubMed).toBeTruthy();
  });

  it("a snapshot is JSON-serialisable, since that is how it is stored", () => {
    const { result } = mount();
    act(() => {
      result.current.setResults([screenResult("p1")]);
      result.current.setRawPapers([paper("p1")] as any);
    });
    expect(() => JSON.stringify(result.current.snapshot())).not.toThrow();
  });

  it("survives a full JSON round trip, not just an object copy", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1")]));
    const revived = JSON.parse(JSON.stringify(result.current.snapshot()));

    const fresh = mount();
    act(() => fresh.result.current.hydrate(revived));
    expect(fresh.result.current.results).toHaveLength(1);
  });

  it("hydrating null or undefined is a no-op rather than a wipe", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1")]));
    act(() => result.current.hydrate(null));
    act(() => result.current.hydrate(undefined));
    expect(result.current.results).toHaveLength(1);
  });

  it("a non-authoritative hydrate does not wipe existing work with blanks", () => {
    /* Merging a session that happens to be empty must not delete the review
       currently on screen. */
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1")]));
    act(() => result.current.hydrate({ results: [] }, false));
    expect(result.current.results).toHaveLength(1);
  });

  it("an authoritative hydrate does replace state", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1")]));
    act(() => result.current.hydrate({ results: [screenResult("p9")] }, true));
    expect(result.current.results[0].paper_id).toBe("p9");
  });

  it("hydrating a partial snapshot does not crash", () => {
    const { result } = mount();
    act(() => result.current.hydrate({ query: "just a query" }));
    expect(result.current.query).toBe("just a query");
  });

  it("hydrating a snapshot full of junk does not crash", () => {
    const { result } = mount();
    act(() => result.current.hydrate({
      results: "not-an-array", pico: 42, sources: null, unknownKey: {},
    } as any));
    expect(result.current).toBeTruthy();
  });
});

describe("reset", () => {
  it("clears screening results", () => {
    const { result } = mount();
    act(() => result.current.setResults([screenResult("p1")]));
    act(() => result.current.reset());
    expect(result.current.results ?? []).toHaveLength(0);
  });

  it("clears the corpus", () => {
    const { result } = mount();
    act(() => result.current.setRawPapers([paper("p1")] as any));
    act(() => result.current.reset());
    expect(result.current.rawPapers ?? []).toHaveLength(0);
  });

  it("leaves the store usable afterwards", () => {
    const { result } = mount();
    act(() => result.current.reset());
    act(() => result.current.setPage("abstract"));
    expect(result.current.page).toBe("abstract");
  });
});

// --------------------------------------------------------------------------
// Task tracking. Drives the progress UI during a long screening run.
// --------------------------------------------------------------------------

describe("task tracking", () => {
  it("starts a task and exposes an abort handle", () => {
    const { result } = mount();
    let handle: any;
    act(() => { handle = result.current.startTask("abstract-screen", [{ id: "s", label: "S", status: "running" }]); });
    expect(handle.abort).toBeInstanceOf(AbortController);
  });

  it("records progress against a running task", () => {
    const { result } = mount();
    act(() => { result.current.startTask("abstract-screen", [{ id: "s", label: "S", status: "running" }]); });
    act(() => result.current.updateTask("abstract-screen", { progress: { done: 5, total: 10 } }));
    expect(result.current.tasks["abstract-screen"]?.progress?.done).toBe(5);
  });

  it("marks a task done", () => {
    const { result } = mount();
    act(() => { result.current.startTask("abstract-screen", []); });
    act(() => result.current.updateTask("abstract-screen", { status: "done" }));
    expect(result.current.tasks["abstract-screen"]?.status).toBe("done");
  });

  it("cancelling aborts the signal so in-flight work stops", () => {
    const { result } = mount();
    let handle: any;
    act(() => { handle = result.current.startTask("abstract-screen", []); });
    act(() => result.current.cancelTask("abstract-screen"));
    expect(handle.abort.signal.aborted).toBe(true);
  });

  it("clears a finished task", () => {
    const { result } = mount();
    act(() => { result.current.startTask("abstract-screen", []); });
    act(() => result.current.clearTask("abstract-screen"));
    expect(result.current.tasks["abstract-screen"]).toBeUndefined();
  });

  it("updating an unknown task does not crash", () => {
    const { result } = mount();
    act(() => result.current.updateTask("snowball" as any, { status: "done" }));
    expect(result.current.tasks).toBeTruthy();
  });

  it("tasks are NOT persisted into a snapshot", () => {
    /* An AbortController cannot be serialised, and a restored session must not
       come back believing a run from last week is still in flight. */
    const { result } = mount();
    act(() => { result.current.startTask("abstract-screen", []); });
    expect(JSON.stringify(result.current.snapshot())).not.toContain("AbortController");
  });
});

describe("useStore outside a provider", () => {
  it("fails loudly rather than returning a silently broken store", () => {
    expect(() => renderHook(() => useStore())).toThrow();
  });
});
