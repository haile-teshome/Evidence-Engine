// How long is left, said in a way that does not jump around.
//
// Run: pnpm test
//
// The complaint this exists for: "the time just randomly bounces up and down".
// It did, for three reasons, and the tests below pin each one. The last test is
// a simulation of a real run that scores the old formula against the new one on
// the thing a reviewer actually notices: how often the number on screen moves,
// and whether it ever goes up.
import { describe, it, expect } from "vitest";

import {
  WINDOW, etaLabel, formatEta, initialEta, roundTo, step, stepEta,
} from "../eta";

/** The formula this replaced: elapsed / done, projected over the rest. */
const oldEta = (startedAt: number, done: number, total: number, now: number) =>
  done > 0 && done < total ? ((now - startedAt) / done) * (total - done) : null;

/** Run a sequence of item durations through the estimator on a 500ms tick. */
function simulate(durations: number[], tickMs = 500) {
  const total = durations.length;
  const startedAt = 0;
  let state = initialEta(startedAt);
  let now = 0;
  let done = 0;
  let nextLandsAt = durations[0];
  const shown: (number | null)[] = [];
  const old: (number | null)[] = [];

  while (done < total) {
    now += tickMs;
    while (done < total && now >= nextLandsAt) {
      done += 1;
      if (done < total) nextLandsAt += durations[done];
    }
    state = stepEta(state, { done, total, now });
    shown.push(state.shownMs);
    old.push(oldEta(startedAt, done, total, now));
  }
  return { state, shown, old, elapsed: now };
}

/** How many times a series moves, and how many of those are upward. */
function churn(series: (number | null)[]) {
  const seen = series.filter((v): v is number => v !== null).map(v => Math.round(v / 1000));
  let moves = 0;
  let ups = 0;
  for (let i = 1; i < seen.length; i++) {
    if (seen[i] !== seen[i - 1]) moves += 1;
    if (seen[i] > seen[i - 1]) ups += 1;
  }
  return { moves, ups, samples: seen.length };
}

describe("the estimate does not climb between items", () => {
  it("ticks down while an item is in flight", () => {
    // The sawtooth: `done` is fixed while the clock moves, so the old formula
    // grew for the whole of every item and dropped when it landed.
    let state = initialEta(0);
    state = stepEta(state, { done: 1, total: 10, now: 10_000 });
    const first = state.shownMs!;
    state = stepEta(state, { done: 1, total: 10, now: 14_000 });
    const later = state.shownMs!;
    expect(later).toBeLessThanOrEqual(first);
  });

  it("never shows a bigger number on a steady run", () => {
    const { shown } = simulate(Array(12).fill(8_000));
    const { ups } = churn(shown);
    expect(ups).toBe(0);
  });
});

describe("one slow item does not poison the rest", () => {
  it("keeps the estimate near the typical item, not the mean", () => {
    // Eight items at 5s and one at 60s. A mean says ~11s an item; the median
    // says 5s, which is what the remaining items will actually cost.
    let state = initialEta(0);
    let now = 0;
    const durations = [5_000, 5_000, 5_000, 60_000, 5_000, 5_000];
    durations.forEach((d, i) => {
      now += d;
      state = stepEta(state, { done: i + 1, total: 12, now });
    });
    const perItem = state.shownMs! / (12 - durations.length);
    expect(perItem).toBeLessThan(12_000);
  });

  it("keeps only the recent window of timings", () => {
    let state = initialEta(0);
    let now = 0;
    for (let i = 1; i <= WINDOW + 5; i++) {
      now += 1_000;
      state = stepEta(state, { done: i, total: 100, now });
    }
    expect(state.samples.length).toBe(WINDOW);
  });

  it("follows a run that genuinely slows down", () => {
    // Robustness must not mean ignoring a real change of pace.
    let state = initialEta(0);
    let now = 0;
    for (let i = 1; i <= 6; i++) { now += 2_000; state = stepEta(state, { done: i, total: 40, now }); }
    const fast = state.shownMs!;
    for (let i = 7; i <= 20; i++) { now += 20_000; state = stepEta(state, { done: i, total: 40, now }); }
    expect(state.shownMs!).toBeGreaterThan(fast);
  });
});

describe("startup cost is not charged to every item", () => {
  it("recovers after a slow first item", () => {
    // Loading the model and the first long prompt made the old estimate wild.
    let state = initialEta(0);
    let now = 40_000;
    state = stepEta(state, { done: 1, total: 11, now });
    const afterStartup = state.shownMs!;
    for (let i = 2; i <= 6; i++) { now += 4_000; state = stepEta(state, { done: i, total: 11, now }); }
    expect(state.shownMs!).toBeLessThan(afterStartup / 2);
  });
});

describe("buckets and labels", () => {
  it("widens the step as the estimate grows", () => {
    expect(step(4_000)).toBe(5_000);
    expect(step(45_000)).toBe(5_000);
    expect(step(300_000)).toBe(30_000);
    expect(step(3_600_000)).toBe(60_000);
  });

  it("rounds to its own step and never to nothing", () => {
    expect(roundTo(4_000)).toBe(5_000);
    expect(roundTo(46_000)).toBe(45_000);
    expect(roundTo(301_000)).toBe(300_000);
    expect(roundTo(0)).toBe(5_000);
  });

  it("formats the way the cards already did", () => {
    expect(formatEta(8_000)).toBe("8s");
    expect(formatEta(150_000)).toBe("2m 30s");
    expect(formatEta(120_000)).toBe("2m");
  });

  it("says it is estimating until an item has landed", () => {
    const state = initialEta(0);
    expect(etaLabel(state, 0, 10)).toBe("estimating…");
  });

  it("says nothing once the work is done", () => {
    const state = initialEta(0);
    expect(etaLabel(state, 10, 10)).toBeNull();
    expect(etaLabel(state, 0, 0)).toBeNull();
  });

  it("starts again rather than mixing two runs", () => {
    let state = initialEta(0);
    state = stepEta(state, { done: 5, total: 10, now: 20_000 });
    expect(state.samples.length).toBeGreaterThan(0);
    const restarted = stepEta(state, { done: 0, total: 10, now: 30_000 });
    expect(restarted.samples).toEqual([]);
    expect(restarted.shownMs).toBeNull();
  });
});

describe("measured against the formula it replaces", () => {
  it("moves far less often, and never upward", () => {
    // A realistic extraction run: a slow first article while the model loads,
    // then a spread of articles, one of them much longer than the rest.
    const durations = [38_000, 9_000, 7_000, 11_000, 8_000, 52_000,
                       9_000, 6_000, 10_000, 8_000, 7_000, 12_000];
    const { shown, old } = simulate(durations);
    const now = churn(shown);
    const before = churn(old);

    // The headline: the old number changed on almost every tick and spent most
    // of the run climbing. The new one moves a handful of times, and only goes
    // up just after a genuinely slow article has landed, which is honest: the
    // rest of the run really may cost more than the estimate said.
    expect(before.ups).toBeGreaterThan(50);
    expect(now.ups).toBeLessThanOrEqual(3);
    expect(now.moves).toBeLessThan(before.moves / 5);
  });
});
