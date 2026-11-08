// How long is left, said in a way that does not jump around.
//
// The first version divided the whole elapsed time by the number of items done
// and projected that over the rest, recomputed every 500 ms. That bounces for
// three separate reasons, and all three are visible on a long run:
//
//   1. Between two items finishing, `done` is fixed while the clock moves, so
//      the estimate CLIMBS, then drops the moment an item lands. A sawtooth.
//   2. Averaging from the start folds in the cost of starting up: loading the
//      model and the first long prompt. The early numbers are far too high and
//      sink for the rest of the run.
//   3. One slow article drags the average for every item after it, because a
//      mean has no way to treat it as the outlier it is.
//
// So: time each item, take the MEDIAN of the recent ones (an outlier moves a
// median by one position, not by its size), count the time already spent on the
// item in flight against the estimate so the number only falls, and round the
// result into buckets so the text stops changing twice a second.

import { useEffect, useRef, useState } from "react";

/** Per-item timings and what is currently displayed. */
export type EtaState = {
  /** Item count at the last change, so a jump of several is handled. */
  lastDone: number;
  /** When `lastDone` was reached. */
  lastChangeAt: number;
  /** Durations of recent items, newest last, in ms. */
  samples: number[];
  /** The bucketed estimate being shown, in ms, or null while unknown. */
  shownMs: number | null;
};

/** How many item timings to keep. Enough for a median to settle, few enough
 *  that it follows a run which genuinely speeds up or slows down. */
export const WINDOW = 7;

export function initialEta(now: number, done = 0): EtaState {
  return { lastDone: done, lastChangeAt: now, samples: [], shownMs: null };
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * How coarsely to say an estimate of this size.
 *
 * The steps widen with the estimate because the precision a reviewer can use
 * widens too: five seconds is worth distinguishing from ten, fourteen minutes
 * is not worth distinguishing from fourteen minutes and ten seconds.
 */
export function step(ms: number): number {
  const s = Math.abs(ms) / 1000;
  if (s < 60) return 5_000;
  if (s < 600) return 30_000;
  return 60_000;
}

/** The estimate rounded to its own step, never below one step. */
export function roundTo(ms: number): number {
  const g = step(ms);
  return Math.max(g, Math.round(ms / g) * g);
}

/**
 * Fold one observation into the estimate.
 *
 * Call it on every tick; it only re-bases when an item completes. Between
 * completions the time already spent on the item in flight is subtracted, so
 * the number ticks down instead of climbing.
 */
export function stepEta(
  state: EtaState,
  { done, total, now }: { done: number; total: number; now: number },
): EtaState {
  if (total <= 0 || done > total) return state;

  let { lastDone, lastChangeAt, samples } = state;

  if (done > lastDone) {
    // One or more items landed. Share the interval between them: a batch that
    // reports in one go did not take all that time on its last member.
    const per = (now - lastChangeAt) / (done - lastDone);
    const next = [...samples, ...Array(Math.min(done - lastDone, WINDOW)).fill(per)];
    samples = next.slice(-WINDOW);
    lastDone = done;
    lastChangeAt = now;
  } else if (done < lastDone) {
    // A new run reusing the same card. Start again rather than mixing rates.
    return initialEta(now, done);
  }

  if (!samples.length || done >= total) {
    return { lastDone, lastChangeAt, samples, shownMs: state.shownMs };
  }

  const rate = median(samples);
  const left = total - done;
  // What is left, minus the part of the current item already behind us. Never
  // below one bucket: claiming zero while work is still running reads as a
  // stall, and the estimate has simply run out.
  const raw = Math.max(rate * left - (now - lastChangeAt), 0);
  const rounded = roundTo(raw);

  // Hysteresis: the number on screen stays put until the new one is a whole
  // step away, in either direction. Without this it flickers between two
  // neighbouring values for the whole of every item.
  const shown = state.shownMs;
  const moved = shown === null
    || Math.abs(rounded - shown) >= step(Math.max(shown, rounded));
  return { lastDone, lastChangeAt, samples, shownMs: moved ? rounded : shown };
}

/**
 * The label for a running job, recomputed on a timer.
 *
 * Each caller keeps its own estimate, so two jobs on one page do not pollute
 * each other's timings. A new run resets rather than inheriting the last one's
 * pace, which would otherwise make the first estimate of a fast run look slow.
 */
export function useEta(done: number, total: number, running: boolean): string | null {
  const [state, setState] = useState<EtaState>(() => initialEta(Date.now(), done));
  const doneNow = useRef(done);
  doneNow.current = done;

  useEffect(() => {
    if (running) setState(initialEta(Date.now(), doneNow.current));
  }, [running]);

  useEffect(() => {
    if (!running || total <= 0) return;
    const tick = () => setState(s => stepEta(s, { done, total, now: Date.now() }));
    tick();                                   // record the item that just landed
    const timer = setInterval(tick, 500);
    return () => clearInterval(timer);
  }, [done, total, running]);

  return running ? etaLabel(state, done, total) : null;
}

/** "8s", "2m 30s" — the shape the progress cards already used. */
export function formatEta(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rest = s % 60;
  return rest ? `${m}m ${rest}s` : `${m}m`;
}

/**
 * The label to show beside a progress bar, or null when there is nothing
 * honest to say yet.
 */
export function etaLabel(state: EtaState, done: number, total: number): string | null {
  if (total <= 0 || done >= total) return null;
  if (state.shownMs === null) return "estimating…";
  return `~${formatEta(state.shownMs)} left`;
}
