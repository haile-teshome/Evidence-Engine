// How long is left, said in a way that does not jump around.
//
// The first version divided the whole elapsed time by the number of items done
// and projected that over the rest, recomputed every 500 ms. That bounces for
// four separate reasons, and all four are visible on a long run:
//
//   1. Between two items finishing, `done` is fixed while the clock moves, so
//      the estimate CLIMBS, then drops the moment an item lands. A sawtooth.
//   2. Averaging from the start folds in the cost of starting up: loading the
//      model and the first long prompt. The early numbers are far too high and
//      sink for the rest of the run.
//   3. One slow article drags the average for every item after it, because a
//      mean has no way to treat it as the outlier it is.
//   4. Several stages screen with a pool of workers (abstract screening runs
//      eight at once), so completions arrive in CLUMPS: seven land within a
//      few milliseconds of each other, then nothing for the length of an
//      article. The gap between two completions is not the cost of an article
//      any more, and a median of those gaps reads the near-zero ones inside a
//      clump as the pace. That is the one that made the number swing hardest,
//      because it is wrong by a factor of the pool size.
//
// So: time each item, average the recent ones over a long window, count the
// time already spent on the item in flight against the estimate so the number
// only falls, and round the result into buckets that widen with the estimate so
// the text stops changing twice a second.
//
// The window is read in blocks: the mean WITHIN a block, because each
// completion carries the share of wall-clock time it is responsible for, so a
// block of them is the true pace of that stretch however they were grouped
// (eight landing together and eight landing one by one give the same answer,
// which is what (4) needs); then the median ACROSS the blocks, because a
// stalled article is in the few blocks that overlap it and in none of the
// rest, which is what (2) and (3) need.

import { useEffect, useRef, useState } from "react";

/** Per-item timings and what is currently displayed. */
export type EtaState = {
  /** Item count at the last change, so a jump of several is handled. */
  lastDone: number;
  /** When `lastDone` was reached. */
  lastChangeAt: number;
  /** Durations of recent items, newest last, in ms. */
  samples: number[];
  /** The smoothed pace in ms per item, or null before there is one. */
  rateMs: number | null;
  /** The bucketed estimate being shown, in ms, or null while unknown. */
  shownMs: number | null;
};

/** How many item timings to keep. Long enough to span several clumps from a
 *  worker pool and to dilute one slow article, short enough to follow a run
 *  that genuinely speeds up or slows down. */
export const WINDOW = 32;

/** Timings needed before a number is shown. Projecting one or two arrivals over
 *  a thousand articles is a guess, and with a pool of workers the first few
 *  arrivals are the opening clump, which says nothing about the pace yet. */
export const MIN_SAMPLES = 4;

/** How much of a new reading to take, per item. The blocks are accurate but
 *  they step as the window slides, and a number that is right to within a few
 *  per cent still reads as restless if it redraws every second. Applied per
 *  item rather than per report so a batch of eight moves the pace as far as
 *  eight arriving one at a time, and the two cannot disagree. */
export const SMOOTH = 0.3;

/** Completions per block. The largest worker pool any stage runs, so a block
 *  always spans at least one full cycle of it. */
export const BLOCK = 8;

export function initialEta(now: number, done = 0): EtaState {
  return { lastDone: done, lastChangeAt: now, samples: [], rateMs: null, shownMs: null };
}

/**
 * The pace, in ms per item, from the recent timings.
 *
 * Any run of consecutive samples covers exactly the wall-clock time those items
 * took, so its MEAN is a valid local pace however the completions were grouped:
 * eight landing together are eight samples that sum to what the clump cost.
 * Averaging inside a block is what makes a pool of workers readable at all.
 *
 * Taking the MEDIAN across the blocks is what keeps one stalled article from
 * becoming the pace of the whole run. The blocks slide one completion at a
 * time, so a stall, or the slow first article that loads the model, falls
 * inside the eight blocks that overlap it and outside every other, and over a
 * full window those are the minority. It takes a full window to do that: until
 * then a stall is in most of what there is to look at, and the estimate says
 * so rather than pretending otherwise.
 *
 * Neither alone is enough. A median of single samples reads a worker pool as
 * eight times faster than it is, because most of the gaps it sorts are the
 * milliseconds inside a clump; a mean over all of them charges every remaining
 * article its share of the one that stalled.
 */
function pace(samples: number[]): number {
  const n = samples.length;
  // A block has to span at least a pool's worth of completions, or it can land
  // wholly inside one clump and report that the run costs nothing. Eight is the
  // largest pool any stage uses.
  const w = Math.min(BLOCK, n);
  const means: number[] = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += samples[i];
    if (i >= w) sum -= samples[i - w];
    if (i >= w - 1) means.push(sum / w);
  }
  means.sort((a, b) => a - b);
  const mid = means.length >> 1;
  return means.length % 2 ? means[mid] : (means[mid - 1] + means[mid]) / 2;
}

/**
 * How coarsely to say an estimate of this size.
 *
 * The steps widen with the estimate because the precision a reviewer can use
 * widens too: five seconds is worth distinguishing from ten, fourteen minutes
 * is not worth distinguishing from fourteen minutes and ten seconds.
 *
 * A fixed step cannot do that across the whole range. A minute is a tenth of a
 * ten-minute estimate and a sixtieth of an hour-long one, so the old fixed
 * steps left a long run redrawing itself over noise far smaller than its own
 * error. The step is a share of the estimate instead, snapped to a unit a
 * person reads without counting.
 */
const STEPS_S = [5, 10, 15, 30, 60, 120, 300];

export function step(ms: number): number {
  const want = (Math.abs(ms) / 1000) * 0.08;
  return (STEPS_S.find(x => x >= want) ?? STEPS_S[STEPS_S.length - 1]) * 1000;
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

  let { lastDone, lastChangeAt, samples, rateMs } = state;

  if (done > lastDone) {
    // One or more items landed. Share the interval between them: a batch that
    // reports in one go did not take all that time on its last member.
    const landed = done - lastDone;
    const per = (now - lastChangeAt) / landed;
    const next = [...samples, ...Array(Math.min(landed, WINDOW)).fill(per)];
    samples = next.slice(-WINDOW);
    lastDone = done;
    lastChangeAt = now;
    if (samples.length >= MIN_SAMPLES) {
      const reading = pace(samples);
      const take = 1 - Math.pow(1 - SMOOTH, Math.min(landed, WINDOW));
      rateMs = rateMs === null ? reading : rateMs + take * (reading - rateMs);
    }
  } else if (done < lastDone) {
    // A new run reusing the same card. Start again rather than mixing rates.
    return initialEta(now, done);
  }

  if (rateMs === null || done >= total) {
    return { lastDone, lastChangeAt, samples, rateMs, shownMs: state.shownMs };
  }

  const rate = rateMs;
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
  return { lastDone, lastChangeAt, samples, rateMs, shownMs: moved ? rounded : shown };
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
