// The time-left label on a running job.
//
// Run: pnpm test
//
// The maths lives in lib/eta and is tested there. What is tested here is that
// the card is wired to it: the estimate shown on screen has to come down as
// work proceeds, because the complaint that prompted this was that the number
// bounced up and down while nothing was wrong.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";

import { TaskProgressCard } from "../TaskProgressCard";
import type { TaskRecord } from "../../lib/store";

const task = (done: number, total: number, status: TaskRecord["status"] = "running"): TaskRecord =>
  ({
    kind: "text-extract", status, startedAt: Date.now() - 1000,
    detail: "", log: [], progress: { done, total },
  } as unknown as TaskRecord);

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-11-07T12:00:00Z"));
});
afterEach(() => vi.useRealTimers());

/** Move the clock and let the card's timers fire. */
const advance = async (ms: number) => {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
};

describe("time left", () => {
  it("says it is estimating before anything has finished", async () => {
    const view = render(<TaskProgressCard task={task(0, 10)} title="Extracting" />);
    await advance(600);
    expect(view.container.textContent).toContain("estimating");
  });

  it("gives a number once items have landed, and counts down from it", async () => {
    // Four items in forty seconds: ten seconds each, six left, so about a
    // minute. Then thirty seconds pass with nothing landing, and the estimate
    // has to be smaller, not larger.
    const view = render(<TaskProgressCard task={task(0, 10)} title="Extracting" />);
    await advance(40_000);
    view.rerender(<TaskProgressCard task={task(4, 10)} title="Extracting" />);
    await advance(600);

    const first = view.container.textContent || "";
    expect(first).toMatch(/left/);
    // Labels read "~1m left", "~1m 30s left" or "~45s left".
    const seconds = (text: string) => {
      const m = text.match(/~\s*(?:(\d+)m)?\s*(?:(\d+)s)?\s*left/);
      return m ? Number(m[1] || 0) * 60 + Number(m[2] || 0) : NaN;
    };
    const before = seconds(first);
    expect(before).toBeGreaterThan(0);

    await advance(30_000);
    const after = seconds(view.container.textContent || "");
    expect(after).toBeLessThanOrEqual(before);
  });

  it("says nothing about time once the job is not running", async () => {
    const view = render(<TaskProgressCard task={task(10, 10, "done")} title="Extracting" />);
    await advance(600);
    expect(view.container.textContent).not.toMatch(/left|estimating/);
  });
});
