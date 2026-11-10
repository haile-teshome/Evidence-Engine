// The startup screen: ticking its steps off, and taking it away.
//
// index.html paints it before a line of JavaScript runs, outside #root so that
// mounting React cannot destroy it. That placement is the whole point: its
// animation is already running on the compositor by the time the bundle starts
// executing, so it keeps moving through a busy main thread. Anything React
// renders instead is born mid-block and sits frozen, which is what made the
// app look hung at exactly the moment it was working hardest.
//
// So nothing dismisses it on a timer or on "the script loaded". The app is
// shown when it is genuinely ready: the engine answers, the session is back,
// and the first page has been painted. Each of those is a step on the screen,
// so the wait reads as work being done rather than as a stall.
//
// The one thing deliberately NOT waited on is the AI model. It loads in the
// background for minutes on a cold start, the app is fully usable without it
// (everything but screening), and the header already reports it. Holding the
// whole app behind it would trade one complaint for a worse one.

/** The steps shown on the startup screen, in the order index.html lists them. */
export type BootStep = "engine" | "session" | "workspace";
export type StepState = "running" | "done";

const REQUIRED: BootStep[] = ["engine", "session", "workspace"];

/**
 * The least time the startup screen stays on screen, measured from the page
 * starting to load.
 *
 * A warm start has all three steps done inside a couple of hundred
 * milliseconds, and the screen came and went so fast it read as a flicker
 * rather than as the app starting. It is also just longer than the entrance
 * animation takes to finish, so the screen is never whisked away mid-arrival.
 */
const MIN_ON_SCREEN_MS = 1200;

const row = (step: BootStep): Element | null =>
  typeof document === "undefined" ? null : document.querySelector(`[data-boot-step="${step}"]`);

/**
 * Report where a step has got to.
 *
 * The screen is the only state there is: a step is done because its row says
 * so. That keeps this honest when the screen has already gone (nothing to
 * update, nothing to reveal) and keeps repeat calls free.
 */
export function bootStep(step: BootStep, state: StepState): void {
  const el = row(step);
  if (el) {
    el.classList.toggle("ee-step--running", state === "running");
    el.classList.toggle("ee-step--done", state === "done");
  }
  const done = REQUIRED.filter(s => row(s)?.classList.contains("ee-step--done")).length;
  // The bar fills by thirds. It is the one part of the screen that says how far
  // along this is rather than merely that something is happening.
  const screen = document.getElementById("ee-boot");
  if (screen) screen.setAttribute("data-done", String(done));
  if (done === REQUIRED.length) appReady();
}

/** Reveal the app now, whatever the steps say. */
export function appReady(): void {
  if (typeof document === "undefined") return;
  const el = document.getElementById("ee-boot");
  if (!el || el.classList.contains("ee-boot--gone")) return;   // already going
  const age = typeof performance === "undefined" ? MIN_ON_SCREEN_MS : performance.now();
  if (age < MIN_ON_SCREEN_MS) {
    // Ready early. Let the screen have its moment, then go. The app is already
    // built and waiting behind it, so this costs the reviewer nothing but the
    // difference between seeing the app start and seeing something blink.
    if (el.dataset.leaving) return;              // a wait is already running
    el.dataset.leaving = "1";
    setTimeout(() => { delete el.dataset.leaving; appReady(); }, MIN_ON_SCREEN_MS - age);
    return;
  }
  el.classList.add("ee-boot--gone");
  const drop = () => el.remove();
  el.addEventListener("transitionend", drop, { once: true });
  // A transition that never fires (reduced motion, a hidden window) must not
  // leave the screen covered, so the fade has a deadline of its own.
  setTimeout(drop, 500);
}

/**
 * The backstop: show the app even if a step never reports.
 *
 * Long enough that a cold engine (a fresh Python process importing its world)
 * finishes inside it, short enough that a broken one does not leave someone
 * watching a progress bar with nothing behind it.
 */
export function startBootDeadline(ms = 30000): void {
  if (typeof window === "undefined") return;
  window.setTimeout(appReady, ms);
}
