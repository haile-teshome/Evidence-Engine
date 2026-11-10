// The startup screen, and what is allowed to end a session.
//
// Run: pnpm test
//
// Two failures sat on top of each other every time the app opened. The screen
// painted before any script ran was thrown away the moment React mounted and
// replaced by a spinner React had just created, which never animated because
// the main thread was busy mounting the app: a still grey circle on a white
// page for several seconds. Underneath it, the session restore was asking the
// engine for the account before the engine had started, reading the refusal as
// "signed out", and deleting the stored token on the way.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

const apiFetch = vi.fn();
const store = new Map<string, string>();
vi.mock("../backendClient", () => ({
  apiFetch: (...a: any[]) => (apiFetch as any)(...a),
  getAuthToken: () => store.get("token") || "",
  setAuthSession: (t: string, r: string) => { store.set("token", t); store.set("reviewer", r); },
  clearAuthSession: () => { store.delete("token"); store.delete("reviewer"); },
}));

import { appReady, bootStep, startBootDeadline } from "../boot";
import { AuthProvider, useAuth } from "../auth";

/** The markup index.html paints, as the parser leaves it. */
function paintSplash() {
  document.body.innerHTML =
    '<div id="root"></div>' +
    '<div id="ee-boot" class="ee-boot"><div class="ee-boot__bar"></div><ul>' +
    '<li class="ee-step ee-step--running" data-boot-step="engine"></li>' +
    '<li class="ee-step" data-boot-step="session"></li>' +
    '<li class="ee-step" data-boot-step="workspace"></li>' +
    "</ul></div>";
  return document.getElementById("ee-boot")!;
}

const splashIsUp = () => {
  const el = document.getElementById("ee-boot");
  return !!el && !el.classList.contains("ee-boot--gone");
};
const stepClass = (step: string) =>
  document.querySelector(`[data-boot-step="${step}"]`)!.className;

/** An engine that has not started yet answers like the launcher's proxy does. */
const starting = () => Object.assign(new Error("The local engine is still starting."), { status: 503 });
const rejected = () => Object.assign(new Error("Invalid token"), { status: 401 });

describe("the startup screen", () => {
  // Most of these are about what the screen does, not how long it lingers, so
  // they run as though the page has been up a while and the minimum hold is
  // already satisfied. The one test below covers the hold itself.
  beforeEach(() => {
    vi.useRealTimers();
    vi.spyOn(performance, "now").mockReturnValue(5000);
  });
  afterEach(() => { vi.restoreAllMocks(); });

  it("leaves the screen alone until something real is mounted", () => {
    const el = paintSplash();
    expect(el.isConnected).toBe(true);
    expect(el.className).not.toContain("ee-boot--gone");
  });

  it("shows each step moving from waiting to running to done", () => {
    paintSplash();
    expect(stepClass("session")).not.toContain("ee-step--running");

    bootStep("session", "running");
    expect(stepClass("session")).toContain("ee-step--running");
    expect(stepClass("session")).not.toContain("ee-step--done");

    bootStep("session", "done");
    expect(stepClass("session")).toContain("ee-step--done");
    expect(stepClass("session")).not.toContain("ee-step--running");
  });

  it("fills the bar by thirds as the steps land", () => {
    paintSplash();
    const screen = () => document.getElementById("ee-boot")!.getAttribute("data-done");
    bootStep("engine", "done");
    expect(screen()).toBe("1");
    bootStep("session", "done");
    expect(screen()).toBe("2");
  });

  it("holds the app back while any step is still outstanding", () => {
    paintSplash();
    bootStep("engine", "done");
    bootStep("session", "done");
    expect(splashIsUp()).toBe(true);   // the workspace has not painted yet
  });

  it("shows the app the moment the last step ticks", () => {
    paintSplash();
    bootStep("engine", "done");
    bootStep("session", "done");
    bootStep("workspace", "done");
    expect(splashIsUp()).toBe(false);
  });

  it("does not wait on the AI model, which is not one of the steps", () => {
    // It loads for minutes on a cold start and the app works without it, so
    // holding the whole app behind it would be the worse complaint.
    paintSplash();
    expect(document.querySelectorAll("[data-boot-step]")).toHaveLength(3);
    expect(document.querySelector('[data-boot-step="model"]')).toBeNull();
  });

  it("costs nothing once the screen has gone", () => {
    paintSplash();
    appReady();
    expect(() => {
      bootStep("engine", "running");
      bootStep("workspace", "done");
    }).not.toThrow();
    expect(splashIsUp()).toBe(false);
  });

  it("shows the app anyway when a step never reports", () => {
    vi.useFakeTimers();
    paintSplash();
    startBootDeadline(30000);
    bootStep("engine", "running");
    vi.advanceTimersByTime(29000);
    expect(splashIsUp()).toBe(true);
    vi.advanceTimersByTime(2000);
    expect(splashIsUp()).toBe(false);
    vi.useRealTimers();
  });

  it("fades it out and then takes it off the page", async () => {
    const el = paintSplash();
    appReady();
    expect(el.className).toContain("ee-boot--gone");   // fading, still covering
    el.dispatchEvent(new Event("transitionend"));
    expect(el.isConnected).toBe(false);
  });

  it("removes it even where the fade never fires", async () => {
    vi.useFakeTimers();
    const el = paintSplash();
    appReady();
    vi.advanceTimersByTime(600);
    expect(el.isConnected).toBe(false);
    vi.useRealTimers();
  });

  it("gives the screen its moment however fast the app is ready", () => {
    // A warm start finishes all three steps inside a few hundred milliseconds,
    // and the screen came and went so quickly it read as a flicker.
    vi.useFakeTimers();
    const el = paintSplash();
    vi.spyOn(performance, "now").mockReturnValue(250);
    appReady();
    expect(splashIsUp()).toBe(true);

    vi.advanceTimersByTime(900);
    expect(splashIsUp()).toBe(true);          // still inside the minimum

    vi.spyOn(performance, "now").mockReturnValue(1300);
    vi.advanceTimersByTime(200);
    expect(splashIsUp()).toBe(false);
    expect(el.classList.contains("ee-boot--gone")).toBe(true);
    vi.useRealTimers();
  });

  it("waits once, not once per caller", () => {
    vi.useFakeTimers();
    paintSplash();
    vi.spyOn(performance, "now").mockReturnValue(100);
    appReady(); appReady(); appReady();
    vi.spyOn(performance, "now").mockReturnValue(1300);
    vi.advanceTimersByTime(1200);
    expect(splashIsUp()).toBe(false);
    vi.useRealTimers();
  });

  it("is safe to call twice, and with no screen there at all", () => {
    const el = paintSplash();
    appReady();
    el.dispatchEvent(new Event("transitionend"));
    expect(() => appReady()).not.toThrow();
    document.body.innerHTML = "";
    expect(() => appReady()).not.toThrow();
  });
});

describe("restoring the session while the engine is still starting", () => {
  beforeEach(() => { apiFetch.mockReset(); store.clear(); });
  afterEach(() => { vi.useRealTimers(); });

  const mount = () => renderHook(() => useAuth(), { wrapper: AuthProvider });

  it("waits for the engine instead of signing the reviewer out", async () => {
    store.set("token", "t1");
    apiFetch.mockRejectedValueOnce(starting())
            .mockRejectedValueOnce(starting())
            .mockResolvedValueOnce({ user: { id: "u1", email: "a@b.c" } });

    const { result } = mount();
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("authed"), { timeout: 3000 });
    expect(result.current.user?.id).toBe("u1");
    expect(store.get("token")).toBe("t1");   // never thrown away
  });

  it("keeps the stored token through a refusal that is not about the token", async () => {
    store.set("token", "t1");
    apiFetch.mockRejectedValue(starting());
    mount();
    await new Promise(r => setTimeout(r, 400));
    expect(store.get("token")).toBe("t1");
  });

  it("ends the session when the engine rejects the token", async () => {
    store.set("token", "stale");
    apiFetch.mockRejectedValue(rejected());

    const { result } = mount();
    await waitFor(() => expect(result.current.status).toBe("anon"));
    expect(store.has("token")).toBe(false);
    expect(apiFetch).toHaveBeenCalledTimes(1);   // no point asking again
  });

  it("does not call the engine at all with no token to restore", async () => {
    const { result } = mount();
    await waitFor(() => expect(result.current.status).toBe("anon"));
    expect(apiFetch).not.toHaveBeenCalled();
  });
});
