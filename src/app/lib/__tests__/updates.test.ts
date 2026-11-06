// Tests for the release-check client.
//
// Run: pnpm test
//
// `apiFetch` is mocked. The cases that matter are the ones that decide whether
// a banner appears: a failed check must be silent rather than throwing into a
// page load, and a dismissal must be remembered per version so the next release
// still notifies.
import { describe, it, expect, beforeEach, vi } from "vitest";

import * as backendClient from "../backendClient";
import { getUpdate, dismiss, isDismissed, shouldNotify, type UpdateInfo } from "../updates";

function info(over: Partial<UpdateInfo> = {}): UpdateInfo {
  return {
    current: "1.0.0", latest: "v2.0.0", newer: true, url: "https://example.test/r",
    name: "Evidence Engine v2.0.0", notes: "", published_at: "", checked_at: 1,
    enabled: true, error: "",
    ...over,
  };
}

function mockApi(handler: (path: string) => any) {
  const paths: string[] = [];
  vi.spyOn(backendClient, "apiFetch").mockImplementation((async (path: string) => {
    paths.push(path);
    return handler(path);
  }) as any);
  return paths;
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("getUpdate", () => {
  it("asks the backend and returns the check", async () => {
    const paths = mockApi(() => info());
    const out = await getUpdate();
    expect(paths[0]).toBe("/update");
    expect(out?.latest).toBe("v2.0.0");
  });

  it("passes force through for a manual re-check", async () => {
    const paths = mockApi(() => info());
    await getUpdate(true);
    expect(paths[0]).toContain("force=true");
  });

  it("returns null instead of throwing when the backend is unreachable", async () => {
    mockApi(() => {
      throw new Error("backend still starting");
    });
    await expect(getUpdate()).resolves.toBeNull();
  });

  it("returns null on a non-object answer", async () => {
    mockApi(() => "not json");
    await expect(getUpdate()).resolves.toBeNull();
  });
});

describe("dismissal", () => {
  it("remembers the dismissed version", () => {
    dismiss("v2.0.0");
    expect(isDismissed("v2.0.0")).toBe(true);
  });

  it("does not suppress a later release", () => {
    dismiss("v2.0.0");
    expect(isDismissed("v3.0.0")).toBe(false);
  });

  it("treats an empty version as not dismissed", () => {
    expect(isDismissed("")).toBe(false);
  });
});

describe("shouldNotify", () => {
  it("notifies for a newer release", () => {
    expect(shouldNotify(info())).toBe(true);
  });

  it("stays silent when the check failed", () => {
    expect(shouldNotify(null)).toBe(false);
  });

  it("stays silent when there is nothing newer", () => {
    expect(shouldNotify(info({ newer: false }))).toBe(false);
  });

  it("stays silent when checking is switched off", () => {
    expect(shouldNotify(info({ enabled: false }))).toBe(false);
  });

  it("stays silent with no link to offer", () => {
    expect(shouldNotify(info({ url: "" }))).toBe(false);
  });

  it("stays silent once that version is dismissed", () => {
    dismiss("v2.0.0");
    expect(shouldNotify(info())).toBe(false);
  });

  it("notifies again when a newer release follows the dismissed one", () => {
    dismiss("v2.0.0");
    expect(shouldNotify(info({ latest: "v2.1.0" }))).toBe(true);
  });
});
