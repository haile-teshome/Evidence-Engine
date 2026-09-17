// Global test setup. Runs before every test file.
//
// Three jobs:
//   1. jest-dom matchers (toBeInTheDocument and friends).
//   2. A real IndexedDB, because the store persists snapshots through it and a
//      missing implementation would make every persistence test silently pass
//      by falling into the localStorage fallback.
//   3. Stub the browser APIs jsdom does not implement, so a component that
//      merely *renders* does not fail on an unrelated missing global.
import "@testing-library/jest-dom/vitest";
import "fake-indexeddb/auto";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Node 22+ exposes its own experimental global `localStorage`, which shadows
// jsdom's and is a non-functional stub unless --localstorage-file is given.
// Install a real Map-backed Storage so persistence behaves like a browser's
// regardless of which Node is running the suite.
function installStorage(name: "localStorage" | "sessionStorage") {
  const existing = (globalThis as any)[name];
  if (existing && typeof existing.clear === "function") return;
  const store = new Map<string, string>();
  const impl: Storage = {
    get length() { return store.size; },
    clear: () => store.clear(),
    getItem: (k: string) => (store.has(String(k)) ? store.get(String(k))! : null),
    key: (i: number) => Array.from(store.keys())[i] ?? null,
    removeItem: (k: string) => void store.delete(String(k)),
    setItem: (k: string, v: string) => void store.set(String(k), String(v)),
  };
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value: impl });
  if (typeof window !== "undefined") {
    Object.defineProperty(window, name, { configurable: true, writable: true, value: impl });
  }
}

installStorage("localStorage");
installStorage("sessionStorage");

const clearStorage = (s: Storage | undefined) => {
  try { s?.clear?.(); } catch { /* opaque origin, or a stubbed store */ }
};

afterEach(() => {
  cleanup();
  clearStorage(globalThis.localStorage);
  clearStorage(globalThis.sessionStorage);
  vi.restoreAllMocks();
});

// jsdom implements neither of these, and Radix/shadcn components use both.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as any;
}

for (const name of ["ResizeObserver", "IntersectionObserver"] as const) {
  if (!(globalThis as any)[name]) {
    (globalThis as any)[name] = class {
      observe() {}
      unobserve() {}
      disconnect() {}
      takeRecords() { return []; }
    };
  }
}

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

// jsdom has no layout engine, so anything measuring an element gets zeros.
if (!(HTMLCanvasElement.prototype as any).getContext) {
  (HTMLCanvasElement.prototype as any).getContext = () => null;
}

// Object URLs are used for PDF blobs; jsdom does not provide them.
if (!URL.createObjectURL) {
  URL.createObjectURL = (() => "blob:mock") as any;
  URL.revokeObjectURL = (() => {}) as any;
}

// No test may reach the network. A component that fetches on mount should be
// given an explicit stub; hitting this means one was missed.
vi.stubGlobal(
  "fetch",
  vi.fn(() =>
    Promise.reject(new Error("Unstubbed fetch in a test. Mock it explicitly.")),
  ),
);
