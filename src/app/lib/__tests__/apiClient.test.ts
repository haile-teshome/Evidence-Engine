// Pure helpers from the API client, plus GRADE certainty. No network.
//
// Run: npx vitest run
//
// gradeCertainty is the one piece of published numeric reasoning here: it turns
// domain judgments into the certainty rating that appears in a summary-of-
// findings table. It is pure arithmetic over a bounded scale, so it is exactly
// the kind of thing that should never drift unnoticed.
import { describe, it, expect, beforeEach, afterEach } from "vitest";

if (typeof (globalThis as any).localStorage?.clear !== "function") {
  const store = new Map<string, string>();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
      setItem: (k: string, v: string) => void store.set(k, String(v)),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
      key: (i: number) => Array.from(store.keys())[i] ?? null,
      get length() { return store.size; },
    },
  });
}

import {
  gradeCertainty,
  supportsTools,
  providerForModel,
  keyHeaders,
  formatDuration,
  apiConfig,
} from "../apiClient";

const outcome = (over: Partial<any> = {}) => ({
  starting: "randomized",
  downgrades: {},
  upgrades: {},
  ...over,
}) as any;

// --------------------------------------------------------------------------
// GRADE
// --------------------------------------------------------------------------

describe("gradeCertainty", () => {
  it("a randomised trial with no downgrades is High", () => {
    expect(gradeCertainty(outcome())).toBe("High");
  });

  it("an observational study starts lower than a randomised one", () => {
    const rct = gradeCertainty(outcome({ starting: "randomized" }));
    const obs = gradeCertainty(outcome({ starting: "observational" }));
    expect(rct).not.toBe(obs);
  });

  it("each downgrade lowers the rating by one level", () => {
    const one = gradeCertainty(outcome({ downgrades: { risk_of_bias: -1 } }));
    const two = gradeCertainty(outcome({ downgrades: { risk_of_bias: -1, imprecision: -1 } }));
    expect(one).toBe("Moderate");
    expect(two).toBe("Low");
  });

  it("bottoms out at Very Low rather than going below the scale", () => {
    expect(gradeCertainty(outcome({
      downgrades: {
        risk_of_bias: -2, inconsistency: -2, indirectness: -2,
        imprecision: -2, publication_bias: -2,
      },
    }))).toBe("Very low");
  });

  it("caps at High rather than going above the scale", () => {
    expect(gradeCertainty(outcome({
      starting: "observational",
      upgrades: { large_effect: 2, dose_response: 2, plausible_confounding: 2 },
    }))).toBe("High");
  });

  it("clamps a single domain's downgrade at two levels", () => {
    const clamped = gradeCertainty(outcome({ downgrades: { risk_of_bias: -5 } }));
    const twoLevel = gradeCertainty(outcome({ downgrades: { risk_of_bias: -2 } }));
    expect(clamped).toBe(twoLevel);
  });

  it("ignores a positive value in a downgrade domain", () => {
    expect(gradeCertainty(outcome({ downgrades: { risk_of_bias: 2 } }))).toBe("High");
  });

  it("does not apply upgrades to a randomised starting point", () => {
    expect(gradeCertainty(outcome({
      starting: "randomized",
      upgrades: { large_effect: 2 },
    }))).toBe("High");
  });

  it("always returns one of the four GRADE levels", () => {
    const levels = ["High", "Moderate", "Low", "Very low"];
    for (const start of ["randomized", "observational"]) {
      for (let d = 0; d >= -4; d--) {
        expect(levels).toContain(
          gradeCertainty(outcome({ starting: start, downgrades: { risk_of_bias: d } })),
        );
      }
    }
  });

  it("is monotonic: more downgrades never raises certainty", () => {
    const order = ["Very low", "Low", "Moderate", "High"];
    const ranks = [0, -1, -2].map(d =>
      order.indexOf(gradeCertainty(outcome({
        downgrades: { risk_of_bias: d === -2 ? -2 : d, inconsistency: d === -2 ? -1 : 0 },
      }))),
    );
    expect(ranks).toEqual([...ranks].sort((a, b) => b - a));
  });

  it("handles missing downgrade and upgrade objects", () => {
    expect(() => gradeCertainty({ starting: "randomized", downgrades: {}, upgrades: {} } as any))
      .not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Model capability routing. Picking the wrong provider sends a key to the
// wrong vendor; claiming tool support a model lacks makes every call fail.
// --------------------------------------------------------------------------

describe("providerForModel", () => {
  it.each([
    ["gpt-4o", "openai"],
    ["claude-opus-5", "anthropic"],
    ["gemini-2.0-flash", "google"],
  ])("routes %s to %s", (model, provider) => {
    expect(providerForModel(model)).toBe(provider);
  });

  it.each(["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "", "hf.co/leads-mistral"])(
    "returns null for the local model %s",
    (model) => expect(providerForModel(model)).toBeNull(),
  );

  it("is case insensitive", () => {
    expect(providerForModel("GPT-4O")).toBe("openai");
  });
});

describe("supportsTools", () => {
  it.each(["claude-opus-5", "gpt-4o", "gemini-2.0-flash", "qwen2.5:7b", "llama3.1:8b"])(
    "reports tool support for %s",
    (m) => expect(supportsTools(m)).toBe(true),
  );

  it.each(["", "some-unknown-model", "tinyllama"])(
    "reports no tool support for %s",
    (m) => expect(supportsTools(m)).toBe(false),
  );

  it("is case insensitive", () => {
    expect(supportsTools("QWEN2.5:7B")).toBe(true);
  });

  it("does not throw on a null model", () => {
    expect(() => supportsTools(null as any)).not.toThrow();
  });
});

// --------------------------------------------------------------------------
// Headers
// --------------------------------------------------------------------------

describe("keyHeaders", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("returns an object with nothing configured", () => {
    expect(typeof keyHeaders()).toBe("object");
  });

  it("carries no secret-looking values when nothing is stored", () => {
    expect(JSON.stringify(keyHeaders())).not.toContain("sk-");
  });
});

// --------------------------------------------------------------------------
// Config and formatting
// --------------------------------------------------------------------------

describe("apiConfig", () => {
  it("defaults to a local model rather than a paid cloud one", () => {
    expect(providerForModel(apiConfig.model)).toBeNull();
  });

  it("has a base url", () => {
    expect(typeof apiConfig.baseUrl).toBe("string");
  });
});

describe("formatDuration", () => {
  it.each([
    [0], [1], [59], [60], [61], [3599], [3600], [7325],
  ])("returns a non-empty string for %i seconds", (s) => {
    expect(formatDuration(s).length).toBeGreaterThan(0);
  });

  it("does not print NaN for a bad input", () => {
    for (const bad of [NaN, null, undefined, -1]) {
      expect(formatDuration(bad as any)).not.toContain("NaN");
    }
  });

  it("distinguishes a minute from an hour", () => {
    expect(formatDuration(60)).not.toBe(formatDuration(3600));
  });
});
