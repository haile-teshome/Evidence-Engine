// Tests for the encrypted API-key store. No network, no DOM beyond a storage stub.
//
// Run: npx vitest run
//
// This is the only cryptography in the codebase, and it guards the user's LLM
// billing keys. The failure modes are quiet and bad in both directions: keys that
// look saved but cannot be recovered, or ciphertext that a wrong passphrase
// happily "decrypts" into garbage the app then sends to a provider.
//
// AES-GCM is authenticated, so a wrong passphrase must REJECT rather than return
// nonsense. That property is the single most important thing here.
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

const ENC_KEY = "ee:llm-keys-enc:v1";  // must match keystore.ts

import {
  LLM_PROVIDERS,
  getMode,
  setMode,
  hasEncrypted,
  isUnlocked,
  saveEncrypted,
  unlock,
  lock,
  clearEncrypted,
  unlockedKeys,
  providerReady,
  needsUnlock,
  requestHeaders,
} from "../keystore";

const KEYS = {
  openai: "sk-openai-secret-0001",
  anthropic: "sk-ant-secret-0002",
  google: "AIza-secret-0003",
} as const;

const PASS = "correct horse battery staple";

beforeEach(() => {
  localStorage.clear();
  lock();
});
afterEach(() => {
  localStorage.clear();
  lock();
});

describe("storage mode", () => {
  it("has a valid default", () => {
    expect(["keychain", "encrypted"]).toContain(getMode());
  });

  it("round-trips a mode change", () => {
    setMode("encrypted");
    expect(getMode()).toBe("encrypted");
  });

  it("recovers from a corrupt stored mode", () => {
    localStorage.setItem("ee:key-mode", "garbage-value");
    expect(["keychain", "encrypted"]).toContain(getMode());
  });
});

describe("encrypt and decrypt round trip", () => {
  it("starts with nothing stored", () => {
    expect(hasEncrypted()).toBe(false);
    expect(isUnlocked()).toBe(false);
  });

  it("saves, locks, and recovers every key with the right passphrase", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    lock();
    expect(isUnlocked()).toBe(false);
    await unlock(PASS);
    expect(unlockedKeys()).toEqual({ ...KEYS });
  });

  it("reports that something is stored after saving", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    expect(hasEncrypted()).toBe(true);
  });

  it("REJECTS a wrong passphrase rather than returning garbage", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    lock();
    await expect(unlock("the wrong passphrase")).rejects.toThrow();
    expect(isUnlocked()).toBe(false);
  });

  it("leaves no key readable while locked", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    lock();
    expect(unlockedKeys()).toBeNull();
  });

  it("never writes a key in plaintext to storage", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    const dump = JSON.stringify(
      Object.fromEntries(
        Array.from({ length: localStorage.length }, (_, i) => {
          const k = localStorage.key(i)!;
          return [k, localStorage.getItem(k)];
        }),
      ),
    );
    for (const secret of Object.values(KEYS)) expect(dump).not.toContain(secret);
    expect(dump).not.toContain(PASS);
  });

  it("produces different ciphertext for the same input each time", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    const first = localStorage.getItem(ENC_KEY);
    localStorage.clear();
    await saveEncrypted({ ...KEYS }, PASS);
    const second = localStorage.getItem(ENC_KEY);
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    // A fresh salt and IV per save. Identical ciphertext would leak that the
    // same keys were stored twice.
    expect(first).not.toBe(second);
  });

  it("clearEncrypted removes the blob and the unlocked keys", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    clearEncrypted();
    expect(hasEncrypted()).toBe(false);
    expect(unlockedKeys()).toBeNull();
  });

  it("rejects unlocking when nothing has been stored", async () => {
    await expect(unlock(PASS)).rejects.toThrow();
  });

  it("survives a corrupt ciphertext blob without hanging or returning keys", async () => {
    localStorage.setItem(ENC_KEY, "not-valid-json{{{");
    await expect(unlock(PASS)).rejects.toThrow();
    expect(unlockedKeys()).toBeNull();
  });

  it("rejects a blob whose ciphertext has been tampered with", async () => {
    await saveEncrypted({ ...KEYS }, PASS);
    const raw = localStorage.getItem(ENC_KEY);
    expect(raw).toBeTruthy();
    const blob = JSON.parse(raw!);
    blob.ct = blob.ct.slice(0, -8) + "AAAAAAAA";
    localStorage.setItem(ENC_KEY, JSON.stringify(blob));
    lock();
    // AES-GCM authenticates the ciphertext, so tampering must fail loudly.
    await expect(unlock(PASS)).rejects.toThrow();
  });

  it("round-trips unicode and very long keys", async () => {
    const odd = { openai: "sk-" + "x".repeat(4000), anthropic: "ключ-日本語-🔐", google: "" } as any;
    await saveEncrypted(odd, PASS);
    lock();
    await unlock(PASS);
    expect(unlockedKeys()).toEqual(odd);
  });

  it("round-trips an empty passphrase if the app allows one", async () => {
    await saveEncrypted({ ...KEYS }, "");
    lock();
    await unlock("");
    expect(unlockedKeys()).toEqual({ ...KEYS });
  });
});

describe("readiness and headers", () => {
  it("no provider is ready before anything is stored", () => {
    for (const p of LLM_PROVIDERS) expect(providerReady(p)).toBe(false);
  });

  it("providers become ready once unlocked", async () => {
    setMode("encrypted");
    await saveEncrypted({ ...KEYS }, PASS);
    await unlock(PASS);
    expect(providerReady("openai")).toBe(true);
  });

  it("needsUnlock is true when a blob exists but is locked", async () => {
    setMode("encrypted");
    await saveEncrypted({ ...KEYS }, PASS);
    lock();
    expect(needsUnlock()).toBe(true);
  });

  it("needsUnlock is false once unlocked", async () => {
    setMode("encrypted");
    await saveEncrypted({ ...KEYS }, PASS);
    await unlock(PASS);
    expect(needsUnlock()).toBe(false);
  });

  it("sends no key headers while locked", async () => {
    setMode("encrypted");
    await saveEncrypted({ ...KEYS }, PASS);
    lock();
    const h = requestHeaders();
    for (const secret of Object.values(KEYS)) {
      expect(JSON.stringify(h)).not.toContain(secret);
    }
  });

  it("sends key headers once unlocked", async () => {
    setMode("encrypted");
    await saveEncrypted({ ...KEYS }, PASS);
    await unlock(PASS);
    expect(JSON.stringify(requestHeaders())).toContain(KEYS.openai);
  });

  it("omits a provider with a blank key", async () => {
    setMode("encrypted");
    await saveEncrypted({ openai: "", anthropic: "", google: "" } as any, PASS);
    await unlock(PASS);
    expect(providerReady("openai")).toBe(false);
  });
});
