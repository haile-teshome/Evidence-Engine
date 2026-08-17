// ---------------------------------------------------------------------------
// Credential store for cloud LLM provider keys, with two at-rest options:
//
//   • "keychain"  — keys live in the OS keychain via the local backend. The
//                   backend reads them directly when building a model; they
//                   never come back to the browser. Strongest at rest.
//   • "encrypted" — keys are AES-GCM encrypted in this browser with a passphrase
//                   you set (Web Crypto, PBKDF2-derived key). The ciphertext sits
//                   in localStorage; you unlock once per session, and the
//                   decrypted keys are sent to the local backend as request
//                   headers for that session only.
//
// Nothing is ever uploaded off this device except the request to the provider
// you chose. No plaintext key is written to disk in either mode.
// ---------------------------------------------------------------------------

export type LlmProvider = "openai" | "anthropic" | "google";
export const LLM_PROVIDERS: LlmProvider[] = ["openai", "anthropic", "google"];
export type StorageMode = "keychain" | "encrypted";

const MODE_KEY = "ee:key-mode";
const ENC_KEY = "ee:llm-keys-enc:v1";
const LEGACY_PLAINTEXT_KEY = "ee:llm-keys:v1";   // from an earlier plaintext build; purge on load

const baseUrl = (import.meta as any)?.env?.VITE_API_BASE_URL || "/api";

// ---- in-memory state -------------------------------------------------------
let unlocked: Record<LlmProvider, string> | null = null;      // encrypted mode, after unlock
let keychainAvailable = false;
let keychainProviders: Record<LlmProvider, boolean> = { openai: false, anthropic: false, google: false };
let statusLoaded = false;

const listeners = new Set<() => void>();
function emit() { listeners.forEach(l => l()); }
export function subscribe(l: () => void): () => void { listeners.add(l); return () => listeners.delete(l); }

// Purge any plaintext keys left by an earlier build the first time we load.
try { if (localStorage.getItem(LEGACY_PLAINTEXT_KEY)) localStorage.removeItem(LEGACY_PLAINTEXT_KEY); } catch { /* ignore */ }

// ---- mode ------------------------------------------------------------------
export function getMode(): StorageMode {
  const m = (() => { try { return localStorage.getItem(MODE_KEY); } catch { return null; } })();
  if (m === "keychain" || m === "encrypted") return m;
  return keychainAvailable ? "keychain" : "encrypted";
}
export function setMode(mode: StorageMode) {
  try { localStorage.setItem(MODE_KEY, mode); } catch { /* ignore */ }
  emit();
}

// ---- encrypted (browser) storage ------------------------------------------
type EncBlob = { v: 1; salt: string; iv: string; ct: string };

function b64(bytes: Uint8Array): string { let s = ""; bytes.forEach(b => (s += String.fromCharCode(b))); return btoa(s); }
function ub64(s: string): Uint8Array { return Uint8Array.from(atob(s), c => c.charCodeAt(0)); }

async function deriveKey(passphrase: string, salt: Uint8Array): Promise<CryptoKey> {
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(passphrase), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: 200_000, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"],
  );
}

function readEncBlob(): EncBlob | null {
  try { const raw = localStorage.getItem(ENC_KEY); return raw ? JSON.parse(raw) as EncBlob : null; } catch { return null; }
}

export function hasEncrypted(): boolean { return !!readEncBlob(); }
export function isUnlocked(): boolean { return unlocked !== null; }

export async function saveEncrypted(keys: Record<LlmProvider, string>, passphrase: string): Promise<void> {
  const clean: Record<string, string> = {};
  LLM_PROVIDERS.forEach(p => { if (keys[p]?.trim()) clean[p] = keys[p].trim(); });
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const data = new TextEncoder().encode(JSON.stringify(clean));
  const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, data));
  const blob: EncBlob = { v: 1, salt: b64(salt), iv: b64(iv), ct: b64(ct) };
  try { localStorage.setItem(ENC_KEY, JSON.stringify(blob)); } catch { /* ignore */ }
  unlocked = { openai: "", anthropic: "", google: "", ...clean };
  emit();
}

// Unlock the stored ciphertext into memory. Throws if the passphrase is wrong.
export async function unlock(passphrase: string): Promise<void> {
  const blob = readEncBlob();
  if (!blob) throw new Error("No encrypted keys stored");
  const salt = ub64(blob.salt), iv = ub64(blob.iv), ct = ub64(blob.ct);
  const key = await deriveKey(passphrase, salt);
  let plain: string;
  try {
    const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
    plain = new TextDecoder().decode(pt);
  } catch {
    throw new Error("Incorrect passphrase");
  }
  const parsed = JSON.parse(plain) as Record<string, string>;
  unlocked = { openai: "", anthropic: "", google: "", ...parsed };
  emit();
}

export function lock() { unlocked = null; emit(); }

export function clearEncrypted() {
  try { localStorage.removeItem(ENC_KEY); } catch { /* ignore */ }
  unlocked = null;
  emit();
}

// The decrypted keys for the current session (encrypted mode), or nulls.
export function unlockedKeys(): Record<LlmProvider, string> | null { return unlocked; }

// ---- keychain (backend) storage -------------------------------------------
export function keychainIsAvailable(): boolean { return keychainAvailable; }
export function keychainStatus(): Record<LlmProvider, boolean> { return keychainProviders; }

export async function refreshKeychainStatus(): Promise<void> {
  try {
    const r = await fetch(`${baseUrl}/keys/status`);
    const d = await r.json();
    keychainAvailable = !!d?.available;
    keychainProviders = { openai: false, anthropic: false, google: false, ...(d?.providers || {}) };
  } catch {
    keychainAvailable = false;
  }
  statusLoaded = true;
  emit();
}

export async function keychainSet(provider: LlmProvider, key: string): Promise<void> {
  const r = await fetch(`${baseUrl}/keys/set`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, key }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail || "Could not save to keychain");
  const d = await r.json();
  keychainProviders = { openai: false, anthropic: false, google: false, ...(d?.providers || {}) };
  emit();
}

export async function keychainDelete(provider: LlmProvider): Promise<void> {
  const r = await fetch(`${baseUrl}/keys/delete`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (r.ok) {
    const d = await r.json();
    keychainProviders = { openai: false, anthropic: false, google: false, ...(d?.providers || {}) };
    emit();
  }
}

// ---- what the request layer / UI ask for ----------------------------------

// Headers to attach to backend requests. In keychain mode the backend reads the
// key itself, so no header is sent. In encrypted mode we send the unlocked keys.
export function requestHeaders(): Record<string, string> {
  if (getMode() !== "encrypted" || !unlocked) return {};
  const h: Record<string, string> = {};
  if (unlocked.openai) h["X-LLM-OpenAI-Key"] = unlocked.openai;
  if (unlocked.anthropic) h["X-LLM-Anthropic-Key"] = unlocked.anthropic;
  if (unlocked.google) h["X-LLM-Google-Key"] = unlocked.google;
  return h;
}

// Is a usable key available RIGHT NOW for this provider (for the sidebar hint)?
export function providerReady(provider: LlmProvider): boolean {
  if (getMode() === "keychain") return !!keychainProviders[provider];
  return !!(unlocked && unlocked[provider]);
}

// True when encrypted keys exist but the session is locked (needs a passphrase).
export function needsUnlock(): boolean {
  return getMode() === "encrypted" && hasEncrypted() && !isUnlocked();
}

export function statusReady(): boolean { return statusLoaded; }

// Kick off a status fetch once at import so the UI knows if keychain exists.
void refreshKeychainStatus();
