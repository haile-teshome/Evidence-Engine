// ---------------------------------------------------------------------------
// Data-source (database) API keys: CORE, Semantic Scholar, and Scopus.
//
// These are separate from the LLM keystore: they are lower-sensitivity than LLM
// billing keys, and, crucially, the backend reads them from x-db-*-key request
// headers via get_cred() in EVERY mode (there is no server-side keychain path for
// them). So unlike LLM keys, they must ship as headers regardless of the LLM
// storage mode. They live here, on-device, and are attached to each request.
// ---------------------------------------------------------------------------

export type DbSource = "core" | "semantic_scholar" | "scopus" | "ncbi" | "springer" | "ieee" | "wos";
export const DB_SOURCES: DbSource[] = ["core", "semantic_scholar", "scopus", "ncbi", "springer", "ieee", "wos"];

const PREFIX = "ee:db-key:";
const HEADER: Record<DbSource, string> = {
  core: "X-DB-Core-Key",
  semantic_scholar: "X-DB-Semantic-Scholar-Key",
  scopus: "X-DB-Scopus-Key",
  ncbi: "X-DB-NCBI-Key",
  springer: "X-DB-Springer-Key",
  ieee: "X-DB-IEEE-Key",
  wos: "X-DB-WOS-Key",
};

const listeners = new Set<() => void>();
export function subscribeDbKeys(l: () => void): () => void { listeners.add(l); return () => listeners.delete(l); }
function emit() { listeners.forEach(l => l()); }

export function getDbKey(source: DbSource): string {
  try { return localStorage.getItem(PREFIX + source) || ""; } catch { return ""; }
}
export function setDbKey(source: DbSource, key: string): void {
  try {
    const k = key.trim();
    if (k) localStorage.setItem(PREFIX + source, k);
    else localStorage.removeItem(PREFIX + source);
  } catch { /* ignore */ }
  emit();
}
export function hasDbKey(source: DbSource): boolean { return !!getDbKey(source); }

// ---------------------------------------------------------------------------
// Contact email. Not a key and not an account: several scholarly APIs require a
// real, working address as their politeness policy. Unpaywall hard-rejects any
// call without one (HTTP 422), and NCBI raises your rate limit when it can tell
// who you are. Stored on-device beside the keys and sent as a request header, so
// no address is ever baked into the app.
// ---------------------------------------------------------------------------
const EMAIL_KEY = "ee:contact-email";

export function getContactEmail(): string {
  try { return localStorage.getItem(EMAIL_KEY) || ""; } catch { return ""; }
}
export function setContactEmail(email: string): void {
  try {
    const e = email.trim();
    if (e) localStorage.setItem(EMAIL_KEY, e);
    else localStorage.removeItem(EMAIL_KEY);
  } catch { /* ignore */ }
  emit();
}
export function isValidContactEmail(email: string): boolean {
  const e = email.trim();
  if (!/^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$/.test(e)) return false;
  // A placeholder domain is rejected by the very APIs this unlocks.
  return !/(example\.(com|org)|localhost|test\.com)$/i.test(e.split("@")[1] || "");
}

// Headers to attach to every backend request for any data-source key that is set.
export function dbKeyHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  for (const s of DB_SOURCES) {
    const k = getDbKey(s);
    if (k) h[HEADER[s]] = k;
  }
  const email = getContactEmail();
  if (email && isValidContactEmail(email)) h["X-User-Contact-Email"] = email;
  return h;
}
