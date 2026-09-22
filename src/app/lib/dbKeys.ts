// ---------------------------------------------------------------------------
// Data-source (database) API keys: CORE, Semantic Scholar, and Scopus.
//
// These are separate from the LLM keystore: they are lower-sensitivity than LLM
// billing keys, and, crucially, the backend reads them from x-db-*-key request
// headers via get_cred() in EVERY mode (there is no server-side keychain path for
// them). So unlike LLM keys, they must ship as headers regardless of the LLM
// storage mode. They live here, on-device, and are attached to each request.
// ---------------------------------------------------------------------------

export type DbSource =
  | "core" | "semantic_scholar" | "scopus" | "ncbi" | "springer" | "ieee" | "wos"
  // OpenAlex became usage-based in February 2026. Without a key a caller
  // shares a free budget of roughly a hundred requests a day with every other
  // machine on the same network address, so on a campus network it is usually
  // spent before you begin. A key is free and has its own budget.
  | "openalex"
  // Publisher text-and-data-mining credentials. Used only by full-text
  // acquisition, to reach SUBSCRIPTION content through the interface each
  // publisher provides for text mining. Entitlement comes from the reader's
  // institution; with no credential stored, that tier makes no request at all.
  | "wiley" | "elsevier_insttoken"
  // LibKey (Third Iron): a DOI resolves to the copy THIS library is entitled
  // to. Needs the library's numeric id as well as the token.
  | "libkey_id" | "libkey";
export const DB_SOURCES: DbSource[] = [
  "openalex", "core", "semantic_scholar", "scopus", "ncbi", "springer", "ieee", "wos",
  "wiley", "elsevier_insttoken", "libkey_id", "libkey",
];
/** The subset that unlocks subscription full text rather than search. */
export const TDM_SOURCES: DbSource[] = ["scopus", "springer", "wiley", "elsevier_insttoken"];

const PREFIX = "ee:db-key:";
const HEADER: Record<DbSource, string> = {
  openalex: "X-DB-OpenAlex-Key",
  core: "X-DB-Core-Key",
  semantic_scholar: "X-DB-Semantic-Scholar-Key",
  scopus: "X-DB-Scopus-Key",
  ncbi: "X-DB-NCBI-Key",
  springer: "X-DB-Springer-Key",
  ieee: "X-DB-IEEE-Key",
  wos: "X-DB-WOS-Key",
  wiley: "X-TDM-Wiley-Token",
  elsevier_insttoken: "X-TDM-Elsevier-Insttoken",
  libkey_id: "X-LibKey-Id",
  libkey: "X-LibKey-Token",
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
// ---------------------------------------------------------------------------
// Library OpenURL resolver.
//
// Not a key either: the base URL of the institution's link resolver. Given a
// DOI it returns the copy this library is entitled to, which for a paywalled
// record is the difference between a dead end and one click. Stored on-device
// beside the keys; the library supplies the base URL.
// ---------------------------------------------------------------------------
const RESOLVER_KEY = "ee:openurl-base";

export function getResolverBase(): string {
  try { return localStorage.getItem(RESOLVER_KEY) || ""; } catch { return ""; }
}
export function setResolverBase(url: string): void {
  try {
    const u = url.trim().replace(/[?&]$/, "");
    if (u) localStorage.setItem(RESOLVER_KEY, u);
    else localStorage.removeItem(RESOLVER_KEY);
  } catch { /* ignore */ }
  emit();
}

/** An OpenURL 1.0 (KEV) link to this library's entitled copy.
 *
 *  Returns "" when no resolver is configured, so every caller renders nothing
 *  rather than a broken link. A DOI alone resolves reliably; title and date are
 *  added when present because resolvers fall back to citation matching for
 *  records the DOI lookup misses. */
export function openUrlFor(item: { doi?: string; title?: string; year?: string | number; pmid?: string }): string {
  const base = getResolverBase();
  if (!base) return "";
  const id = item.doi ? `info:doi/${item.doi}` : item.pmid ? `info:pmid/${item.pmid}` : "";
  if (!id && !item.title) return "";
  const p = new URLSearchParams({ "ctx_ver": "Z39.88-2004", "rfr_id": "info:sid/evidence-engine" });
  if (id) p.set("rft_id", id);
  if (item.title) p.set("rft.atitle", item.title);
  if (item.year) p.set("rft.date", String(item.year));
  return `${base}${base.includes("?") ? "&" : "?"}${p.toString()}`;
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
