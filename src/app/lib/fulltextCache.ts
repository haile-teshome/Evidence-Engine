/** Full-text bodies live in the backend's text cache (~/.evidence-engine/textcache)
 *  and the saved session carries only their keys. Without this a review with a few
 *  hundred acquired papers rewrote 20+ MB of article text into IndexedDB and SQLite
 *  on every autosave, which is what made each tab switch stall. These two calls
 *  move a body across in either direction. */
const BASE: string = (import.meta as any)?.env?.VITE_API_BASE_URL || "/api";

/** Read a cached body. Resolves null when the cache has no entry for the key. */
export async function fetchFullTextBody(key: string, signal?: AbortSignal): Promise<string | null> {
  const res = await fetch(`${BASE}/fulltext/text/${encodeURIComponent(key)}`, { signal });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Full-text cache read failed (${res.status})`);
  const j = await res.json();
  return typeof j?.text === "string" ? j.text : null;
}

/** Store a body the client extracted itself (an uploaded PDF). Returns the key the
 *  backend filed it under, which is what the session should keep. */
export async function putFullTextBody(key: string, text: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(`${BASE}/fulltext/text/${encodeURIComponent(key)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }), signal,
  });
  if (!res.ok) throw new Error(`Full-text cache write failed (${res.status})`);
  const j = await res.json();
  return typeof j?.key === "string" && j.key ? j.key : key;
}
