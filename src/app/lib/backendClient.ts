// Local backend client.
//
// Talks to the Evidence Engine FastAPI backend at `/api` (Vite proxies this to
// http://localhost:8000). This replaces the former Supabase edge-function
// transport so the whole app runs locally with no external service.
//
// "Users" are local reviewer profiles rather than authenticated accounts. The
// active profile id is sent in the `X-Reviewer-Id` header so the backend can
// scope sessions and multi-reviewer project data per reviewer. When we later
// host this backend for remote collaboration, only the server-side identity
// check changes — this client stays the same.

export const REVIEWER_ID_KEY = "ee_reviewer_id";

// The built-in single-user profile every install starts with.
export const DEFAULT_REVIEWER_ID = "local";

export function getReviewerId(): string {
  try {
    return localStorage.getItem(REVIEWER_ID_KEY) || DEFAULT_REVIEWER_ID;
  } catch {
    return DEFAULT_REVIEWER_ID;
  }
}

export async function apiFetch(path: string, opts: RequestInit = {}) {
  const res = await fetch(`/api${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      "X-Reviewer-Id": getReviewerId(),
      ...(opts.headers || {}),
    },
  });
  const text = await res.text();
  let json: any = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    /* non-json */
  }
  if (!res.ok) {
    const msg = json?.detail || json?.error || text || `Request failed (${res.status})`;
    console.error(`API ${path} failed: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`);
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return json;
}
