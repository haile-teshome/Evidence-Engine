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
// check changes; this client stays the same.

import { keyHeaders } from "./apiClient";

export const REVIEWER_ID_KEY = "ee_reviewer_id";
export const AUTH_TOKEN_KEY = "ee_auth_token";

// The built-in single-user profile used before anyone has signed in.
export const DEFAULT_REVIEWER_ID = "local";

export function getReviewerId(): string {
  try {
    return localStorage.getItem(REVIEWER_ID_KEY) || DEFAULT_REVIEWER_ID;
  } catch {
    return DEFAULT_REVIEWER_ID;
  }
}

// Auth session token (from login/signup). Stored locally; sent as a bearer token
// so the backend resolves the acting account server-side.
export function getAuthToken(): string {
  try { return localStorage.getItem(AUTH_TOKEN_KEY) || ""; } catch { return ""; }
}

export function setAuthSession(token: string, reviewerId: string): void {
  try {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(REVIEWER_ID_KEY, reviewerId);
  } catch { /* storage unavailable */ }
}

export function clearAuthSession(): void {
  try {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(REVIEWER_ID_KEY);
  } catch { /* ignore */ }
}

export async function apiFetch(path: string, opts: RequestInit = {}) {
  const token = getAuthToken();
  const res = await fetch(`/api${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      "X-Reviewer-Id": getReviewerId(),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...keyHeaders(),   // per-request cloud-model keys (local-only, see apiClient)
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
