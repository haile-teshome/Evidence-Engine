// Advisory release check. The comparison itself is done by the backend (see
// Backend/updates.py) so it works however the app was started, and so the
// once-a-day cache is shared rather than per browser.
//
// Nothing here installs anything: the banner links to the release page.
import { apiFetch } from "./backendClient";

export type UpdateInfo = {
  current: string;
  latest: string;
  newer: boolean;
  url: string;
  name: string;
  notes: string;
  published_at: string;
  checked_at: number;
  enabled: boolean;
  error: string;
};

/** The check, or null if it could not be made. A failure shows no banner. */
export async function getUpdate(force = false): Promise<UpdateInfo | null> {
  try {
    const out = await apiFetch(`/update${force ? "?force=true" : ""}`);
    return out && typeof out === "object" ? (out as UpdateInfo) : null;
  } catch {
    // Offline, or the backend is still starting. Silent on purpose: an update
    // notice is the least important thing on screen.
    return null;
  }
}

const DISMISS_KEY = "ee.update.dismissed";

/** Dismissal is remembered per version, so the next release notifies again. */
export function isDismissed(version: string): boolean {
  if (!version) return false;
  try {
    return localStorage.getItem(DISMISS_KEY) === version;
  } catch {
    return false;
  }
}

export function dismiss(version: string): void {
  try {
    localStorage.setItem(DISMISS_KEY, version);
  } catch {
    /* private window, blocked storage: the banner just comes back */
  }
}

/** Should the banner show? Keeps the rule in one testable place. */
export function shouldNotify(info: UpdateInfo | null): boolean {
  if (!info || !info.enabled || !info.newer) return false;
  if (!info.latest || !info.url) return false;    // nothing to point the user at
  return !isDismissed(info.latest);
}
