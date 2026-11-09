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

/**
 * How long "Later" lasts.
 *
 * Dismissal used to be forever for that version, so a reviewer who was busy
 * the one time they were asked never heard about the release again. A notice
 * nobody can be reminded of is the same as no notice. Three days is long
 * enough not to nag and short enough that an update actually lands.
 */
export const SNOOZE_MS = 3 * 24 * 60 * 60 * 1000;

type Snooze = { version: string; until: number };

function readSnooze(): Snooze | null {
  try {
    const raw = localStorage.getItem(DISMISS_KEY);
    if (!raw) return null;
    // A plain version string is what older builds wrote. Treat it as a snooze
    // that has already run out rather than silencing the banner forever.
    if (!raw.startsWith("{")) return { version: raw, until: 0 };
    const d = JSON.parse(raw);
    return typeof d?.version === "string" && typeof d?.until === "number" ? d : null;
  } catch {
    return null;
  }
}

/** True while this version is snoozed. It stops being true on its own. */
export function isDismissed(version: string, now = Date.now()): boolean {
  if (!version) return false;
  const snooze = readSnooze();
  return !!snooze && snooze.version === version && snooze.until > now;
}

export function dismiss(version: string, now = Date.now()): void {
  try {
    localStorage.setItem(DISMISS_KEY, JSON.stringify({ version, until: now + SNOOZE_MS }));
  } catch {
    /* private window, blocked storage: the banner just comes back */
  }
}

/** Where an update is up to. "idle" until one is asked for. */
export type UpdateProgress = {
  state: "idle" | "downloading" | "verifying" | "staging" | "ready" | "failed";
  version?: string;
  received?: number;
  total?: number;
  error?: string;
  staged?: { version: string; path: string } | null;
};

/** Start downloading a release. The app is not replaced until it restarts. */
export async function startInstall(version: string): Promise<UpdateProgress> {
  return apiFetch("/update/install", { method: "POST", body: JSON.stringify({ version }) });
}

export async function installProgress(): Promise<UpdateProgress | null> {
  try { return await apiFetch("/update/status"); } catch { return null; }
}

/**
 * Quit and come back as the new version.
 *
 * This one does NOT go to the backend: the backend cannot replace the app it
 * is part of. It goes to the launcher, which serves this page, and which can
 * hand the swap to a script that outlives it.
 */
export async function applyAndRestart(): Promise<boolean> {
  try {
    const res = await fetch("/__ee/apply-update", { method: "POST" });
    return res.ok && !!(await res.json())?.ok;
  } catch {
    return false;
  }
}

/** Should the banner show? Keeps the rule in one testable place. */
export function shouldNotify(info: UpdateInfo | null): boolean {
  if (!info || !info.enabled || !info.newer) return false;
  if (!info.latest || !info.url) return false;    // nothing to point the user at
  return !isDismissed(info.latest);
}
