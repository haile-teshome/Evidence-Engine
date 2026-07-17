// The launcher opens the app window with `?new=1` on a fresh app launch (double-
// clicking the app), so the app can start on a clean Home page / new session
// instead of restoring the most recent one. A normal in-app refresh has no such
// flag, so it still restores the current session and keeps your work in place.
//
// Captured once at module load and then stripped from the URL, so a subsequent
// refresh of the same window doesn't keep forcing a fresh start.
export const FRESH_LAUNCH: boolean = (() => {
  try {
    const url = new URL(window.location.href);
    if (url.searchParams.has("new")) {
      url.searchParams.delete("new");
      window.history.replaceState({}, "", url.toString());
      return true;
    }
  } catch {
    /* SSR / no window — treat as a normal load */
  }
  return false;
})();
