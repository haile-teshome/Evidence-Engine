const _projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID as string | undefined;
const _publicAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

// Supabase is optional. When unset, the app renders without auth / session
// persistence (everything else still works — the screening flow uses the local
// FastAPI Backend, not Supabase).
export const supabaseConfigured: boolean = Boolean(_projectId && _publicAnonKey);

if (!supabaseConfigured) {
  // eslint-disable-next-line no-console
  console.warn(
    "[supabase] VITE_SUPABASE_PROJECT_ID / VITE_SUPABASE_ANON_KEY not set — " +
      "auth and session persistence are disabled. To enable, copy .env.example to " +
      ".env.local and fill in your project values."
  );
}

export const projectId = _projectId ?? "";
export const publicAnonKey = _publicAnonKey ?? "";
