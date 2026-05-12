const _projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID as string | undefined;
const _publicAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

if (!_projectId || !_publicAnonKey) {
  throw new Error(
    "Supabase configuration missing. Copy .env.example to .env.local and set VITE_SUPABASE_PROJECT_ID and VITE_SUPABASE_ANON_KEY.",
  );
}

export const projectId = _projectId;
export const publicAnonKey = _publicAnonKey;
