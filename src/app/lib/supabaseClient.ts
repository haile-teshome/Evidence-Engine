import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { projectId, publicAnonKey, supabaseConfigured } from "../../../utils/supabase/info";

// When Supabase isn't configured we still construct a client (with placeholder
// URL/key) so consumers can import `supabase` safely — but we never actually
// call its network methods. `supabaseConfigured` is the runtime gate consumers
// must respect (see auth.tsx).
export const supabase: SupabaseClient = createClient(
  supabaseConfigured ? `https://${projectId}.supabase.co` : "https://placeholder.supabase.co",
  supabaseConfigured ? publicAnonKey : "placeholder-anon-key",
);

export const SERVER_BASE = supabaseConfigured
  ? `https://${projectId}.supabase.co/functions/v1/make-server-7e4eb0f2`
  : "";

export { supabaseConfigured };

export async function apiFetch(path: string, opts: RequestInit = {}) {
  if (!supabaseConfigured) {
    throw new Error(
      "Supabase is not configured (sessions / cloud features disabled). " +
        "Set VITE_SUPABASE_PROJECT_ID and VITE_SUPABASE_ANON_KEY in .env.local to enable."
    );
  }
  const { data: { session } } = await supabase.auth.getSession();
  const token = session?.access_token || publicAnonKey;
  const res = await fetch(`${SERVER_BASE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(opts.headers || {}),
    },
  });
  const text = await res.text();
  let json: any = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* non-json */ }
  if (!res.ok) {
    const msg = json?.error || text || `Request failed (${res.status})`;
    console.error(`API ${path} failed: ${msg}`);
    throw new Error(msg);
  }
  return json;
}
