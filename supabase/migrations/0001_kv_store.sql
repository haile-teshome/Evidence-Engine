-- KV table used by the Hono edge function (supabase/functions/server/kv_store.tsx).
-- The function name is hashed into the table suffix so multiple deployments
-- can coexist; for this project the suffix is `7e4eb0f2`.

create table if not exists public.kv_store_7e4eb0f2 (
    key text primary key,
    value jsonb not null
);

-- Speeds up `like 'prefix:%'` lookups used by kv.getByPrefix().
create index if not exists kv_store_7e4eb0f2_prefix
    on public.kv_store_7e4eb0f2 (key text_pattern_ops);

-- The edge function calls Supabase with the service-role key, which bypasses
-- RLS — so no policies are needed here. If you ever expose this table to the
-- anon role directly, add policies first.
alter table public.kv_store_7e4eb0f2 enable row level security;
