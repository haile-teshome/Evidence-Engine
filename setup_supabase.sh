#!/usr/bin/env bash
# One-shot Supabase wiring for the Evidence Engine project.
#
# Prerequisites (you'll be prompted to install if missing):
#   - Supabase CLI: `brew install supabase/tap/supabase`
#
# What this does:
#   1. Logs you into Supabase (interactive, one-time).
#   2. Links this repo to your `epuqcrytzfqzoxamardf` project.
#   3. Applies the kv_store migration so the table exists.
#   4. Deploys the `server` edge function (with all the project routes).
#   5. Asks you to paste the service-role key, then sets it as a function
#      secret so the function can do admin operations server-side.
#
# Run from the repo root: `bash setup_supabase.sh`

set -e

PROJECT_REF="epuqcrytzfqzoxamardf"

if ! command -v supabase >/dev/null; then
  echo "Supabase CLI not found. Install with:"
  echo "  brew install supabase/tap/supabase"
  echo "Then re-run this script."
  exit 1
fi

echo "==> Logging into Supabase (a browser window will open)…"
supabase login

echo "==> Linking this repo to project $PROJECT_REF…"
supabase link --project-ref "$PROJECT_REF"

echo "==> Applying database migration (creates kv_store_7e4eb0f2)…"
supabase db push

echo "==> Deploying the 'server' edge function…"
supabase functions deploy server --no-verify-jwt

echo ""
echo "==> Final step: set the service-role key as a function secret."
echo "    Open https://supabase.com/dashboard/project/$PROJECT_REF/settings/api"
echo "    and copy the 'service_role' key (NOT the anon key)."
echo ""
read -p "Paste service_role key here: " SERVICE_ROLE_KEY
if [ -z "$SERVICE_ROLE_KEY" ]; then
  echo "No key entered. Skipping. You can run this later:"
  echo "  supabase secrets set SUPABASE_SERVICE_ROLE_KEY=<key>"
else
  supabase secrets set SUPABASE_SERVICE_ROLE_KEY="$SERVICE_ROLE_KEY"
fi

echo ""
echo "Done. Now:"
echo "  1. Copy the *anon public* key from the same API settings page."
echo "  2. Paste it into .env.local in place of __PASTE_ANON_PUBLIC_KEY_HERE__."
echo "  3. Restart the Vite dev server (npm run dev)."
echo "  4. Sign up (top-right), then visit Projects in the sidebar."
