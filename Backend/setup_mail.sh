#!/usr/bin/env bash
# Configure invitation email in one command, then prove it works.
#
#   Gmail / Google Workspace (most convenient: no IT request, works in ~2 minutes)
#     1. https://myaccount.google.com/apppasswords  (needs 2-step verification on)
#     2. bash setup_mail.sh gmail you@gmail.com "xxxx xxxx xxxx xxxx" you@ucsf.edu
#
#   Resend HTTPS API (no SMTP; sender stays a resend.dev address until you add DNS)
#     1. https://resend.com  ->  API Keys  ->  create
#     2. bash setup_mail.sh resend re_xxxxxxxx you@ucsf.edu
#
#   UCSF / Microsoft 365 (only if IT has enabled SMTP AUTH on the account)
#     bash setup_mail.sh m365 haile.teshome@ucsf.edu 'password'
#
#   See the flow without sending anything (prints invitations to the backend log)
#     bash setup_mail.sh console
#
# Writes to Backend/.env (gitignored), restarts the API, then sends one test message
# to the reply-to address so you can confirm delivery. Values are never echoed.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-}"; [ -z "$MODE" ] && { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
PY=./.venv/bin/python
cp .env ".env.bak-$(date +%Y%m%d-%H%M%S)"

set_key() {                       # set_key NAME VALUE : replace or append, one line
  local name="$1" val="$2"
  $PY - "$name" "$val" <<'PYEOF'
import sys, re, pathlib
name, val = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env"); lines = p.read_text().splitlines()
out, done = [], False
for l in lines:
    if re.match(rf"^\s*#?\s*{re.escape(name)}\s*=", l):
        if not done: out.append(f"{name}={val}"); done = True
        continue                                   # drop duplicates and the commented template line
    out.append(l)
if not done: out.append(f"{name}={val}")
p.write_text("\n".join(out) + "\n")
PYEOF
}

case "$MODE" in
  gmail)
    USER="${2:?Gmail address}"; PASS="${3:?16-character app password}"; REPLY="${4:-}"
    set_key EE_MAIL_TRANSPORT smtp; set_key EE_SMTP_HOST smtp.gmail.com; set_key EE_SMTP_PORT 587
    set_key EE_SMTP_USER "$USER"; set_key EE_SMTP_PASSWORD "${PASS// /}"   # Google prints it in groups of four
    set_key EE_MAIL_FROM "Evidence Engine <$USER>"
    [ -n "$REPLY" ] && set_key EE_MAIL_REPLY_TO "$REPLY"
    TEST_TO="${REPLY:-$USER}" ;;
  resend)
    KEY="${2:?Resend API key}"; REPLY="${3:-}"
    set_key EE_MAIL_TRANSPORT resend; set_key EE_RESEND_API_KEY "$KEY"
    set_key EE_MAIL_FROM "Evidence Engine <onboarding@resend.dev>"
    [ -n "$REPLY" ] && set_key EE_MAIL_REPLY_TO "$REPLY"
    TEST_TO="${REPLY:?Pass the address to send the test to}" ;;
  m365)
    USER="${2:?UCSF address}"; PASS="${3:?password or app password}"
    set_key EE_MAIL_TRANSPORT smtp; set_key EE_SMTP_HOST smtp.office365.com; set_key EE_SMTP_PORT 587
    set_key EE_SMTP_USER "$USER"; set_key EE_SMTP_PASSWORD "$PASS"
    set_key EE_MAIL_FROM "$USER"; TEST_TO="$USER" ;;
  console)
    set_key EE_MAIL_TRANSPORT console; set_key EE_MAIL_FROM "Evidence Engine <review@localhost>"
    TEST_TO="console@example.edu" ;;
  *) echo "Unknown mode '$MODE'. Use gmail | resend | m365 | console."; exit 1 ;;
esac
echo "Backend/.env updated (previous copy kept as .env.bak-*)."

echo "Restarting the API…"
pkill -f "uvicorn api:app" 2>/dev/null || true
sleep 2
nohup $PY -m uvicorn api:app --host 127.0.0.1 --port 8000 > /tmp/ee_api.log 2>&1 &
for i in $(seq 1 20); do curl -fs http://localhost:8000/api/mail/status >/dev/null 2>&1 && break; sleep 1; done
curl -s http://localhost:8000/api/mail/status; echo

echo "Sending a test invitation to $TEST_TO …"
$PY check_mail.py "$TEST_TO" || true
echo
echo "If that arrived, the Projects tab will send invitations. If it did not, the error above"
echo "says why; the invitation link always remains shareable in the meantime."
