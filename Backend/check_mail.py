#!/usr/bin/env python3
"""Verify the invitation mail setup, and optionally send one real test message.

    ./.venv/bin/python check_mail.py                  # just report the configuration
    ./.venv/bin/python check_mail.py you@ucsf.edu     # also send a test invitation there

Reads Backend/.env exactly as the API does, so a pass here means the Projects tab
will send. Prints the transport and any error verbatim; never prints credentials.
"""
import sys
from dotenv import load_dotenv

load_dotenv()
import mailer                                        # noqa: E402  (after load_dotenv)

st = mailer.status()
print(f"transport : {st['transport']}")
print(f"from      : {st['from'] or '(not set)'}")
print(f"configured: {'yes' if st['configured'] else 'no'}")
if st["missing"]:
    print(f"missing   : {', '.join(st['missing'])}")

if not st["configured"]:
    print("\nNothing will be emailed yet. Uncomment one block in Backend/.env, fill it in,")
    print("then restart the backend. The Projects tab still gives you a shareable link.")
    sys.exit(1)

to = sys.argv[1] if len(sys.argv) > 1 else None
if not to:
    print("\nConfiguration looks complete. Pass an address to send a test invitation:")
    print("  ./.venv/bin/python check_mail.py you@ucsf.edu")
    sys.exit(0)

if not mailer.valid_address(to):
    print(f"\n'{to}' is not a valid email address"); sys.exit(2)

subject, text, html = mailer.invite_message(
    "Mail configuration test", "reviewer", "http://localhost:5173/?invite=TEST-TOKEN",
    inviter="Evidence Engine", note="This is a test of the invitation mailer. No project was created.")
res = mailer.send_email(to, subject, text, html)
print(f"\nsent      : {res['sent']}")
print(f"transport : {res['transport']}")
if res.get("message_id"):
    print(f"message id: {res['message_id']}")
if res["error"]:
    print(f"error     : {res['error']}")
    print("\nCommon causes: an app password is required rather than the account password;")
    print("the provider disables SMTP AUTH (Microsoft 365 does by default); or EE_MAIL_FROM")
    print("is not an address the account is allowed to send as.")
sys.exit(0 if res["sent"] else 3)
