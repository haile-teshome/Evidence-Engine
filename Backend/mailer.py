# ============================================================================
# FILE: mailer.py
# Outbound email for collaboration invites.
# ============================================================================
"""One small, dependency-free mail transport.

Why it exists: project invites used to be links the lead had to copy and send by
hand. Giving a reviewer's address should be enough. Nothing else in the backend
sends mail, so this module is the single place that does.

Transports, selected by EE_MAIL_TRANSPORT (default "auto"):
  smtp     any SMTP server (institutional Exchange, Gmail app password, Fastmail,
           an SMTP relay such as Resend/SendGrid/Postmark SMTP). Standard library.
  resend   the Resend HTTPS API, for hosts that block outbound SMTP ports.
  console  print the message to the backend log. Development only: nothing is sent.
  off      refuse to send, and say so.
  auto     smtp if a host is set, else resend if a key is set, else unconfigured.

Configuration (backend environment, never the Vite .env):
  EE_SMTP_HOST, EE_SMTP_PORT (587), EE_SMTP_USER, EE_SMTP_PASSWORD
  EE_SMTP_STARTTLS (1), EE_SMTP_SSL (auto-on for port 465), EE_SMTP_TIMEOUT (15)
  EE_RESEND_API_KEY
  EE_MAIL_FROM        the From header, e.g. "Evidence Engine <review@example.edu>".
                      Defaults to EE_SMTP_USER when that looks like an address.
  EE_MAIL_REPLY_TO    optional Reply-To, usually the inviting author's address.

Every function here is non-raising: a send that fails returns a reason. Creating an
invite must still succeed when mail is misconfigured, because the copyable link is
always a valid fallback. Credentials are never logged or returned.
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from typing import Optional

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RESEND_ENDPOINT = "https://api.resend.com/emails"


def valid_address(value: Optional[str]) -> bool:
    return bool(value) and bool(EMAIL_RE.match(value.strip()))


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _flag(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _from_header() -> str:
    explicit = _env("EE_MAIL_FROM")
    if explicit:
        return explicit
    user = _env("EE_SMTP_USER")
    if valid_address(user):
        return formataddr(("Evidence Engine", user))
    return ""


def transport_name() -> str:
    """Which transport a send would use right now."""
    choice = _env("EE_MAIL_TRANSPORT", "auto").lower()
    if choice in ("smtp", "resend", "console", "off"):
        return choice
    if _env("EE_SMTP_HOST"):
        return "smtp"
    if _env("EE_RESEND_API_KEY"):
        return "resend"
    return "unconfigured"


def status() -> dict:
    """Non-secret description of the mail setup, for the UI to decide whether to
    offer an address field. Shows only the From address, never credentials."""
    t = transport_name()
    sender = parseaddr(_from_header())[1]
    missing = []
    if t == "smtp" and not _env("EE_SMTP_HOST"):
        missing.append("EE_SMTP_HOST")
    if t == "resend" and not _env("EE_RESEND_API_KEY"):
        missing.append("EE_RESEND_API_KEY")
    if t in ("smtp", "resend") and not sender:
        missing.append("EE_MAIL_FROM")
    return {
        "configured": t in ("smtp", "resend", "console") and not missing,
        "transport": t,
        "from": sender or None,
        "missing": missing,
    }


def _result(sent: bool, transport: str, error: Optional[str] = None, message_id: Optional[str] = None) -> dict:
    return {"sent": sent, "transport": transport, "error": error, "message_id": message_id}


def _send_smtp(msg: EmailMessage) -> dict:
    host = _env("EE_SMTP_HOST")
    if not host:
        return _result(False, "smtp", "EE_SMTP_HOST is not set")
    port = int(_env("EE_SMTP_PORT", "587") or 587)
    user, password = _env("EE_SMTP_USER"), os.getenv("EE_SMTP_PASSWORD") or ""
    timeout = float(_env("EE_SMTP_TIMEOUT", "15") or 15)
    use_ssl = _flag("EE_SMTP_SSL", port == 465)
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=timeout)
        with server:
            server.ehlo()
            if not use_ssl and _flag("EE_SMTP_STARTTLS", True):
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return _result(True, "smtp", message_id=msg.get("Message-ID"))
    except Exception as exc:                      # deliberately broad: see module docstring
        return _result(False, "smtp", f"{type(exc).__name__}: {exc}"[:300])


def _send_resend(msg: EmailMessage, text: str, html: Optional[str]) -> dict:
    key = _env("EE_RESEND_API_KEY")
    if not key:
        return _result(False, "resend", "EE_RESEND_API_KEY is not set")
    payload = {"from": msg["From"], "to": [msg["To"]], "subject": msg["Subject"], "text": text}
    if html:
        payload["html"] = html
    if msg.get("Reply-To"):
        payload["reply_to"] = msg["Reply-To"]
    req = urllib.request.Request(
        RESEND_ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(_env("EE_SMTP_TIMEOUT", "15") or 15)) as r:
            body = json.loads(r.read().decode() or "{}")
        return _result(True, "resend", message_id=body.get("id"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:200]
        except Exception:
            pass
        return _result(False, "resend", f"HTTP {exc.code}: {detail}".strip()[:300])
    except Exception as exc:
        return _result(False, "resend", f"{type(exc).__name__}: {exc}"[:300])


def format_sender(name: Optional[str], address: Optional[str]) -> str:
    """"Name <address>" when both are given, the bare address otherwise, "" if unusable."""
    addr = (address or "").strip()
    if not valid_address(addr):
        return ""
    return formataddr(((name or "").strip(), addr)) if (name or "").strip() else addr


def send_email(to: str, subject: str, text: str, html: Optional[str] = None,
               reply_to: Optional[str] = None, sender: Optional[str] = None) -> dict:
    """Send one message. `sender` overrides EE_MAIL_FROM, which is how a user-chosen
    sender identity reaches the wire. Providers commonly refuse a From they do not own:
    that refusal is returned verbatim rather than hidden, and Reply-To still carries the
    author's address in that case. Returns {sent, transport, error, message_id}; never raises."""
    if not valid_address(to):
        return _result(False, transport_name(), "Recipient is not a valid email address")
    t = transport_name()
    if t == "off":
        return _result(False, "off", "Email sending is disabled (EE_MAIL_TRANSPORT=off)")
    if t == "unconfigured":
        return _result(False, "unconfigured",
                       "No mail transport configured. Set EE_SMTP_HOST (and EE_SMTP_USER / "
                       "EE_SMTP_PASSWORD / EE_MAIL_FROM), or EE_RESEND_API_KEY.")
    sender = (sender or "").strip() or _from_header()
    if not sender:
        return _result(False, t, "No sender address. Set one in the Projects tab, or EE_MAIL_FROM.")
    if not valid_address(parseaddr(sender)[1]):
        return _result(False, t, f"Sender '{sender}' is not a valid email address")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to.strip()
    msg["Subject"] = subject
    rt = reply_to or _env("EE_MAIL_REPLY_TO")
    if valid_address(parseaddr(rt)[1] if rt else None):
        msg["Reply-To"] = rt
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    if t == "console":
        print(f"[mailer:console] To: {msg['To']}\n[mailer:console] Subject: {subject}\n{text}", flush=True)
        return _result(True, "console")
    if t == "resend":
        return _send_resend(msg, text, html)
    return _send_smtp(msg)


def invite_message(project_name: str, role: str, url: str, inviter: Optional[str] = None,
                   expires_at: Optional[str] = None, note: Optional[str] = None) -> tuple[str, str, str]:
    """Subject, plain text and HTML for one project invite. The screening corpus is
    never included: the mail carries only who invited whom, to what, and the link."""
    # Without a signed-in name there is no credible "who", and a placeholder like
    # "A colleague invited you" reads like spam. Address the reader directly instead.
    who = (inviter or "").strip() or None
    subject = (f"{who} invited you to screen \"{project_name}\" in Evidence Engine" if who
               else f"Invitation to screen \"{project_name}\" in Evidence Engine")
    expiry_line = f"\nThis link expires on {expires_at[:10]}." if expires_at else ""
    note_line = (f"\n\nFrom {who}:\n{note.strip()}\n" if who else f"\n\n{note.strip()}\n") if (note or "").strip() else ""
    opening = (f"{who} has invited you" if who else "You have been invited")
    text = (
        f"{opening} to join the systematic review \"{project_name}\" "
        f"as {role}.{note_line}\n\n"
        f"Open this link to accept:\n{url}\n\n"
        f"The link works once and is personal to you.{expiry_line}\n\n"
        f"Accepting adds you to the project, where you will see the articles to screen "
        f"and record your own decisions. In blinded projects you cannot see other "
        f"reviewers' decisions until you have made your own.\n"
    )
    esc = lambda v: (v or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (
        f"<p>{esc(opening)} to join the systematic review "
        f"<strong>{esc(project_name)}</strong> as <strong>{esc(role)}</strong>.</p>"
        + (f"<blockquote>{esc(note)}</blockquote>" if (note or "").strip() else "")
        + f"<p><a href=\"{esc(url)}\">Accept the invitation</a></p>"
        f"<p style=\"color:#666;font-size:13px\">The link works once and is personal to you."
        + (f" It expires on {esc(expires_at[:10])}." if expires_at else "")
        + "</p><p style=\"color:#666;font-size:13px\">Accepting adds you to the project, where you "
        "will see the articles to screen and record your own decisions. In blinded projects you "
        "cannot see other reviewers' decisions until you have made your own.</p>"
    )
    return subject, text, html
