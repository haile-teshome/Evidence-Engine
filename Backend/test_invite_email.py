"""Emailed project invites: validation, delivery, binding, and the unchanged link path.

The console transport is used so nothing leaves the machine; one test replaces the SMTP
sender to prove the real path is reached with the right recipient and that a transport
failure never costs the lead their invite link.
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

import mailer
from api import app

client = TestClient(app)


def _hdr(uid):
    return {"x-reviewer-id": uid}


@pytest.fixture
def console_mail(monkeypatch):
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "console")
    monkeypatch.setenv("EE_MAIL_FROM", "Evidence Engine <review@example.edu>")


@pytest.fixture
def project():
    r = client.post("/api/projects", json={"name": "Invite email test", "screening_mode": "dual_blinded"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200, r.text
    return r.json()["project"]["id"]


def test_mail_status_reports_transport(console_mail):
    s = client.get("/api/mail/status").json()
    assert s["configured"] is True and s["transport"] == "console"
    assert s["from"] == "review@example.edu"
    assert "password" not in str(s).lower()          # never leak credentials


def test_invite_without_email_is_unchanged(project):
    r = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"}, headers=_hdr("lead_mail"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["invite"]["email"] is None
    assert body["email"]["sent"] is False             # nothing asked for, nothing sent
    assert body["url"].endswith(body["invite"]["token"])


def test_invalid_address_is_rejected(project):
    r = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "not-an-address"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 400
    assert "valid email" in r.json()["detail"].lower()


def test_valid_address_is_emailed_and_recorded(project, console_mail):
    r = client.post(f"/api/projects/{project}/invites",
                    json={"role": "adjudicator", "email": "Reviewer@Example.edu",
                          "note": "120 abstracts to start", "base_url": "https://review.example.edu"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"]["sent"] is True and body["email"]["transport"] == "console"
    assert body["invite"]["email"] == "reviewer@example.edu"     # normalised
    assert body["invite"]["emailed_at"]
    assert body["url"].startswith("https://review.example.edu/?invite=")


def test_transport_failure_still_returns_a_usable_link(project, monkeypatch):
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "smtp")
    monkeypatch.setenv("EE_MAIL_FROM", "review@example.edu")
    monkeypatch.setenv("EE_SMTP_HOST", "smtp.invalid.example")
    monkeypatch.setattr(mailer, "_send_smtp", lambda msg: mailer._result(False, "smtp", "Connection refused"))
    r = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "x@example.edu"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200                       # the invite is not lost to a mail outage
    body = r.json()
    assert body["email"]["sent"] is False and "Connection refused" in body["email"]["error"]
    assert body["url"]


def test_real_smtp_path_receives_the_address(project, monkeypatch):
    seen = {}
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "smtp")
    monkeypatch.setenv("EE_MAIL_FROM", "Evidence Engine <review@example.edu>")
    monkeypatch.setenv("EE_SMTP_HOST", "smtp.example.edu")

    class FakeServer:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self, context=None): seen["tls"] = True
        def login(self, u, p): seen["login"] = u
        def send_message(self, msg):
            seen["to"] = msg["To"]; seen["from"] = msg["From"]; seen["subject"] = msg["Subject"]
            seen["body"] = msg.get_body(preferencelist=("plain",)).get_content()

    monkeypatch.setenv("EE_SMTP_USER", "review@example.edu")
    monkeypatch.setenv("EE_SMTP_PASSWORD", "app-password")
    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda host, port, timeout=None: FakeServer())
    r = client.post(f"/api/projects/{project}/invites",
                    json={"role": "viewer", "email": "colleague@example.org", "base_url": "https://review.example.edu"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200 and r.json()["email"]["sent"] is True
    assert seen["to"] == "colleague@example.org"
    assert seen["tls"] is True and seen["login"] == "review@example.edu"
    assert r.json()["invite"]["token"] in seen["body"]        # the link is in the message
    assert "app-password" not in str(r.json())                # credentials never echoed


def test_preview_masks_the_address(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "jane.doe@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    shown = client.get(f"/api/invites/{tok}").json()["invite"]["email"]
    assert shown == "ja***@example.edu"


def test_emailed_invite_binds_to_that_user(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "owner@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    # a fresh account each run: the test database persists between runs
    addr = f"other.{uuid.uuid4().hex[:10]}@example.edu"
    other = client.post("/api/auth/signup", json={"email": addr, "password": "password123", "name": "Other"})
    assert other.status_code == 200, other.text
    tok_other = other.json()["token"]
    r = client.post(f"/api/invites/{tok}/accept", headers={"authorization": f"Bearer {tok_other}"})
    assert r.status_code == 403 and "owner@example.edu" in r.json()["detail"]


def test_unauthenticated_local_user_can_still_accept(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "someone@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    r = client.post(f"/api/invites/{tok}/accept", headers=_hdr("walk_in"))
    assert r.status_code == 200 and r.json()["role"] == "reviewer"


# ---- resending -------------------------------------------------------------

def test_resend_sends_again_with_the_same_token(project, console_mail):
    created = client.post(f"/api/projects/{project}/invites",
                          json={"role": "reviewer", "email": "lost@example.edu"}, headers=_hdr("lead_mail")).json()
    tok = created["invite"]["token"]
    r = client.post(f"/api/projects/{project}/invites/{tok}/resend", json={}, headers=_hdr("lead_mail"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"]["sent"] is True
    assert body["invite"]["token"] == tok          # the shared link keeps working
    assert body["invite"]["resend_count"] == 1
    assert body["invite"]["state"] == "pending"


def test_resend_can_correct_the_address(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites",
                      json={"role": "reviewer", "email": "typo@exampl.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    r = client.post(f"/api/projects/{project}/invites/{tok}/resend",
                    json={"email": "Correct@Example.edu"}, headers=_hdr("lead_mail"))
    assert r.status_code == 200 and r.json()["invite"]["email"] == "correct@example.edu"


def test_resend_rejects_a_bad_address(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    r = client.post(f"/api/projects/{project}/invites/{tok}/resend", json={"email": "nope"}, headers=_hdr("lead_mail"))
    assert r.status_code == 400


def test_resend_needs_an_address_when_the_invite_was_link_only(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    r = client.post(f"/api/projects/{project}/invites/{tok}/resend", json={}, headers=_hdr("lead_mail"))
    assert r.status_code == 400 and "no email address" in r.json()["detail"].lower()


def test_accepted_invite_cannot_be_resent(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "a@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    assert client.post(f"/api/invites/{tok}/accept", headers=_hdr("joiner")).status_code == 200
    r = client.post(f"/api/projects/{project}/invites/{tok}/resend", json={}, headers=_hdr("lead_mail"))
    assert r.status_code == 410 and "already been accepted" in r.json()["detail"]


def test_only_the_lead_may_list_or_resend(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "a@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    client.post(f"/api/invites/{tok}/accept", headers=_hdr("plain_reviewer"))
    assert client.get(f"/api/projects/{project}/invites", headers=_hdr("plain_reviewer")).status_code == 403
    tok2 = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "b@example.edu"},
                       headers=_hdr("lead_mail")).json()["invite"]["token"]
    assert client.post(f"/api/projects/{project}/invites/{tok2}/resend", json={},
                       headers=_hdr("plain_reviewer")).status_code == 403


def test_list_shows_state_per_invite(project, console_mail):
    pending = client.post(f"/api/projects/{project}/invites", json={"role": "viewer", "email": "p@example.edu"},
                          headers=_hdr("lead_mail")).json()["invite"]["token"]
    used = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "u@example.edu"},
                       headers=_hdr("lead_mail")).json()["invite"]["token"]
    client.post(f"/api/invites/{used}/accept", headers=_hdr(f"acceptor_{used[:6]}"))
    rows = client.get(f"/api/projects/{project}/invites", headers=_hdr("lead_mail")).json()["invites"]
    states = {r["token"]: r["state"] for r in rows}
    assert states[pending] == "pending" and states[used] == "accepted"


# ---- the zero-setup path: compose for the author's own mail client -----------

def test_composed_message_needs_no_mail_configuration(project, monkeypatch):
    monkeypatch.delenv("EE_MAIL_TRANSPORT", raising=False)
    monkeypatch.delenv("EE_SMTP_HOST", raising=False)
    monkeypatch.delenv("EE_RESEND_API_KEY", raising=False)
    assert client.get("/api/mail/status").json()["configured"] is False
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    r = client.get(f"/api/projects/{project}/invites/{tok}/message?email=colleague@ucsf.edu",
                   headers=_hdr("lead_mail"))
    assert r.status_code == 200, r.text            # works with nothing configured
    m = r.json()
    assert m["to"] == "colleague@ucsf.edu"
    assert m["mailto"].startswith("mailto:colleague%40ucsf.edu?subject=")
    assert tok in m["body"] and tok in m["url"]
    assert len(m["mailto"]) < 1900                 # within what mail clients accept


def test_composed_message_uses_the_stored_address_when_none_is_given(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "viewer", "email": "stored@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    m = client.get(f"/api/projects/{project}/invites/{tok}/message", headers=_hdr("lead_mail")).json()
    assert m["to"] == "stored@example.edu"


def test_composed_message_is_lead_only(project, console_mail):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "x@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    client.post(f"/api/invites/{tok}/accept", headers=_hdr("nosy_reviewer"))
    tok2 = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                       headers=_hdr("lead_mail")).json()["invite"]["token"]
    assert client.get(f"/api/projects/{project}/invites/{tok2}/message",
                      headers=_hdr("nosy_reviewer")).status_code == 403


def test_composed_message_matches_the_automatic_one(project, console_mail):
    """One wording for both routes: whatever the mailer would send is what the mail app gets."""
    import mailer as m
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "adjudicator", "email": "both@example.edu"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    composed = client.get(f"/api/projects/{project}/invites/{tok}/message", headers=_hdr("lead_mail")).json()
    subject, text, _ = m.invite_message("Invite email test", "adjudicator", composed["url"],
                                        inviter=None, expires_at=None)
    assert composed["subject"] == subject and composed["body"] == text


# ---- editable sender identity ------------------------------------------------

def test_sender_defaults_to_the_environment(console_mail):
    r = client.get("/api/mail/identity", headers=_hdr("sender_user"))
    assert r.status_code == 200
    d = r.json()
    assert d["saved"] is False
    assert d["effective_from"] == "review@example.edu"      # EE_MAIL_FROM


def test_sender_can_be_edited_and_cleared(console_mail):
    saved = client.put("/api/mail/identity", json={"name": "Haile Teshome", "email": "haile.teshome@ucsf.edu"},
                       headers=_hdr("sender_user")).json()
    assert saved["saved"] is True
    assert saved["effective_from"] == "Haile Teshome <haile.teshome@ucsf.edu>"
    cleared = client.put("/api/mail/identity", json={"name": "", "email": ""}, headers=_hdr("sender_user")).json()
    assert cleared["saved"] is False and cleared["effective_from"] == "review@example.edu"


def test_sender_address_is_validated(console_mail):
    r = client.put("/api/mail/identity", json={"name": "X", "email": "not-an-address"}, headers=_hdr("sender_user"))
    assert r.status_code == 400 and "valid email" in r.json()["detail"].lower()


def test_edited_sender_is_used_on_the_wire(project, monkeypatch):
    seen = {}
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "smtp")
    monkeypatch.setenv("EE_MAIL_FROM", "Evidence Engine <default@example.edu>")
    monkeypatch.setenv("EE_SMTP_HOST", "smtp.example.edu")

    class FakeServer:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self, context=None): pass
        def send_message(self, msg):
            seen["from"] = msg["From"]; seen["reply_to"] = msg.get("Reply-To")
            seen["body"] = msg.get_body(preferencelist=("plain",)).get_content()

    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda host, port, timeout=None: FakeServer())
    client.put("/api/mail/identity", json={"name": "Haile Teshome", "email": "haile.teshome@ucsf.edu"},
               headers=_hdr("lead_mail"))
    r = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "rev@example.edu"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200 and r.json()["email"]["sent"] is True
    assert seen["from"] == "Haile Teshome <haile.teshome@ucsf.edu>"     # not the env default
    assert seen["reply_to"] == "haile.teshome@ucsf.edu"
    assert "Haile Teshome has invited you" in seen["body"]              # signs the message too
    client.put("/api/mail/identity", json={"name": "", "email": ""}, headers=_hdr("lead_mail"))


def test_sender_reaches_the_mail_app_path_too(project, console_mail):
    client.put("/api/mail/identity", json={"name": "Haile Teshome", "email": "haile.teshome@ucsf.edu"},
               headers=_hdr("lead_mail"))
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                      headers=_hdr("lead_mail")).json()["invite"]["token"]
    m = client.get(f"/api/projects/{project}/invites/{tok}/message?email=colleague@ucsf.edu",
                   headers=_hdr("lead_mail")).json()
    assert m["sender"] == "haile.teshome@ucsf.edu"
    assert "Haile Teshome invited you" in m["subject"]
    assert "from=haile.teshome%40ucsf.edu" in m["mailto"]
    client.put("/api/mail/identity", json={"name": "", "email": ""}, headers=_hdr("lead_mail"))


def test_a_rejected_sender_reports_the_providers_reason(project, monkeypatch):
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "smtp")
    monkeypatch.setenv("EE_SMTP_HOST", "smtp.example.edu")
    monkeypatch.setattr(mailer, "_send_smtp",
                        lambda msg: mailer._result(False, "smtp", "550 5.7.60 Cannot send as this sender"))
    client.put("/api/mail/identity", json={"name": "Someone", "email": "not.mine@elsewhere.org"},
               headers=_hdr("lead_mail"))
    r = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "rev2@example.edu"},
                    headers=_hdr("lead_mail"))
    assert r.status_code == 200                              # the invitation still exists
    assert "Cannot send as this sender" in r.json()["email"]["error"]
    assert r.json()["url"]                                   # and the link still works
    client.put("/api/mail/identity", json={"name": "", "email": ""}, headers=_hdr("lead_mail"))


# ---- the sender is the login address ----------------------------------------

def _signin(email="sender.login@example.edu", name="Signed In User"):
    addr = email.replace("@", f".{uuid.uuid4().hex[:6]}@")
    r = client.post("/api/auth/signup", json={"email": addr, "password": "password123", "name": name})
    assert r.status_code == 200, r.text
    return {"authorization": f"Bearer {r.json()['token']}"}, addr


def test_signed_in_sender_is_the_login_address(console_mail):
    hdr, addr = _signin()
    d = client.get("/api/mail/identity", headers=hdr).json()
    assert d["login_email"] == addr and d["email"] == addr
    assert d["from_login"] is True and d["signed_in"] is True
    assert addr in d["effective_from"]


def test_login_address_cannot_be_overridden(console_mail):
    hdr, addr = _signin()
    r = client.put("/api/mail/identity", json={"name": "Someone", "email": "spoofed@elsewhere.org"}, headers=hdr)
    assert r.status_code == 400
    assert addr in r.json()["detail"]                       # says which address is used
    assert client.get("/api/mail/identity", headers=hdr).json()["email"] == addr


def test_name_remains_editable_when_signed_in(console_mail):
    hdr, addr = _signin()
    d = client.put("/api/mail/identity", json={"name": "Haile Teshome, UCSF", "email": addr}, headers=hdr).json()
    assert d["name"] == "Haile Teshome, UCSF" and d["email"] == addr


def test_invitation_is_sent_from_the_login_address(project, monkeypatch):
    seen = {}
    monkeypatch.setenv("EE_MAIL_TRANSPORT", "smtp")
    monkeypatch.setenv("EE_MAIL_FROM", "Evidence Engine <default@example.edu>")
    monkeypatch.setenv("EE_SMTP_HOST", "smtp.example.edu")

    class FakeServer:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self, context=None): pass
        def send_message(self, msg): seen["from"] = msg["From"]; seen["reply_to"] = msg.get("Reply-To")

    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda host, port, timeout=None: FakeServer())
    hdr, addr = _signin(name="Lead Person")
    pid = client.post("/api/projects", json={"name": "Login sender test"}, headers=hdr).json()["project"]["id"]
    r = client.post(f"/api/projects/{pid}/invites", json={"role": "reviewer", "email": "rev@example.edu"}, headers=hdr)
    assert r.status_code == 200 and r.json()["email"]["sent"] is True
    assert addr in seen["from"] and seen["from"].startswith("Lead Person")
    assert seen["reply_to"] == addr


def test_unsigned_in_user_may_still_set_an_address(console_mail):
    """Single-machine use has no account, so a typed address is the only option there."""
    d = client.put("/api/mail/identity", json={"name": "Local User", "email": "local.user@example.edu"},
                   headers=_hdr("local")).json()
    assert d["email"] == "local.user@example.edu" and d["from_login"] is False
    client.put("/api/mail/identity", json={"name": "", "email": ""}, headers=_hdr("local"))
