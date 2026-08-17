"""Local persistence layer for Evidence Engine.

Replaces the former Supabase edge function + Postgres KV store with a
zero-install SQLite database served directly by the FastAPI backend. This keeps
the app fully local ("double-click and go"): sessions, projects, and the
multi-reviewer collaboration data all live in a single file at
``~/.evidence-engine/evidence.db`` (override with ``EE_DB_PATH``).

Design notes
------------
* The data model is an exact port of the edge function's KV layout, so the
  frontend contract is unchanged — only the transport (Supabase functions URL
  → local ``/api``) and the identity source (Supabase auth → local reviewer
  profile) differ.
* "Users" are now local *reviewer profiles*: a name (+ optional email) with a
  generated id. There are no passwords or accounts. The frontend sends the
  active profile id in the ``X-Reviewer-Id`` header; ``current_user`` reads it
  and falls back to ``"local"`` (the default single-user profile).
* Because it's your own backend, remote collaboration later is just a matter of
  hosting this same app and swapping ``current_user`` for real auth — every
  route below stays identical.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# SQLite key-value store
# ---------------------------------------------------------------------------
# One table, ``kv(key TEXT PRIMARY KEY, value TEXT)`` where value is JSON. This
# mirrors the edge function's ``kv_store_7e4eb0f2`` table so the port is 1:1.

def _db_path() -> Path:
    override = os.getenv("EE_DB_PATH")
    if override:
        p = Path(override).expanduser()
    else:
        p = Path.home() / ".evidence-engine" / "evidence.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_DB_PATH = _db_path()
_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _init() -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.commit()
        _initialized = True


def kv_get(key: str) -> Any:
    _init()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def kv_set(key: str, value: Any) -> None:
    _init()
    payload = json.dumps(value)
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, payload),
        )
        conn.commit()


def kv_del(key: str) -> None:
    _init()
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        conn.commit()


def kv_get_by_prefix(prefix: str) -> List[Any]:
    _init()
    # Escape LIKE wildcards in the prefix so keys with % or _ match literally.
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT value FROM kv WHERE key LIKE ? ESCAPE '\\'", (escaped + "%",)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):x}_{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Authentication: local accounts (email + salted-hashed password) and bearer
# tokens. Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib, no native dep);
# the plaintext is never stored. A valid token, resolved in middleware, pins the
# acting reviewer so the X-Reviewer-Id header alone can't impersonate an account.
# ---------------------------------------------------------------------------

_PBKDF2_ITERS = 200_000
_TOKEN_TTL_DAYS = 30
_authed_uid: "contextvars.ContextVar[str]" = contextvars.ContextVar("authed_uid", default="")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = (stored or "").split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def _create_token(uid: str) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    kv_set(f"authtoken:{token}", {
        "uid": uid,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=_TOKEN_TTL_DAYS)).isoformat(),
    })
    return token


def _uid_for_token(token: str) -> str:
    if not token:
        return ""
    rec = kv_get(f"authtoken:{token}")
    if not rec:
        return ""
    try:
        if datetime.fromisoformat(rec.get("expires_at", "")) < datetime.now(timezone.utc):
            kv_del(f"authtoken:{token}")
            return ""
    except Exception:
        pass
    return rec.get("uid", "")


def _bearer(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def resolve_auth(authorization: Optional[str]) -> None:
    """Middleware hook: if a valid bearer token is present, pin the acting uid for
    this request so current_user() returns the authenticated account."""
    _authed_uid.set(_uid_for_token(_bearer(authorization)))


def _public_reviewer(rec: dict) -> dict:
    """A reviewer/account record with the password hash stripped, safe to return."""
    return {
        "id": rec.get("id"),
        "name": rec.get("name") or "You",
        "email": rec.get("email", ""),
        "created_at": rec.get("created_at"),
    }


def current_user(x_reviewer_id: Optional[str] = Header(default=None)) -> str:
    """Resolve the acting reviewer. A verified auth token (set in middleware) wins;
    otherwise fall back to the ``X-Reviewer-Id`` header, defaulting to ``"local"``
    so the app still works before anyone has signed in."""
    authed = _authed_uid.get()
    if authed:
        return authed
    return (x_reviewer_id or "").strip() or "local"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api")


# ---- Reviewer profiles (replaces Supabase auth / signup) ------------------

class ReviewerCreate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


def _reviewer_record(uid: str) -> dict:
    return kv_get(f"reviewer:{uid}") or {"id": uid, "name": "You", "email": ""}


@router.get("/reviewers")
def list_reviewers():
    items = [_public_reviewer(r) for r in kv_get_by_prefix("reviewer:")]
    # Ensure the default local profile always exists in the list.
    if not any(r.get("id") == "local" for r in items):
        items.insert(0, {"id": "local", "name": "You", "email": "", "created_at": _now()})
    items.sort(key=lambda r: r.get("created_at", ""))
    return {"reviewers": items}


@router.post("/reviewers")
def create_reviewer(body: ReviewerCreate):
    uid = _new_id("rev")
    rec = {
        "id": uid,
        "name": (body.name or body.email or "Reviewer").strip(),
        "email": (body.email or "").strip(),
        "created_at": _now(),
    }
    kv_set(f"reviewer:{uid}", rec)
    return {"reviewer": _public_reviewer(rec)}


# ---- Auth: local accounts (signup / login / logout / me) ------------------

class SignupBody(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginBody(BaseModel):
    email: str
    password: str


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _claim_local_data(uid: str) -> None:
    """One-time migration: when the FIRST account is created, move the work built
    under the default ``local`` profile to that account so nothing is lost."""
    if kv_get("meta:local_claimed"):
        return
    moved = 0
    for s in kv_get_by_prefix("session:local:"):
        sid = s.get("id")
        if not sid:
            continue
        kv_set(f"session:{uid}:{sid}", s)
        kv_del(f"session:local:{sid}")
        moved += 1
    kv_set("meta:local_claimed", {"uid": uid, "at": _now(), "moved": moved})


@router.post("/auth/signup")
def auth_signup(body: SignupBody):
    email = (body.email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(body.password or "") < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if kv_get(f"emailidx:{email}"):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    uid = _new_id("rev")
    rec = {
        "id": uid,
        "name": (body.name or email.split("@")[0]).strip(),
        "email": email,
        "password_hash": _hash_password(body.password),
        "created_at": _now(),
    }
    kv_set(f"reviewer:{uid}", rec)
    kv_set(f"emailidx:{email}", uid)
    _claim_local_data(uid)
    token = _create_token(uid)
    return {"token": token, "user": _public_reviewer(rec)}


@router.post("/auth/login")
def auth_login(body: LoginBody):
    email = (body.email or "").strip().lower()
    uid = kv_get(f"emailidx:{email}")
    rec = kv_get(f"reviewer:{uid}") if uid else None
    if not rec or not _verify_password(body.password or "", rec.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = _create_token(uid)
    return {"token": token, "user": _public_reviewer(rec)}


@router.post("/auth/logout")
def auth_logout(request: Request):
    token = _bearer(request.headers.get("authorization"))
    if token:
        kv_del(f"authtoken:{token}")
    return {"ok": True}


@router.get("/auth/me")
def auth_me():
    uid = _authed_uid.get()
    rec = kv_get(f"reviewer:{uid}") if uid else None
    if not rec:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": _public_reviewer(rec)}


# ---- Sessions -------------------------------------------------------------

class SessionPut(BaseModel):
    title: Optional[str] = None
    data: Optional[dict] = None


@router.get("/sessions")
def list_sessions(request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    items = kv_get_by_prefix(f"session:{uid}:")
    meta = [
        {
            "id": s.get("id"),
            "title": s.get("title"),
            "updated_at": s.get("updated_at"),
            "created_at": s.get("created_at"),
        }
        for s in items
    ]
    meta.sort(key=lambda m: (m.get("updated_at") or ""), reverse=True)
    return {"sessions": meta}


@router.get("/sessions/{sid}")
def get_session(sid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    session = kv_get(f"session:{uid}:{sid}")
    if not session:
        raise HTTPException(status_code=404, detail="Not found")
    return {"session": session}


@router.put("/sessions/{sid}")
def put_session(sid: str, body: SessionPut, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    existing = kv_get(f"session:{uid}:{sid}")
    now = _now()
    session = {
        "id": sid,
        "title": body.title or (existing or {}).get("title") or "Untitled session",
        "data": body.data if body.data is not None else {},
        "created_at": (existing or {}).get("created_at") or now,
        "updated_at": now,
    }
    kv_set(f"session:{uid}:{sid}", session)
    return {"session": session}


@router.delete("/sessions/{sid}")
def delete_session(sid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    kv_del(f"session:{uid}:{sid}")
    return {"ok": True}


# ---- Projects (multi-reviewer) --------------------------------------------

ROLES = ("lead", "reviewer", "adjudicator", "viewer")


def _get_role(pid: str, uid: str) -> Optional[str]:
    m = kv_get(f"project_member:{pid}:{uid}")
    return m.get("role") if m else None


def _require_role(pid: str, uid: str, allowed) -> str:
    role = _get_role(pid, uid)
    if not role:
        raise HTTPException(status_code=403, detail="Not a project member")
    if role not in allowed:
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot perform this action")
    return role


class ProjectCreate(BaseModel):
    name: Optional[str] = None
    pico: Optional[dict] = None
    inclusion: Optional[list] = None
    exclusion: Optional[list] = None
    screening_mode: Optional[str] = None
    visibility: Optional[str] = None


@router.post("/projects")
def create_project(body: ProjectCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    now = _now()
    pid = _new_id("prj")
    project = {
        "id": pid,
        "name": body.name or "Untitled project",
        "owner_user_id": uid,
        "pico": body.pico or {"population": "", "intervention": "", "comparator": "", "outcome": ""},
        "inclusion": body.inclusion or [],
        "exclusion": body.exclusion or [],
        "screening_mode": body.screening_mode or "dual_blinded",
        "visibility": body.visibility or "invite",
        "locked_at": None,
        "created_at": now,
        "updated_at": now,
    }
    kv_set(f"project:{pid}", project)
    kv_set(f"project_member:{pid}:{uid}", {"project_id": pid, "user_id": uid, "role": "lead", "joined_at": now})
    kv_set(f"user_project:{uid}:{pid}", {"project_id": pid, "joined_at": now, "role": "lead"})
    return {"project": project}


@router.get("/projects")
def list_projects(request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    backlinks = kv_get_by_prefix(f"user_project:{uid}:")
    projects = []
    for bl in backlinks:
        p = kv_get(f"project:{bl['project_id']}")
        if p:
            projects.append({**p, "my_role": bl.get("role")})
    projects.sort(key=lambda p: (p.get("updated_at") or ""), reverse=True)
    return {"projects": projects}


@router.get("/projects/{pid}")
def get_project(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    role = _require_role(pid, uid, ROLES)
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    members = kv_get_by_prefix(f"project_member:{pid}:")
    return {"project": {**project, "my_role": role}, "members": members}


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    pico: Optional[dict] = None
    inclusion: Optional[list] = None
    exclusion: Optional[list] = None
    screening_mode: Optional[str] = None
    visibility: Optional[str] = None
    locked_at: Optional[Any] = "__unset__"


@router.put("/projects/{pid}")
def update_project(pid: str, body: ProjectUpdate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    existing = kv_get(f"project:{pid}")
    if not existing:
        raise HTTPException(status_code=404, detail="Not found")
    locked_provided = body.locked_at != "__unset__"
    if existing.get("locked_at") and not locked_provided:
        patch = {**existing, "name": body.name if body.name is not None else existing.get("name"), "updated_at": _now()}
        kv_set(f"project:{pid}", patch)
        return {"project": patch}
    nxt = {
        **existing,
        "name": body.name if body.name is not None else existing.get("name"),
        "pico": body.pico if body.pico is not None else existing.get("pico"),
        "inclusion": body.inclusion if body.inclusion is not None else existing.get("inclusion"),
        "exclusion": body.exclusion if body.exclusion is not None else existing.get("exclusion"),
        "screening_mode": existing.get("screening_mode") if existing.get("locked_at")
            else (body.screening_mode if body.screening_mode is not None else existing.get("screening_mode")),
        "visibility": body.visibility if body.visibility is not None else existing.get("visibility"),
        "locked_at": body.locked_at if locked_provided else existing.get("locked_at"),
        "updated_at": _now(),
    }
    kv_set(f"project:{pid}", nxt)
    return {"project": nxt}


@router.post("/projects/{pid}/lock")
def lock_project(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    existing = kv_get(f"project:{pid}")
    if not existing:
        raise HTTPException(status_code=404, detail="Not found")
    nxt = {**existing, "locked_at": _now(), "updated_at": _now()}
    kv_set(f"project:{pid}", nxt)
    return {"project": nxt}


# ---- Members + invites ----------------------------------------------------

class RoleUpdate(BaseModel):
    role: str


@router.put("/projects/{pid}/members/{target_uid}/role")
def set_member_role(pid: str, target_uid: str, body: RoleUpdate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    if body.role not in ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")
    m = kv_get(f"project_member:{pid}:{target_uid}")
    if not m:
        raise HTTPException(status_code=404, detail="Member not found")
    nxt = {**m, "role": body.role}
    kv_set(f"project_member:{pid}:{target_uid}", nxt)
    kv_set(f"user_project:{target_uid}:{pid}", {"project_id": pid, "joined_at": m.get("joined_at"), "role": body.role})
    return {"member": nxt}


class InviteCreate(BaseModel):
    role: Optional[str] = None
    expires_at: Optional[str] = None


@router.post("/projects/{pid}/invites")
def create_invite(pid: str, body: InviteCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    token = _new_id("inv").split("_", 1)[1]
    invite = {
        "token": token,
        "project_id": pid,
        "role": body.role or "reviewer",
        "created_by": uid,
        "created_at": _now(),
        "expires_at": body.expires_at or None,
        "used_at": None,
        "used_by": None,
    }
    kv_set(f"invite:{token}", invite)
    return {"invite": invite}


@router.get("/invites/{token}")
def get_invite(token: str):
    invite = kv_get(f"invite:{token}")
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.get("used_at"):
        raise HTTPException(status_code=410, detail="Invite already used")
    if invite.get("expires_at") and invite["expires_at"] < _now():
        raise HTTPException(status_code=410, detail="Invite expired")
    project = kv_get(f"project:{invite['project_id']}")
    return {"invite": invite, "project": {"id": project["id"], "name": project["name"]} if project else None}


@router.post("/invites/{token}/accept")
def accept_invite(token: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    invite = kv_get(f"invite:{token}")
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.get("used_at"):
        raise HTTPException(status_code=410, detail="Invite already used")
    if invite.get("expires_at") and invite["expires_at"] < _now():
        raise HTTPException(status_code=410, detail="Invite expired")
    pid = invite["project_id"]
    existing = kv_get(f"project_member:{pid}:{uid}")
    if existing:
        return {"project_id": pid, "already_member": True, "role": existing.get("role")}
    now = _now()
    kv_set(f"project_member:{pid}:{uid}", {"project_id": pid, "user_id": uid, "role": invite["role"], "joined_at": now})
    kv_set(f"user_project:{uid}:{pid}", {"project_id": pid, "joined_at": now, "role": invite["role"]})
    kv_set(f"invite:{token}", {**invite, "used_at": now, "used_by": uid})
    return {"project_id": pid, "role": invite["role"]}


# ---- Papers ---------------------------------------------------------------

class PapersPut(BaseModel):
    papers: Optional[list] = None


@router.get("/projects/{pid}/papers")
def get_papers(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    role = _require_role(pid, uid, ROLES)
    papers = kv_get(f"project_papers:{pid}") or []
    if role == "reviewer":
        my_assign = kv_get_by_prefix(f"paper_assignment:{pid}:")
        if my_assign:
            my_ids = {a["paper_id"] for a in my_assign if a.get("user_id") == uid}
            return {"papers": [p for p in papers if p.get("paper_id") in my_ids], "assigned": True, "total": len(papers)}
    return {"papers": papers}


@router.put("/projects/{pid}/papers")
def put_papers(pid: str, body: PapersPut, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    papers = body.papers if isinstance(body.papers, list) else []
    kv_set(f"project_papers:{pid}", papers)
    return {"count": len(papers)}


# ---- Assignments ----------------------------------------------------------

class AssignmentCreate(BaseModel):
    strategy: Optional[str] = None
    reviewers_per_paper: Optional[Any] = None
    custom: Optional[list] = None


@router.post("/projects/{pid}/assignments")
def create_assignments(pid: str, body: AssignmentCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    strategy = body.strategy or "full_overlap"
    papers = kv_get(f"project_papers:{pid}") or []
    members = [m for m in kv_get_by_prefix(f"project_member:{pid}:") if m.get("role") in ("reviewer", "lead")]
    if not members:
        raise HTTPException(status_code=400, detail="No reviewers in project")

    for r in kv_get_by_prefix(f"paper_assignment:{pid}:"):
        kv_del(f"paper_assignment:{pid}:{r['paper_id']}:{r['user_id']}")

    now = _now()
    assigned = 0
    if strategy == "full_overlap":
        for p in papers:
            for m in members:
                kv_set(f"paper_assignment:{pid}:{p['paper_id']}:{m['user_id']}",
                       {"project_id": pid, "paper_id": p["paper_id"], "user_id": m["user_id"], "assigned_at": now, "strategy": strategy})
                assigned += 1
    elif strategy == "split":
        n = max(1, min(len(members), int(body.reviewers_per_paper or 2)))
        for i, p in enumerate(papers):
            for k in range(n):
                m = members[(i + k) % len(members)]
                kv_set(f"paper_assignment:{pid}:{p['paper_id']}:{m['user_id']}",
                       {"project_id": pid, "paper_id": p["paper_id"], "user_id": m["user_id"], "assigned_at": now, "strategy": strategy})
                assigned += 1
    elif strategy == "custom" and isinstance(body.custom, list):
        for a in body.custom:
            for target in (a.get("user_ids") or []):
                kv_set(f"paper_assignment:{pid}:{a['paper_id']}:{target}",
                       {"project_id": pid, "paper_id": a["paper_id"], "user_id": target, "assigned_at": now, "strategy": strategy})
                assigned += 1
    else:
        raise HTTPException(status_code=400, detail="Unknown strategy")

    return {"strategy": strategy, "assigned": assigned, "papers": len(papers), "reviewers": len(members)}


@router.get("/projects/{pid}/assignments")
def get_assignments(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    role = _require_role(pid, uid, ROLES)
    allx = kv_get_by_prefix(f"paper_assignment:{pid}:")
    if role == "reviewer":
        return {"assignments": [a for a in allx if a.get("user_id") == uid]}
    return {"assignments": allx}


@router.delete("/projects/{pid}/assignments")
def clear_assignments(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    existing = kv_get_by_prefix(f"paper_assignment:{pid}:")
    for r in existing:
        kv_del(f"paper_assignment:{pid}:{r['paper_id']}:{r['user_id']}")
    return {"cleared": len(existing)}


# ---- Participants (author-defined reviewer slots; no server join needed) ---
# The author's copy is the master. Reviewers are local "slots" the lead defines
# and assigns papers to; their work comes back via exported/imported bundles.

class ParticipantCreate(BaseModel):
    name: str
    role: Optional[str] = "reviewer"
    weight: Optional[float] = 1.0


@router.get("/projects/{pid}/participants")
def list_participants(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ROLES)
    parts = kv_get_by_prefix(f"project_participant:{pid}:")
    parts.sort(key=lambda x: x.get("created_at") or "")
    return {"participants": parts}


@router.post("/projects/{pid}/participants")
def add_participant(pid: str, body: ParticipantCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    part_id = _new_id("rev")
    rec = {
        "id": part_id, "project_id": pid, "name": (body.name or "Reviewer").strip(),
        "role": body.role or "reviewer", "weight": max(0.0, float(body.weight or 1.0)),
        "created_at": _now(),
    }
    kv_set(f"project_participant:{pid}:{part_id}", rec)
    return {"participant": rec}


class ParticipantUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    weight: Optional[float] = None


@router.put("/projects/{pid}/participants/{part_id}")
def update_participant(pid: str, part_id: str, body: ParticipantUpdate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    rec = kv_get(f"project_participant:{pid}:{part_id}")
    if not rec:
        raise HTTPException(status_code=404, detail="Participant not found")
    if body.name is not None:
        rec["name"] = body.name.strip()
    if body.role is not None:
        rec["role"] = body.role
    if body.weight is not None:
        rec["weight"] = max(0.0, float(body.weight))
    kv_set(f"project_participant:{pid}:{part_id}", rec)
    return {"participant": rec}


@router.delete("/projects/{pid}/participants/{part_id}")
def remove_participant(pid: str, part_id: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    kv_del(f"project_participant:{pid}:{part_id}")
    for a in kv_get_by_prefix(f"paper_assignment:{pid}:"):
        if a.get("user_id") == part_id:
            kv_del(f"paper_assignment:{pid}:{a['paper_id']}:{part_id}")
    return {"ok": True}


# ---- Granular auto-assignment across participants -------------------------

class AutoAssign(BaseModel):
    strategy: str = "dual"            # dual | overlap | weighted | manual
    overlap_pct: Optional[int] = 100  # for 'overlap': % of papers double-screened
    reviewers_per_paper: Optional[int] = 2
    include_calibration: Optional[bool] = True   # give every reviewer the calibration set
    manual: Optional[List[dict]] = None          # [{paper_id, participant_ids:[...]}]


@router.post("/projects/{pid}/auto-assign")
def auto_assign(pid: str, body: AutoAssign, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    papers = kv_get(f"project_papers:{pid}") or []
    parts = [p for p in kv_get_by_prefix(f"project_participant:{pid}:") if p.get("role") in ("reviewer", "lead", "adjudicator")]
    reviewers = [p for p in parts if p.get("role") in ("reviewer", "lead")]
    if not reviewers:
        raise HTTPException(status_code=400, detail="Add at least one reviewer participant first")

    # Clear prior assignments, then rebuild.
    for a in kv_get_by_prefix(f"paper_assignment:{pid}:"):
        kv_del(f"paper_assignment:{pid}:{a['paper_id']}:{a['user_id']}")

    now = _now()
    counts: Dict[str, int] = {r["id"]: 0 for r in reviewers}

    def assign(paper_id: str, rid: str):
        kv_set(f"paper_assignment:{pid}:{paper_id}:{rid}",
               {"project_id": pid, "paper_id": paper_id, "user_id": rid, "assigned_at": now, "strategy": body.strategy})
        counts[rid] = counts.get(rid, 0) + 1

    # Deterministic order (no RNG in the store): rotate by index.
    calib_ids = {p["paper_id"] for p in papers if p.get("calibration")}
    work = [p for p in papers if not p.get("calibration")]

    strat = body.strategy or "dual"
    if strat == "manual" and isinstance(body.manual, list):
        for m in body.manual:
            for rid in (m.get("participant_ids") or []):
                if rid in counts:
                    assign(m["paper_id"], rid)
    elif strat == "dual":
        n = max(1, min(len(reviewers), int(body.reviewers_per_paper or 2)))
        for i, p in enumerate(work):
            for k in range(n):
                assign(p["paper_id"], reviewers[(i + k) % len(reviewers)]["id"])
    elif strat == "overlap":
        pct = max(0, min(100, int(body.overlap_pct if body.overlap_pct is not None else 100)))
        # Every Nth paper (per pct) is double-screened; the rest are single, split round-robin.
        every = 0 if pct <= 0 else max(1, round(100 / pct)) if pct < 100 else 1
        for i, p in enumerate(work):
            double = pct >= 100 or (every and i % every == 0)
            assign(p["paper_id"], reviewers[i % len(reviewers)]["id"])
            if double and len(reviewers) > 1:
                assign(p["paper_id"], reviewers[(i + 1) % len(reviewers)]["id"])
    elif strat == "weighted":
        # Largest-remainder split by weight; each paper to a single reviewer.
        total_w = sum(max(0.0, float(r.get("weight") or 1.0)) for r in reviewers) or 1.0
        # Build a weighted round-robin sequence.
        seq: List[str] = []
        targets = {r["id"]: (float(r.get("weight") or 1.0) / total_w) * len(work) for r in reviewers}
        acc = {r["id"]: 0.0 for r in reviewers}
        for _ in range(len(work)):
            # pick reviewer whose (target - assigned) deficit is largest
            rid = max(reviewers, key=lambda r: targets[r["id"]] - acc[r["id"]])["id"]
            acc[rid] += 1.0
            seq.append(rid)
        for p, rid in zip(work, seq):
            assign(p["paper_id"], rid)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{strat}'")

    # Calibration set → every reviewer screens it (unless disabled).
    if body.include_calibration and calib_ids:
        for pid_ in calib_ids:
            for r in reviewers:
                assign(pid_, r["id"])

    return {
        "strategy": strat,
        "assigned": sum(counts.values()),
        "per_reviewer": [{"id": r["id"], "name": r.get("name"), "count": counts.get(r["id"], 0)} for r in reviewers],
        "calibration": len(calib_ids),
        "papers": len(papers),
    }


# ---- Tags ------------------------------------------------------------------

class TagsBody(BaseModel):
    tags: List[str] = []


@router.get("/projects/{pid}/tags")
def get_tags(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ROLES)
    project_tags = kv_get(f"project_tags:{pid}") or []
    paper_tags = {r["paper_id"]: r["tags"] for r in kv_get_by_prefix(f"paper_tags:{pid}:")}
    return {"tags": project_tags, "paper_tags": paper_tags}


@router.put("/projects/{pid}/tags")
def set_project_tags(pid: str, body: TagsBody, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    clean = sorted({t.strip() for t in body.tags if t.strip()})
    kv_set(f"project_tags:{pid}", clean)
    return {"tags": clean}


class PaperTagsBody(BaseModel):
    paper_id: str
    tags: List[str] = []


@router.put("/projects/{pid}/papers/{paper_id}/tags")
def set_paper_tags(pid: str, paper_id: str, body: PaperTagsBody, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "reviewer", "adjudicator"])
    tags = sorted({t.strip() for t in body.tags if t.strip()})
    if tags:
        kv_set(f"paper_tags:{pid}:{paper_id}", {"paper_id": paper_id, "tags": tags})
    else:
        kv_del(f"paper_tags:{pid}:{paper_id}")
    return {"paper_id": paper_id, "tags": tags}


# ---- Calibration set (gold decisions to train/check reviewers) -------------

class CalibrationItem(BaseModel):
    paper_id: str
    gold: Optional[str] = None       # include | exclude | maybe | None (unset)
    rationale: Optional[str] = ""
    is_calibration: bool = True


@router.put("/projects/{pid}/calibration")
def set_calibration(pid: str, items: List[CalibrationItem], request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    papers = kv_get(f"project_papers:{pid}") or []
    by_id = {p["paper_id"]: p for p in papers}
    for it in items:
        p = by_id.get(it.paper_id)
        if not p:
            continue
        p["calibration"] = bool(it.is_calibration)
        p["gold"] = it.gold
        p["gold_rationale"] = it.rationale or ""
    kv_set(f"project_papers:{pid}", papers)
    return {"calibration": [p["paper_id"] for p in papers if p.get("calibration")]}


# ---- Decisions + adjudications + blinding ---------------------------------

def _summarise_decision(d: dict) -> dict:
    return {
        "paper_id": d.get("paper_id"),
        "stage": d.get("stage"),
        "reviewer_user_id": d.get("reviewer_user_id"),
        "decision": d.get("decision"),
        "decided_at": d.get("decided_at"),
    }


@router.get("/projects/{pid}/decisions")
def get_decisions(pid: str, request: Request, stage: str = "abstract"):
    uid = current_user(request.headers.get("x-reviewer-id"))
    role = _require_role(pid, uid, ROLES)
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    allx = kv_get_by_prefix(f"decision:{pid}:{stage}:")
    adj = kv_get_by_prefix(f"adjudication:{pid}:{stage}:")

    is_blinded = project.get("screening_mode") == "dual_blinded" and role == "reviewer"
    if not is_blinded:
        return {"decisions": allx, "adjudications": adj}
    my_paper_ids = {d["paper_id"] for d in allx if d.get("reviewer_user_id") == uid}
    exposed = [d if (d.get("reviewer_user_id") == uid or d["paper_id"] in my_paper_ids) else _summarise_decision(d) for d in allx]
    exposed_adj = [a if a["paper_id"] in my_paper_ids else {"paper_id": a["paper_id"], "stage": a["stage"]} for a in adj]
    return {"decisions": exposed, "adjudications": exposed_adj, "blinded": True}


class DecisionCreate(BaseModel):
    paper_id: Optional[str] = None
    stage: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    per_pico_verdict: Optional[Any] = None
    ai_decision: Optional[Any] = None
    is_override: Optional[bool] = False


@router.post("/projects/{pid}/decisions")
def post_decision(pid: str, body: DecisionCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "reviewer", "adjudicator"])
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    if project.get("locked_at"):
        raise HTTPException(status_code=409, detail="Project is locked for analysis")
    stage = body.stage or "abstract"
    if not body.paper_id or not body.decision:
        raise HTTPException(status_code=400, detail="paper_id and decision required")
    key = f"decision:{pid}:{stage}:{body.paper_id}:{uid}"
    existing = kv_get(key)
    now = _now()
    dec = {
        "paper_id": body.paper_id,
        "stage": stage,
        "reviewer_user_id": uid,
        "decision": body.decision,
        "reason": body.reason or "",
        "per_pico_verdict": body.per_pico_verdict,
        "ai_decision": body.ai_decision if body.ai_decision is not None else (existing or {}).get("ai_decision"),
        "is_override": bool(body.is_override),
        "decided_at": now,
        "created_at": (existing or {}).get("created_at") or now,
    }
    kv_set(key, dec)
    return {"decision": dec}


@router.get("/projects/{pid}/conflicts")
def get_conflicts(pid: str, request: Request, stage: str = "abstract"):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "adjudicator"])
    allx = kv_get_by_prefix(f"decision:{pid}:{stage}:")
    adj = kv_get_by_prefix(f"adjudication:{pid}:{stage}:")
    adj_papers = {a["paper_id"] for a in adj}
    by_paper: dict = {}
    for d in allx:
        by_paper.setdefault(d["paper_id"], []).append(d)
    conflicts = []
    for paper_id, decisions in by_paper.items():
        if len(decisions) < 2:
            continue
        if len({d["decision"] for d in decisions}) <= 1:
            continue
        if paper_id in adj_papers:
            continue
        conflicts.append({"paper_id": paper_id, "decisions": decisions})
    return {"conflicts": conflicts}


class AdjudicationCreate(BaseModel):
    paper_id: Optional[str] = None
    stage: Optional[str] = None
    final_decision: Optional[str] = None
    rationale: Optional[str] = None


@router.post("/projects/{pid}/adjudications")
def post_adjudication(pid: str, body: AdjudicationCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "adjudicator"])
    stage = body.stage or "abstract"
    if not body.paper_id or not body.final_decision:
        raise HTTPException(status_code=400, detail="paper_id and final_decision required")
    key = f"adjudication:{pid}:{stage}:{body.paper_id}"
    existing = kv_get(key)
    now = _now()
    rec = {
        "paper_id": body.paper_id,
        "stage": stage,
        "adjudicator_user_id": uid,
        "final_decision": body.final_decision,
        "rationale": body.rationale or "",
        "decided_at": now,
        "created_at": (existing or {}).get("created_at") or now,
    }
    kv_set(key, rec)
    return {"adjudication": rec}


# ---------------------------------------------------------------------------
# Dual independent risk-of-bias assessment (scaffold)
# ---------------------------------------------------------------------------
# Reuses the multi-reviewer project model: each reviewer stores a per-paper RoB
# assessment (instrument + per-domain judgments); conflicts are flagged where two
# reviewers disagree on a domain. Mirrors the screening decisions/conflicts flow;
# the appraisal UI wiring lands in a later pass.

class RobAssessmentCreate(BaseModel):
    paper_id: Optional[str] = None
    instrument_id: Optional[str] = None
    domains: Optional[dict] = None          # {domain_id: judgment}
    overall: Optional[str] = None
    notes: Optional[str] = None


@router.post("/projects/{pid}/rob-assessments")
def post_rob_assessment(pid: str, body: RobAssessmentCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "reviewer", "adjudicator"])
    if not body.paper_id or not body.instrument_id:
        raise HTTPException(status_code=400, detail="paper_id and instrument_id required")
    key = f"rob_assessment:{pid}:{body.paper_id}:{uid}"
    existing = kv_get(key)
    now = _now()
    rec = {
        "paper_id": body.paper_id, "reviewer_user_id": uid,
        "instrument_id": body.instrument_id,
        "domains": body.domains or {}, "overall": body.overall,
        "notes": body.notes or "", "assessed_at": now,
        "created_at": (existing or {}).get("created_at") or now,
    }
    kv_set(key, rec)
    return {"assessment": rec}


@router.get("/projects/{pid}/rob-assessments")
def get_rob_assessments(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ROLES)
    return {"assessments": kv_get_by_prefix(f"rob_assessment:{pid}:")}


@router.get("/projects/{pid}/rob-conflicts")
def get_rob_conflicts(pid: str, request: Request):
    """Flag papers where two reviewers disagree on any domain judgment."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "adjudicator"])
    by_paper: dict = {}
    for a in kv_get_by_prefix(f"rob_assessment:{pid}:"):
        by_paper.setdefault(a["paper_id"], []).append(a)
    conflicts = []
    for paper_id, assessments in by_paper.items():
        if len(assessments) < 2:
            continue
        domain_ids = set().union(*[set((a.get("domains") or {}).keys()) for a in assessments])
        disagreements = [
            did for did in domain_ids
            if len({(a.get("domains") or {}).get(did) for a in assessments}) > 1
        ]
        if disagreements:
            conflicts.append({"paper_id": paper_id, "domains": disagreements, "assessments": assessments})
    return {"conflicts": conflicts}


# ---------------------------------------------------------------------------
# Dual independent data extraction
# ---------------------------------------------------------------------------
# Two reviewers extract the same structured fields for each included study,
# independently; disagreements are reconciled into one agreed value per field.
# Mirrors the screening decisions/conflicts/adjudication flow, but keyed by
# (paper, field) rather than a single include/exclude decision. Key layout:
#   extraction_template:{pid}            -> {fields:[{id,label,group,type,options}]}
#   extraction:{pid}:{paper}:{uid}       -> one reviewer's values for a paper
#   extraction_final:{pid}:{paper}       -> the reconciled (agreed) values

DEFAULT_EXTRACTION_FIELDS = [
    {"id": "study_design", "label": "Study design", "group": "Study", "type": "category",
     "options": ["RCT", "Cohort", "Case-control", "Cross-sectional", "Case series", "Other"]},
    {"id": "country", "label": "Country / setting", "group": "Study", "type": "text", "options": []},
    {"id": "year", "label": "Publication year", "group": "Study", "type": "number", "options": []},
    {"id": "funding", "label": "Funding source", "group": "Study", "type": "text", "options": []},
    {"id": "n_total", "label": "Total sample size", "group": "Population", "type": "number", "options": []},
    {"id": "population", "label": "Population / condition", "group": "Population", "type": "text", "options": []},
    {"id": "intervention", "label": "Intervention", "group": "Intervention", "type": "text", "options": []},
    {"id": "comparator", "label": "Comparator", "group": "Intervention", "type": "text", "options": []},
    {"id": "outcome_name", "label": "Primary outcome", "group": "Outcomes", "type": "text", "options": []},
    {"id": "effect_size", "label": "Effect estimate", "group": "Outcomes", "type": "text", "options": []},
    {"id": "ci", "label": "95% CI", "group": "Outcomes", "type": "text", "options": []},
    {"id": "timepoint", "label": "Timepoint", "group": "Outcomes", "type": "text", "options": []},
]


def _extraction_template(pid: str) -> dict:
    tpl = kv_get(f"extraction_template:{pid}")
    if not tpl or not tpl.get("fields"):
        return {"fields": DEFAULT_EXTRACTION_FIELDS, "is_default": True}
    return {"fields": tpl["fields"], "is_default": False}


@router.get("/projects/{pid}/extraction-template")
def get_extraction_template(pid: str, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ROLES)
    return _extraction_template(pid)


class ExtractionTemplatePut(BaseModel):
    fields: List[dict] = []


@router.put("/projects/{pid}/extraction-template")
def put_extraction_template(pid: str, body: ExtractionTemplatePut, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    fields = []
    for f in (body.fields or []):
        fid = (f.get("id") or "").strip() or _new_id("fld")
        ftype = f.get("type") if f.get("type") in ("text", "number", "category", "date") else "text"
        fields.append({
            "id": fid,
            "label": (f.get("label") or fid).strip(),
            "group": (f.get("group") or "General").strip(),
            "type": ftype,
            "options": [str(o).strip() for o in (f.get("options") or []) if str(o).strip()],
        })
    kv_set(f"extraction_template:{pid}", {"fields": fields})
    return {"fields": fields, "is_default": False}


def _summarise_extraction(e: dict) -> dict:
    return {
        "paper_id": e.get("paper_id"),
        "reviewer_user_id": e.get("reviewer_user_id"),
        "submitted": e.get("submitted"),
        "extracted_at": e.get("extracted_at"),
    }


@router.get("/projects/{pid}/extractions")
def get_extractions(pid: str, request: Request):
    """All per-reviewer extractions plus reconciled finals. In dual_blinded mode
    a reviewer only sees others' values for a paper once they have submitted
    their own, mirroring the screening blinding rule."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    role = _require_role(pid, uid, ROLES)
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    allx = kv_get_by_prefix(f"extraction:{pid}:")
    finals = kv_get_by_prefix(f"extraction_final:{pid}:")
    is_blinded = project.get("screening_mode") == "dual_blinded" and role == "reviewer"
    if not is_blinded:
        return {"extractions": allx, "finals": finals}
    my_papers = {e["paper_id"] for e in allx if e.get("reviewer_user_id") == uid and e.get("submitted")}
    exposed = [
        e if (e.get("reviewer_user_id") == uid or e["paper_id"] in my_papers) else _summarise_extraction(e)
        for e in allx
    ]
    exposed_finals = [f for f in finals if f["paper_id"] in my_papers]
    return {"extractions": exposed, "finals": exposed_finals, "blinded": True}


class ExtractionCreate(BaseModel):
    paper_id: Optional[str] = None
    values: Optional[dict] = None
    submitted: Optional[bool] = True
    ai_prefilled: Optional[bool] = False


@router.post("/projects/{pid}/extractions")
def post_extraction(pid: str, body: ExtractionCreate, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "reviewer", "adjudicator"])
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    if project.get("locked_at"):
        raise HTTPException(status_code=409, detail="Project is locked for analysis")
    if not body.paper_id:
        raise HTTPException(status_code=400, detail="paper_id required")
    key = f"extraction:{pid}:{body.paper_id}:{uid}"
    existing = kv_get(key)
    now = _now()
    rec = {
        "paper_id": body.paper_id,
        "reviewer_user_id": uid,
        "values": body.values or {},
        "submitted": bool(body.submitted),
        "ai_prefilled": bool(body.ai_prefilled),
        "extracted_at": now,
        "created_at": (existing or {}).get("created_at") or now,
    }
    kv_set(key, rec)
    return {"extraction": rec}


def _norm_value(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip().lower()


@router.get("/projects/{pid}/extraction-conflicts")
def get_extraction_conflicts(pid: str, request: Request):
    """Per-paper, per-field disagreements between submitted extractions,
    excluding fields already reconciled into the final record."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "adjudicator"])
    field_ids = [f["id"] for f in _extraction_template(pid)["fields"]]
    by_paper: dict = {}
    for e in kv_get_by_prefix(f"extraction:{pid}:"):
        if e.get("submitted"):
            by_paper.setdefault(e["paper_id"], []).append(e)
    finals = {f["paper_id"]: f for f in kv_get_by_prefix(f"extraction_final:{pid}:")}
    out = []
    for paper_id, exts in by_paper.items():
        if len(exts) < 2:
            continue
        final_vals = (finals.get(paper_id) or {}).get("values") or {}
        conflict_fields = []
        for fid in field_ids:
            norm = {_norm_value((e.get("values") or {}).get(fid)) for e in exts}
            if len(norm) > 1 and fid not in final_vals:
                conflict_fields.append(fid)
        if conflict_fields:
            out.append({"paper_id": paper_id, "fields": conflict_fields, "extractions": exts})
    return {"conflicts": out}


class ExtractionReconcile(BaseModel):
    paper_id: Optional[str] = None
    values: Optional[dict] = None
    rationale: Optional[str] = None


@router.post("/projects/{pid}/extraction-reconciliations")
def post_extraction_reconciliation(pid: str, body: ExtractionReconcile, request: Request):
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead", "adjudicator"])
    if not body.paper_id:
        raise HTTPException(status_code=400, detail="paper_id required")
    key = f"extraction_final:{pid}:{body.paper_id}"
    existing = kv_get(key)
    now = _now()
    merged = dict((existing or {}).get("values") or {})
    merged.update(body.values or {})
    rec = {
        "paper_id": body.paper_id,
        "values": merged,
        "reconciled_by": uid,
        "rationale": body.rationale or "",
        "reconciled_at": now,
        "created_at": (existing or {}).get("created_at") or now,
    }
    kv_set(key, rec)
    return {"final": rec}


# ---------------------------------------------------------------------------
# Portable project bundles (federation + reproducibility)
# ---------------------------------------------------------------------------
# One versioned JSON serialisation of a whole project. It powers three things:
#   * Federation: the author exports a bundle, a reviewer imports it as a new
#     local project, works on it, exports their copy, and the author merges the
#     returned contributions back in.
#   * Reproducibility: the same export is a complete, timestamped record of the
#     review (search, criteria, papers, every decision + reason + who + when,
#     adjudications, extractions) that a journal or auditor can reconstruct.
# The bundle is plain JSON; the author owns the file. There is no central server.

BUNDLE_VERSION = 1
BUNDLE_KIND = "evidence-engine-project"


def _gather_project(pid: str) -> dict:
    project = kv_get(f"project:{pid}")
    if not project:
        raise HTTPException(status_code=404, detail="Not found")
    tpl = kv_get(f"extraction_template:{pid}")
    return {
        "bundle_version": BUNDLE_VERSION,
        "kind": BUNDLE_KIND,
        "generated_at": _now(),
        "project": project,
        "members": kv_get_by_prefix(f"project_member:{pid}:"),
        "participants": kv_get_by_prefix(f"project_participant:{pid}:"),
        "papers": kv_get(f"project_papers:{pid}") or [],
        "tags": kv_get(f"project_tags:{pid}") or {},
        "extraction_template": (tpl or {}).get("fields") or DEFAULT_EXTRACTION_FIELDS,
        "assignments": kv_get_by_prefix(f"paper_assignment:{pid}:"),
        "decisions": kv_get_by_prefix(f"decision:{pid}:"),
        "adjudications": kv_get_by_prefix(f"adjudication:{pid}:"),
        "extractions": kv_get_by_prefix(f"extraction:{pid}:"),
        "extraction_finals": kv_get_by_prefix(f"extraction_final:{pid}:"),
        "rob_assessments": kv_get_by_prefix(f"rob_assessment:{pid}:"),
    }


@router.get("/projects/{pid}/export")
def export_project(pid: str, request: Request):
    """Full project bundle: the federation hand-off file and, equally, the
    reproducibility record. Any member may export."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ROLES)
    return _gather_project(pid)


class BundleImport(BaseModel):
    bundle: Optional[dict] = None


def _write_contributions(pid: str, b: dict) -> dict:
    """Upsert reviewer contributions and shared structures from a bundle into
    project `pid`, keyed by their natural (paper, reviewer, stage) keys so the
    merge is idempotent and never clobbers unrelated records."""
    counts = {k: 0 for k in (
        "decisions", "adjudications", "extractions", "extraction_finals",
        "rob_assessments", "participants", "assignments", "papers",
    )}
    for d in b.get("decisions") or []:
        if not d.get("paper_id"):
            continue
        stage = d.get("stage") or "abstract"
        ruid = d.get("reviewer_user_id") or "imported"
        kv_set(f"decision:{pid}:{stage}:{d['paper_id']}:{ruid}", {**d, "stage": stage, "reviewer_user_id": ruid})
        counts["decisions"] += 1
    for a in b.get("adjudications") or []:
        if not a.get("paper_id"):
            continue
        stage = a.get("stage") or "abstract"
        kv_set(f"adjudication:{pid}:{stage}:{a['paper_id']}", {**a, "stage": stage})
        counts["adjudications"] += 1
    for e in b.get("extractions") or []:
        if not e.get("paper_id"):
            continue
        ruid = e.get("reviewer_user_id") or "imported"
        kv_set(f"extraction:{pid}:{e['paper_id']}:{ruid}", {**e, "reviewer_user_id": ruid})
        counts["extractions"] += 1
    for f in b.get("extraction_finals") or []:
        if not f.get("paper_id"):
            continue
        kv_set(f"extraction_final:{pid}:{f['paper_id']}", f)
        counts["extraction_finals"] += 1
    for r in b.get("rob_assessments") or []:
        if not r.get("paper_id"):
            continue
        ruid = r.get("reviewer_user_id") or "imported"
        kv_set(f"rob_assessment:{pid}:{r['paper_id']}:{ruid}", {**r, "reviewer_user_id": ruid})
        counts["rob_assessments"] += 1
    for p in b.get("participants") or []:
        if not p.get("id"):
            continue
        kv_set(f"project_participant:{pid}:{p['id']}", {**p, "project_id": pid})
        counts["participants"] += 1
    for a in b.get("assignments") or []:
        if not (a.get("paper_id") and a.get("user_id")):
            continue
        kv_set(f"paper_assignment:{pid}:{a['paper_id']}:{a['user_id']}", {**a, "project_id": pid})
        counts["assignments"] += 1
    # Union new papers by paper_id; never drop existing ones.
    papers = kv_get(f"project_papers:{pid}") or []
    seen = {p.get("paper_id") for p in papers}
    for p in b.get("papers") or []:
        if p.get("paper_id") and p["paper_id"] not in seen:
            papers.append(p)
            seen.add(p["paper_id"])
            counts["papers"] += 1
    kv_set(f"project_papers:{pid}", papers)
    return counts


@router.post("/projects/{pid}/import")
def import_into_project(pid: str, body: BundleImport, request: Request):
    """Merge a returned bundle's reviewer contributions into an existing
    project. The author's project record and template are left as-is."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    _require_role(pid, uid, ["lead"])
    if not kv_get(f"project:{pid}"):
        raise HTTPException(status_code=404, detail="Not found")
    b = body.bundle or {}
    if b.get("kind") != BUNDLE_KIND:
        raise HTTPException(status_code=400, detail="Not an Evidence Engine project bundle")
    counts = _write_contributions(pid, b)
    # Adopt the template only if this project has none yet.
    if not kv_get(f"extraction_template:{pid}") and b.get("extraction_template"):
        kv_set(f"extraction_template:{pid}", {"fields": b["extraction_template"]})
    return {"merged": counts}


@router.post("/projects/import")
def import_new_project(body: BundleImport, request: Request):
    """Create a NEW local project from a bundle (a reviewer receiving the
    author's hand-off, or restoring a reproducibility export). The importer
    becomes the lead of their local copy."""
    uid = current_user(request.headers.get("x-reviewer-id"))
    b = body.bundle or {}
    if b.get("kind") != BUNDLE_KIND:
        raise HTTPException(status_code=400, detail="Not an Evidence Engine project bundle")
    src = b.get("project") or {}
    now = _now()
    pid = _new_id("prj")
    project = {
        "id": pid,
        "name": (src.get("name") or "Imported project"),
        "owner_user_id": uid,
        "pico": src.get("pico") or {"population": "", "intervention": "", "comparator": "", "outcome": ""},
        "inclusion": src.get("inclusion") or [],
        "exclusion": src.get("exclusion") or [],
        "screening_mode": src.get("screening_mode") or "dual_blinded",
        "visibility": "invite",
        "locked_at": None,
        "created_at": now,
        "updated_at": now,
        "imported_from": src.get("id"),
    }
    kv_set(f"project:{pid}", project)
    kv_set(f"project_member:{pid}:{uid}", {"project_id": pid, "user_id": uid, "role": "lead", "joined_at": now})
    kv_set(f"user_project:{uid}:{pid}", {"project_id": pid, "joined_at": now, "role": "lead"})
    if b.get("tags"):
        kv_set(f"project_tags:{pid}", b["tags"])
    if b.get("extraction_template"):
        kv_set(f"extraction_template:{pid}", {"fields": b["extraction_template"]})
    _write_contributions(pid, b)
    return {"project": project}
