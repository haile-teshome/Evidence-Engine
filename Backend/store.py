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

import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

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


def current_user(x_reviewer_id: Optional[str] = Header(default=None)) -> str:
    """Resolve the acting reviewer from the ``X-Reviewer-Id`` header.

    Local mode has no real auth: the frontend sends the selected profile id, and
    we default to the built-in ``"local"`` profile when none is supplied.
    """
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
    items = kv_get_by_prefix("reviewer:")
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
    return {"reviewer": rec}


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
