// @ts-nocheck — Deno edge function: Deno + npm:/jsr: specifiers are not
// resolvable by the project's browser-targeted tsconfig. Supabase compiles
// this file natively at deploy time.
import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";
import * as kv from "./kv_store.ts";

const app = new Hono();
app.use("*", logger(console.log));
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

const supabaseAdmin = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
);

async function authedUserId(c: any): Promise<string | null> {
  const token = c.req.header("Authorization")?.split(" ")[1];
  if (!token) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(token);
  if (error || !data.user) return null;
  return data.user.id;
}

app.get("/make-server-7e4eb0f2/health", (c) => c.json({ status: "ok" }));

// Sign up
app.post("/make-server-7e4eb0f2/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();
    if (!email || !password) return c.json({ error: "email and password required" }, 400);
    const { data, error } = await supabaseAdmin.auth.admin.createUser({
      email,
      password,
      user_metadata: { name: name || "" },
      // Auto-confirm since no email server is configured in this environment
      email_confirm: true,
    });
    if (error) {
      console.log(`Signup error for ${email}: ${error.message}`);
      return c.json({ error: error.message }, 400);
    }
    return c.json({ user: { id: data.user.id, email: data.user.email } });
  } catch (e) {
    console.log(`Signup unexpected error: ${e}`);
    return c.json({ error: `Signup failed: ${e}` }, 500);
  }
});

// List a user's saved sessions (metadata only)
app.get("/make-server-7e4eb0f2/sessions", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const items = await kv.getByPrefix(`session:${uid}:`);
    const meta = (items || []).map((s: any) => ({
      id: s.id,
      title: s.title,
      updated_at: s.updated_at,
      created_at: s.created_at,
    })).sort((a: any, b: any) => (b.updated_at || "").localeCompare(a.updated_at || ""));
    return c.json({ sessions: meta });
  } catch (e) {
    console.log(`List sessions error for ${uid}: ${e}`);
    return c.json({ error: `Failed to list sessions: ${e}` }, 500);
  }
});

// Load a single session
app.get("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    const session = await kv.get(`session:${uid}:${id}`);
    if (!session) return c.json({ error: "Not found" }, 404);
    return c.json({ session });
  } catch (e) {
    console.log(`Load session error: ${e}`);
    return c.json({ error: `Failed to load session: ${e}` }, 500);
  }
});

// Save / update a session
app.put("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    const body = await c.req.json();
    const now = new Date().toISOString();
    const existing = await kv.get(`session:${uid}:${id}`);
    const session = {
      id,
      title: body.title || existing?.title || "Untitled session",
      data: body.data ?? {},
      created_at: existing?.created_at || now,
      updated_at: now,
    };
    await kv.set(`session:${uid}:${id}`, session);
    return c.json({ session });
  } catch (e) {
    console.log(`Save session error: ${e}`);
    return c.json({ error: `Failed to save session: ${e}` }, 500);
  }
});

// Delete a session
app.delete("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    await kv.del(`session:${uid}:${id}`);
    return c.json({ ok: true });
  } catch (e) {
    console.log(`Delete session error: ${e}`);
    return c.json({ error: `Failed to delete session: ${e}` }, 500);
  }
});

// =========================================================================
// Multi-reviewer projects
// -------------------------------------------------------------------------
// KV layout:
//   project:{pid}                                       → project metadata
//   project_member:{pid}:{uid}                          → role + joined_at
//   user_project:{uid}:{pid}                            → backlink (fast list-by-user)
//   project_papers:{pid}                                → array of paper records
//   decision:{pid}:{stage}:{paperId}:{uid}              → one reviewer's decision
//   adjudication:{pid}:{stage}:{paperId}                → adjudicator final call
//   invite:{token}                                      → invite record
// =========================================================================

type Role = "lead" | "reviewer" | "adjudicator" | "viewer";

async function getRole(pid: string, uid: string): Promise<Role | null> {
  const m = await kv.get(`project_member:${pid}:${uid}`);
  return m ? (m.role as Role) : null;
}

async function requireRole(c: any, pid: string, allowed: Role[]): Promise<{ uid: string; role: Role } | Response> {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  const role = await getRole(pid, uid);
  if (!role) return c.json({ error: "Not a project member" }, 403);
  if (!allowed.includes(role)) return c.json({ error: `Role '${role}' cannot perform this action` }, 403);
  return { uid, role };
}

function newId(prefix: string): string {
  const r = crypto.getRandomValues(new Uint8Array(8));
  const hex = [...r].map(b => b.toString(16).padStart(2, "0")).join("");
  return `${prefix}_${Date.now().toString(36)}_${hex}`;
}

// ---- Project CRUD --------------------------------------------------------

app.post("/make-server-7e4eb0f2/projects", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const body = await c.req.json();
    const now = new Date().toISOString();
    const id = newId("prj");
    const project = {
      id,
      name: body.name || "Untitled project",
      owner_user_id: uid,
      pico: body.pico || { population: "", intervention: "", comparator: "", outcome: "" },
      inclusion: body.inclusion || [],
      exclusion: body.exclusion || [],
      screening_mode: body.screening_mode || "dual_blinded",   // single | dual | dual_blinded
      visibility: body.visibility || "invite",                  // private | invite | link
      locked_at: null,
      created_at: now,
      updated_at: now,
    };
    await kv.set(`project:${id}`, project);
    await kv.set(`project_member:${id}:${uid}`, { project_id: id, user_id: uid, role: "lead", joined_at: now });
    await kv.set(`user_project:${uid}:${id}`, { project_id: id, joined_at: now, role: "lead" });
    return c.json({ project });
  } catch (e) {
    console.log(`Create project error: ${e}`);
    return c.json({ error: `Failed to create project: ${e}` }, 500);
  }
});

app.get("/make-server-7e4eb0f2/projects", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const backlinks = (await kv.getByPrefix(`user_project:${uid}:`)) || [];
    const projects = [];
    for (const bl of backlinks) {
      const p = await kv.get(`project:${bl.project_id}`);
      if (p) projects.push({ ...p, my_role: bl.role });
    }
    projects.sort((a: any, b: any) => (b.updated_at || "").localeCompare(a.updated_at || ""));
    return c.json({ projects });
  } catch (e) {
    return c.json({ error: `Failed to list projects: ${e}` }, 500);
  }
});

app.get("/make-server-7e4eb0f2/projects/:pid", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "reviewer", "adjudicator", "viewer"]);
  if (gate instanceof Response) return gate;
  const project = await kv.get(`project:${pid}`);
  if (!project) return c.json({ error: "Not found" }, 404);
  const members = (await kv.getByPrefix(`project_member:${pid}:`)) || [];
  return c.json({ project: { ...project, my_role: gate.role }, members });
});

app.put("/make-server-7e4eb0f2/projects/:pid", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  try {
    const body = await c.req.json();
    const existing = await kv.get(`project:${pid}`);
    if (!existing) return c.json({ error: "Not found" }, 404);
    if (existing.locked_at && body.locked_at === undefined) {
      // Allow only role/name edits when locked
      const patch: any = { ...existing, name: body.name ?? existing.name, updated_at: new Date().toISOString() };
      await kv.set(`project:${pid}`, patch);
      return c.json({ project: patch });
    }
    const next = {
      ...existing,
      name: body.name ?? existing.name,
      pico: body.pico ?? existing.pico,
      inclusion: body.inclusion ?? existing.inclusion,
      exclusion: body.exclusion ?? existing.exclusion,
      // Mode is mutable only before any screening has happened.
      screening_mode: existing.locked_at ? existing.screening_mode : (body.screening_mode ?? existing.screening_mode),
      visibility: body.visibility ?? existing.visibility,
      locked_at: body.locked_at !== undefined ? body.locked_at : existing.locked_at,
      updated_at: new Date().toISOString(),
    };
    await kv.set(`project:${pid}`, next);
    return c.json({ project: next });
  } catch (e) {
    return c.json({ error: `Failed to update project: ${e}` }, 500);
  }
});

app.post("/make-server-7e4eb0f2/projects/:pid/lock", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  const existing = await kv.get(`project:${pid}`);
  if (!existing) return c.json({ error: "Not found" }, 404);
  const next = { ...existing, locked_at: new Date().toISOString(), updated_at: new Date().toISOString() };
  await kv.set(`project:${pid}`, next);
  return c.json({ project: next });
});

// ---- Members + invites ---------------------------------------------------

app.put("/make-server-7e4eb0f2/projects/:pid/members/:uid/role", async (c) => {
  const pid = c.req.param("pid");
  const targetUid = c.req.param("uid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  const body = await c.req.json();
  const role: Role = body.role;
  if (!["lead", "reviewer", "adjudicator", "viewer"].includes(role)) {
    return c.json({ error: "Invalid role" }, 400);
  }
  const m = await kv.get(`project_member:${pid}:${targetUid}`);
  if (!m) return c.json({ error: "Member not found" }, 404);
  const next = { ...m, role };
  await kv.set(`project_member:${pid}:${targetUid}`, next);
  await kv.set(`user_project:${targetUid}:${pid}`, { project_id: pid, joined_at: m.joined_at, role });
  return c.json({ member: next });
});

app.post("/make-server-7e4eb0f2/projects/:pid/invites", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  const body = await c.req.json();
  const role: Role = body.role || "reviewer";
  const token = newId("inv").replace("inv_", "");
  const invite = {
    token,
    project_id: pid,
    role,
    created_by: gate.uid,
    created_at: new Date().toISOString(),
    expires_at: body.expires_at || null,
    used_at: null,
    used_by: null,
  };
  await kv.set(`invite:${token}`, invite);
  return c.json({ invite });
});

app.get("/make-server-7e4eb0f2/invites/:token", async (c) => {
  const token = c.req.param("token");
  const invite = await kv.get(`invite:${token}`);
  if (!invite) return c.json({ error: "Invite not found" }, 404);
  if (invite.used_at) return c.json({ error: "Invite already used" }, 410);
  if (invite.expires_at && new Date(invite.expires_at) < new Date()) {
    return c.json({ error: "Invite expired" }, 410);
  }
  const project = await kv.get(`project:${invite.project_id}`);
  return c.json({ invite, project: project ? { id: project.id, name: project.name } : null });
});

app.post("/make-server-7e4eb0f2/invites/:token/accept", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  const token = c.req.param("token");
  const invite = await kv.get(`invite:${token}`);
  if (!invite) return c.json({ error: "Invite not found" }, 404);
  if (invite.used_at) return c.json({ error: "Invite already used" }, 410);
  if (invite.expires_at && new Date(invite.expires_at) < new Date()) {
    return c.json({ error: "Invite expired" }, 410);
  }
  const pid = invite.project_id;
  const existing = await kv.get(`project_member:${pid}:${uid}`);
  if (existing) return c.json({ project_id: pid, already_member: true, role: existing.role });
  const now = new Date().toISOString();
  await kv.set(`project_member:${pid}:${uid}`, { project_id: pid, user_id: uid, role: invite.role, joined_at: now });
  await kv.set(`user_project:${uid}:${pid}`, { project_id: pid, joined_at: now, role: invite.role });
  await kv.set(`invite:${token}`, { ...invite, used_at: now, used_by: uid });
  return c.json({ project_id: pid, role: invite.role });
});

// ---- Papers (the corpus under screening) ---------------------------------

app.get("/make-server-7e4eb0f2/projects/:pid/papers", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "reviewer", "adjudicator", "viewer"]);
  if (gate instanceof Response) return gate;
  const papers = (await kv.get(`project_papers:${pid}`)) || [];
  // Reviewers see only the papers they're assigned to (if any assignments
  // exist for this project). Lead / adjudicator / viewer always see all.
  if (gate.role === "reviewer") {
    const myAssign = (await kv.getByPrefix(`paper_assignment:${pid}:`)) || [];
    if (myAssign.length > 0) {
      const myPaperIds = new Set(myAssign
        .filter((a: any) => a.user_id === gate.uid)
        .map((a: any) => a.paper_id));
      return c.json({ papers: papers.filter((p: any) => myPaperIds.has(p.paper_id)), assigned: true, total: papers.length });
    }
  }
  return c.json({ papers });
});

app.put("/make-server-7e4eb0f2/projects/:pid/papers", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  const body = await c.req.json();
  const papers = Array.isArray(body.papers) ? body.papers : [];
  await kv.set(`project_papers:${pid}`, papers);
  return c.json({ count: papers.length });
});

// ---- Assignments ---------------------------------------------------------
// Per-(project, paper, reviewer) assignment records. A project with NO
// assignments is treated as full-overlap: every reviewer sees every paper.
// Once `POST /assignments` is called with a strategy, GET /papers above
// filters reviewers down to their assigned subset.

app.post("/make-server-7e4eb0f2/projects/:pid/assignments", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  try {
    const body = await c.req.json();
    const strategy: "full_overlap" | "split" | "custom" = body.strategy || "full_overlap";
    const papers = (await kv.get(`project_papers:${pid}`)) || [];
    const members = ((await kv.getByPrefix(`project_member:${pid}:`)) || [])
      .filter((m: any) => m.role === "reviewer" || m.role === "lead");
    if (members.length === 0) return c.json({ error: "No reviewers in project" }, 400);

    // Wipe prior assignments so re-distributing is idempotent.
    const existing = (await kv.getByPrefix(`paper_assignment:${pid}:`)) || [];
    for (const r of existing) {
      await kv.del(`paper_assignment:${pid}:${r.paper_id}:${r.user_id}`);
    }

    const now = new Date().toISOString();
    let assigned = 0;

    if (strategy === "full_overlap") {
      for (const p of papers) {
        for (const m of members) {
          await kv.set(`paper_assignment:${pid}:${p.paper_id}:${m.user_id}`, {
            project_id: pid, paper_id: p.paper_id, user_id: m.user_id, assigned_at: now, strategy,
          });
          assigned++;
        }
      }
    } else if (strategy === "split") {
      const N = Math.max(1, Math.min(members.length, parseInt(body.reviewers_per_paper || "2", 10)));
      // Round-robin: paper i gets reviewers (i, i+1, ..., i+N-1) mod len(members)
      for (let i = 0; i < papers.length; i++) {
        for (let k = 0; k < N; k++) {
          const m = members[(i + k) % members.length];
          await kv.set(`paper_assignment:${pid}:${papers[i].paper_id}:${m.user_id}`, {
            project_id: pid, paper_id: papers[i].paper_id, user_id: m.user_id, assigned_at: now, strategy,
          });
          assigned++;
        }
      }
    } else if (strategy === "custom" && Array.isArray(body.custom)) {
      for (const a of body.custom) {
        for (const uid of (a.user_ids || [])) {
          await kv.set(`paper_assignment:${pid}:${a.paper_id}:${uid}`, {
            project_id: pid, paper_id: a.paper_id, user_id: uid, assigned_at: now, strategy,
          });
          assigned++;
        }
      }
    } else {
      return c.json({ error: "Unknown strategy" }, 400);
    }

    return c.json({ strategy, assigned, papers: papers.length, reviewers: members.length });
  } catch (e) {
    return c.json({ error: `Failed to assign: ${e}` }, 500);
  }
});

app.get("/make-server-7e4eb0f2/projects/:pid/assignments", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "reviewer", "adjudicator", "viewer"]);
  if (gate instanceof Response) return gate;
  const all = (await kv.getByPrefix(`paper_assignment:${pid}:`)) || [];
  if (gate.role === "reviewer") {
    return c.json({ assignments: all.filter((a: any) => a.user_id === gate.uid) });
  }
  return c.json({ assignments: all });
});

app.delete("/make-server-7e4eb0f2/projects/:pid/assignments", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead"]);
  if (gate instanceof Response) return gate;
  const existing = (await kv.getByPrefix(`paper_assignment:${pid}:`)) || [];
  for (const r of existing) {
    await kv.del(`paper_assignment:${pid}:${r.paper_id}:${r.user_id}`);
  }
  return c.json({ cleared: existing.length });
});

// ---- Decisions + adjudications + blinding --------------------------------

function summariseDecision(d: any) {
  return {
    paper_id: d.paper_id,
    stage: d.stage,
    reviewer_user_id: d.reviewer_user_id,
    decision: d.decision,
    decided_at: d.decided_at,
  };
}

app.get("/make-server-7e4eb0f2/projects/:pid/decisions", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "reviewer", "adjudicator", "viewer"]);
  if (gate instanceof Response) return gate;
  const project = await kv.get(`project:${pid}`);
  if (!project) return c.json({ error: "Not found" }, 404);
  const stage = c.req.query("stage") || "abstract";
  const all = (await kv.getByPrefix(`decision:${pid}:${stage}:`)) || [];
  const adj = (await kv.getByPrefix(`adjudication:${pid}:${stage}:`)) || [];

  // Apply blinding: if dual_blinded and the requester is a reviewer (not
  // lead/adjudicator), strip out other reviewers' full decisions on papers
  // where the requester has not yet decided. The requester always sees their
  // own decisions and the existence (count) of others.
  const isBlinded = project.screening_mode === "dual_blinded"
    && (gate.role === "reviewer");
  if (!isBlinded) {
    return c.json({ decisions: all, adjudications: adj });
  }
  // Group by paper_id, decide what to expose
  const myDecisionPaperIds = new Set<string>(
    all.filter((d: any) => d.reviewer_user_id === gate.uid).map((d: any) => d.paper_id),
  );
  const exposed = all.map((d: any) => {
    if (d.reviewer_user_id === gate.uid) return d;
    if (myDecisionPaperIds.has(d.paper_id)) return d;
    return summariseDecision(d);
  });
  // Same blinding for adjudications (don't reveal adjudicated answer until I've decided)
  const exposedAdj = adj.map((a: any) => myDecisionPaperIds.has(a.paper_id) ? a : { paper_id: a.paper_id, stage: a.stage });
  return c.json({ decisions: exposed, adjudications: exposedAdj, blinded: true });
});

app.post("/make-server-7e4eb0f2/projects/:pid/decisions", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "reviewer", "adjudicator"]);
  if (gate instanceof Response) return gate;
  const project = await kv.get(`project:${pid}`);
  if (!project) return c.json({ error: "Not found" }, 404);
  if (project.locked_at) return c.json({ error: "Project is locked for analysis" }, 409);
  try {
    const body = await c.req.json();
    const stage = body.stage || "abstract";
    if (!body.paper_id || !body.decision) return c.json({ error: "paper_id and decision required" }, 400);
    const key = `decision:${pid}:${stage}:${body.paper_id}:${gate.uid}`;
    const existing = await kv.get(key);
    const now = new Date().toISOString();
    const dec = {
      paper_id: body.paper_id,
      stage,
      reviewer_user_id: gate.uid,
      decision: body.decision,                  // "include" | "exclude" | "maybe"
      reason: body.reason || "",
      per_pico_verdict: body.per_pico_verdict || null,
      ai_decision: body.ai_decision || existing?.ai_decision || null,
      is_override: !!body.is_override,
      decided_at: now,
      created_at: existing?.created_at || now,
    };
    await kv.set(key, dec);
    return c.json({ decision: dec });
  } catch (e) {
    return c.json({ error: `Failed to write decision: ${e}` }, 500);
  }
});

app.get("/make-server-7e4eb0f2/projects/:pid/conflicts", async (c) => {
  const pid = c.req.param("pid");
  // Only adjudicator + lead can SEE conflicts (would defeat blinding otherwise).
  const gate = await requireRole(c, pid, ["lead", "adjudicator"]);
  if (gate instanceof Response) return gate;
  const stage = c.req.query("stage") || "abstract";
  const all = (await kv.getByPrefix(`decision:${pid}:${stage}:`)) || [];
  const adj = (await kv.getByPrefix(`adjudication:${pid}:${stage}:`)) || [];
  const adjByPaper = new Map(adj.map((a: any) => [a.paper_id, a]));
  const byPaper = new Map<string, any[]>();
  for (const d of all) {
    const arr = byPaper.get(d.paper_id) || [];
    arr.push(d);
    byPaper.set(d.paper_id, arr);
  }
  const conflicts: any[] = [];
  for (const [pid_, decisions] of byPaper) {
    if (decisions.length < 2) continue;
    const distinct = new Set(decisions.map((d: any) => d.decision));
    if (distinct.size <= 1) continue;
    if (adjByPaper.has(pid_)) continue;   // already adjudicated
    conflicts.push({ paper_id: pid_, decisions });
  }
  return c.json({ conflicts });
});

app.post("/make-server-7e4eb0f2/projects/:pid/adjudications", async (c) => {
  const pid = c.req.param("pid");
  const gate = await requireRole(c, pid, ["lead", "adjudicator"]);
  if (gate instanceof Response) return gate;
  try {
    const body = await c.req.json();
    const stage = body.stage || "abstract";
    if (!body.paper_id || !body.final_decision) return c.json({ error: "paper_id and final_decision required" }, 400);
    const key = `adjudication:${pid}:${stage}:${body.paper_id}`;
    const existing = await kv.get(key);
    const now = new Date().toISOString();
    const rec = {
      paper_id: body.paper_id,
      stage,
      adjudicator_user_id: gate.uid,
      final_decision: body.final_decision,
      rationale: body.rationale || "",
      decided_at: now,
      created_at: existing?.created_at || now,
    };
    await kv.set(key, rec);
    return c.json({ adjudication: rec });
  } catch (e) {
    return c.json({ error: `Failed to adjudicate: ${e}` }, 500);
  }
});

Deno.serve(app.fetch);
