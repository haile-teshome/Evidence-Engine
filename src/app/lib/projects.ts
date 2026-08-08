// Multi-reviewer projects API client + types.
//
// A "project" is a shared systematic-review workspace with multiple members.
// Each member screens the same corpus independently. The platform supports
// three screening modes:
//   - single        : single-user, AI as accelerator (legacy behavior)
//   - dual          : two+ reviewers, unblinded (faster but biased)
//   - dual_blinded  : two+ reviewers, each can only see their own decisions
//                     until they have decided on a given paper; the system
//                     then surfaces the conflict to the adjudicator role.
//
// All endpoints go through the Hono edge function (apiFetch).

import { apiFetch } from "./backendClient";

export type ProjectRole = "lead" | "reviewer" | "adjudicator" | "viewer";
export type ScreeningMode = "single" | "dual" | "dual_blinded";
export type Stage = "abstract" | "fulltext";
export type DecisionValue = "include" | "exclude" | "maybe";

export type Pico = {
  population: string;
  intervention: string;
  comparator: string;
  outcome: string;
};

export type Project = {
  id: string;
  name: string;
  owner_user_id: string;
  pico: Pico;
  inclusion: string[];
  exclusion: string[];
  screening_mode: ScreeningMode;
  visibility: "private" | "invite" | "link";
  locked_at: string | null;
  created_at: string;
  updated_at: string;
  my_role?: ProjectRole;
};

export type ProjectMember = {
  project_id: string;
  user_id: string;
  role: ProjectRole;
  joined_at: string;
};

export type Invite = {
  token: string;
  project_id: string;
  role: ProjectRole;
  created_by: string;
  created_at: string;
  expires_at: string | null;
  used_at: string | null;
  used_by: string | null;
};

export type ProjectPaper = {
  paper_id: string;
  title: string;
  abstract: string;
  source: string;
  url?: string;
  doi?: string;
};

export type Decision = {
  paper_id: string;
  stage: Stage;
  reviewer_user_id: string;
  decision: DecisionValue;
  reason: string;
  per_pico_verdict?: any;
  ai_decision?: DecisionValue | null;
  is_override: boolean;
  decided_at: string;
  created_at: string;
};

/** A blinding-stripped decision exposed to reviewers who haven't yet decided. */
export type DecisionSummary = Pick<
  Decision,
  "paper_id" | "stage" | "reviewer_user_id" | "decision" | "decided_at"
>;

export type Adjudication = {
  paper_id: string;
  stage: Stage;
  adjudicator_user_id: string;
  final_decision: DecisionValue;
  rationale: string;
  decided_at: string;
  created_at: string;
};

export type Conflict = {
  paper_id: string;
  decisions: Decision[];
};

export type AssignmentStrategy = "full_overlap" | "split" | "custom";

export type Assignment = {
  project_id: string;
  paper_id: string;
  user_id: string;
  assigned_at: string;
  strategy: AssignmentStrategy;
};

// ---- Project CRUD --------------------------------------------------------

export async function listProjects(): Promise<Project[]> {
  const r = await apiFetch("/projects");
  return r.projects || [];
}

export async function createProject(input: {
  name: string;
  pico?: Pico;
  inclusion?: string[];
  exclusion?: string[];
  screening_mode?: ScreeningMode;
  visibility?: Project["visibility"];
}): Promise<Project> {
  const r = await apiFetch("/projects", { method: "POST", body: JSON.stringify(input) });
  return r.project;
}

export async function getProject(pid: string): Promise<{ project: Project; members: ProjectMember[] }> {
  const r = await apiFetch(`/projects/${pid}`);
  return { project: r.project, members: r.members || [] };
}

export async function updateProject(pid: string, patch: Partial<Project>): Promise<Project> {
  const r = await apiFetch(`/projects/${pid}`, { method: "PUT", body: JSON.stringify(patch) });
  return r.project;
}

export async function lockProject(pid: string): Promise<Project> {
  const r = await apiFetch(`/projects/${pid}/lock`, { method: "POST" });
  return r.project;
}

// ---- Members + invites ---------------------------------------------------

export async function setMemberRole(pid: string, userId: string, role: ProjectRole): Promise<ProjectMember> {
  const r = await apiFetch(`/projects/${pid}/members/${userId}/role`, {
    method: "PUT",
    body: JSON.stringify({ role }),
  });
  return r.member;
}

export async function createInvite(pid: string, role: ProjectRole, expiresAt?: string): Promise<Invite> {
  const r = await apiFetch(`/projects/${pid}/invites`, {
    method: "POST",
    body: JSON.stringify({ role, expires_at: expiresAt ?? null }),
  });
  return r.invite;
}

export async function previewInvite(token: string): Promise<{ invite: Invite; project: { id: string; name: string } | null }> {
  const r = await apiFetch(`/invites/${token}`);
  return r;
}

export async function acceptInvite(token: string): Promise<{ project_id: string; role: ProjectRole; already_member?: boolean }> {
  const r = await apiFetch(`/invites/${token}/accept`, { method: "POST" });
  return r;
}

// ---- Papers --------------------------------------------------------------

export async function listProjectPapers(pid: string): Promise<ProjectPaper[]> {
  const r = await apiFetch(`/projects/${pid}/papers`);
  return r.papers || [];
}

export async function setProjectPapers(pid: string, papers: ProjectPaper[]): Promise<number> {
  const r = await apiFetch(`/projects/${pid}/papers`, {
    method: "PUT",
    body: JSON.stringify({ papers }),
  });
  return r.count ?? 0;
}

// ---- Participants (author-defined reviewer slots) ------------------------

export type Participant = { id: string; project_id: string; name: string; role: ProjectRole; weight: number; created_at: string };

export async function listParticipants(pid: string): Promise<Participant[]> {
  const r = await apiFetch(`/projects/${pid}/participants`);
  return r.participants || [];
}
export async function addParticipant(pid: string, name: string, role: ProjectRole = "reviewer", weight = 1): Promise<Participant> {
  const r = await apiFetch(`/projects/${pid}/participants`, { method: "POST", body: JSON.stringify({ name, role, weight }) });
  return r.participant;
}
export async function updateParticipant(pid: string, partId: string, patch: { name?: string; role?: ProjectRole; weight?: number }): Promise<Participant> {
  const r = await apiFetch(`/projects/${pid}/participants/${partId}`, { method: "PUT", body: JSON.stringify(patch) });
  return r.participant;
}
export async function removeParticipant(pid: string, partId: string): Promise<void> {
  await apiFetch(`/projects/${pid}/participants/${partId}`, { method: "DELETE" });
}

// ---- Granular auto-assignment -------------------------------------------

export type AutoAssignResult = { strategy: string; assigned: number; per_reviewer: { id: string; name: string; count: number }[]; calibration: number; papers: number };
export async function autoAssign(
  pid: string,
  body: { strategy: "dual" | "overlap" | "weighted" | "manual"; overlap_pct?: number; reviewers_per_paper?: number; include_calibration?: boolean; manual?: { paper_id: string; participant_ids: string[] }[] },
): Promise<AutoAssignResult> {
  return apiFetch(`/projects/${pid}/auto-assign`, { method: "POST", body: JSON.stringify(body) });
}

// ---- Tags ---------------------------------------------------------------

export async function getProjectTags(pid: string): Promise<{ tags: string[]; paper_tags: Record<string, string[]> }> {
  const r = await apiFetch(`/projects/${pid}/tags`);
  return { tags: r.tags || [], paper_tags: r.paper_tags || {} };
}
export async function setProjectTags(pid: string, tags: string[]): Promise<string[]> {
  const r = await apiFetch(`/projects/${pid}/tags`, { method: "PUT", body: JSON.stringify({ tags }) });
  return r.tags || [];
}
export async function setPaperTags(pid: string, paperId: string, tags: string[]): Promise<string[]> {
  const r = await apiFetch(`/projects/${pid}/papers/${paperId}/tags`, { method: "PUT", body: JSON.stringify({ paper_id: paperId, tags }) });
  return r.tags || [];
}

// ---- Calibration set ----------------------------------------------------

export async function setCalibration(
  pid: string,
  items: { paper_id: string; gold?: string | null; rationale?: string; is_calibration?: boolean }[],
): Promise<string[]> {
  const r = await apiFetch(`/projects/${pid}/calibration`, { method: "PUT", body: JSON.stringify(items) });
  return r.calibration || [];
}

// ---- Data extraction (dual, independent) --------------------------------

export type ExtractionFieldType = "text" | "number" | "category" | "date";
export type ExtractionField = { id: string; label: string; group: string; type: ExtractionFieldType; options: string[] };
export type ExtractionTemplate = { fields: ExtractionField[]; is_default: boolean };
export type Extraction = {
  paper_id: string; reviewer_user_id: string; values: Record<string, any>;
  submitted: boolean; ai_prefilled?: boolean; extracted_at: string; created_at?: string;
};
export type ExtractionFinal = { paper_id: string; values: Record<string, any>; reconciled_by: string; rationale: string; reconciled_at: string };
export type ExtractionConflict = { paper_id: string; fields: string[]; extractions: Extraction[] };

export async function getExtractionTemplate(pid: string): Promise<ExtractionTemplate> {
  const r = await apiFetch(`/projects/${pid}/extraction-template`);
  return { fields: r.fields || [], is_default: !!r.is_default };
}
export async function setExtractionTemplate(pid: string, fields: ExtractionField[]): Promise<ExtractionField[]> {
  const r = await apiFetch(`/projects/${pid}/extraction-template`, { method: "PUT", body: JSON.stringify({ fields }) });
  return r.fields || [];
}
export async function listExtractions(pid: string): Promise<{ extractions: Extraction[]; finals: ExtractionFinal[]; blinded?: boolean }> {
  const r = await apiFetch(`/projects/${pid}/extractions`);
  return { extractions: r.extractions || [], finals: r.finals || [], blinded: !!r.blinded };
}
export async function submitExtraction(
  pid: string,
  input: { paper_id: string; values: Record<string, any>; submitted?: boolean; ai_prefilled?: boolean },
): Promise<Extraction> {
  const r = await apiFetch(`/projects/${pid}/extractions`, { method: "POST", body: JSON.stringify(input) });
  return r.extraction;
}
export async function listExtractionConflicts(pid: string): Promise<ExtractionConflict[]> {
  const r = await apiFetch(`/projects/${pid}/extraction-conflicts`);
  return r.conflicts || [];
}
export async function reconcileExtraction(
  pid: string,
  input: { paper_id: string; values: Record<string, any>; rationale?: string },
): Promise<ExtractionFinal> {
  const r = await apiFetch(`/projects/${pid}/extraction-reconciliations`, { method: "POST", body: JSON.stringify(input) });
  return r.final;
}

// ---- Decisions, conflicts, adjudications --------------------------------

export async function listDecisions(
  pid: string,
  stage: Stage = "abstract",
): Promise<{ decisions: (Decision | DecisionSummary)[]; adjudications: Adjudication[]; blinded?: boolean }> {
  const r = await apiFetch(`/projects/${pid}/decisions?stage=${stage}`);
  return { decisions: r.decisions || [], adjudications: r.adjudications || [], blinded: !!r.blinded };
}

export async function writeDecision(pid: string, input: {
  paper_id: string;
  stage: Stage;
  decision: DecisionValue;
  reason?: string;
  per_pico_verdict?: any;
  ai_decision?: DecisionValue | null;
  is_override?: boolean;
}): Promise<Decision> {
  const r = await apiFetch(`/projects/${pid}/decisions`, {
    method: "POST",
    body: JSON.stringify(input),
  });
  return r.decision;
}

export async function listConflicts(pid: string, stage: Stage = "abstract"): Promise<Conflict[]> {
  const r = await apiFetch(`/projects/${pid}/conflicts?stage=${stage}`);
  return r.conflicts || [];
}

export async function writeAdjudication(pid: string, input: {
  paper_id: string;
  stage: Stage;
  final_decision: DecisionValue;
  rationale?: string;
}): Promise<Adjudication> {
  const r = await apiFetch(`/projects/${pid}/adjudications`, {
    method: "POST",
    body: JSON.stringify(input),
  });
  return r.adjudication;
}

// ---- Assignments ---------------------------------------------------------

export async function assignPapers(pid: string, input: {
  strategy: AssignmentStrategy;
  reviewers_per_paper?: number;
  custom?: { paper_id: string; user_ids: string[] }[];
}): Promise<{ strategy: AssignmentStrategy; assigned: number; papers: number; reviewers: number }> {
  const r = await apiFetch(`/projects/${pid}/assignments`, {
    method: "POST",
    body: JSON.stringify(input),
  });
  return r;
}

export async function listAssignments(pid: string): Promise<Assignment[]> {
  const r = await apiFetch(`/projects/${pid}/assignments`);
  return r.assignments || [];
}

export async function clearAssignments(pid: string): Promise<number> {
  const r = await apiFetch(`/projects/${pid}/assignments`, { method: "DELETE" });
  return r.cleared ?? 0;
}

// ---- Helpers -------------------------------------------------------------

/** Resolve the effective per-paper decision for a project, honouring
 *  adjudications first, then a unanimous consensus across reviewers,
 *  then null (conflict pending), then the AI's cached decision. */
export function effectiveProjectDecision(
  paperId: string,
  decisions: (Decision | DecisionSummary)[],
  adjudications: Adjudication[],
  aiDecisionFallback?: DecisionValue,
): { value: DecisionValue | null; source: "adjudication" | "consensus" | "conflict" | "ai" | "none" } {
  const adj = adjudications.find(a => a.paper_id === paperId);
  if (adj) return { value: adj.final_decision, source: "adjudication" };
  const myDecisions = decisions.filter(d => d.paper_id === paperId);
  if (myDecisions.length === 0) {
    if (aiDecisionFallback) return { value: aiDecisionFallback, source: "ai" };
    return { value: null, source: "none" };
  }
  const distinct = new Set(myDecisions.map(d => d.decision));
  if (distinct.size === 1) {
    return { value: [...distinct][0], source: "consensus" };
  }
  return { value: null, source: "conflict" };
}
