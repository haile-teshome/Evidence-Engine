// Tests for the multi-reviewer projects client and PDF import.
//
// Run: pnpm test
//
// `apiFetch` is mocked, so these test request shaping and response handling for
// the collaboration API, plus `reproReport`, which is pure and produces the
// reproducibility record that accompanies a published review. A missing study
// or a wrong count in that record is a reporting error, not a crash.
import { describe, it, expect, beforeEach, vi } from "vitest";

import * as backendClient from "../backendClient";
import * as projects from "../projects";
import { isAccepted } from "../pdfImport";

type Call = { path: string; init: any };

function mockApi(handler: (c: Call) => any) {
  const calls: Call[] = [];
  vi.spyOn(backendClient, "apiFetch").mockImplementation((async (path: string, init: any = {}) => {
    calls.push({ path, init });
    return handler({ path, init });
  }) as any);
  return calls;
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

// --------------------------------------------------------------------------
// Request shaping
// --------------------------------------------------------------------------

describe("project CRUD", () => {
  it("lists projects", async () => {
    const calls = mockApi(() => ({ projects: [{ id: "p1", name: "Review" }] }));
    const out = await projects.listProjects();
    expect(calls[0].path).toContain("projects");
    expect(out).toHaveLength(1);
  });

  it("returns an empty list when the backend sends nothing", async () => {
    mockApi(() => ({}));
    await expect(projects.listProjects()).resolves.toEqual([]);
  });

  it("creates a project with the supplied name", async () => {
    const calls = mockApi(() => ({ project: { id: "p1", name: "New review" } }));
    const out = await projects.createProject({ name: "New review" } as any);
    expect(String(JSON.stringify(calls[0].init))).toContain("New review");
    expect(out.id).toBe("p1");
  });

  it("fetches one project with its members", async () => {
    mockApi(() => ({ project: { id: "p1", name: "R" }, members: [{ user_id: "u1", role: "lead" }] }));
    const out = await projects.getProject("p1");
    expect(out.project.id).toBe("p1");
    expect(out.members).toHaveLength(1);
  });

  it("puts the project id in the path, not the body", async () => {
    const calls = mockApi(() => ({ project: { id: "p1" }, members: [] }));
    await projects.getProject("p1");
    expect(calls[0].path).toContain("p1");
  });

  it("updates a project", async () => {
    const calls = mockApi(() => ({ project: { id: "p1", name: "Renamed" } }));
    await projects.updateProject("p1", { name: "Renamed" });
    expect(JSON.stringify(calls[0].init)).toContain("Renamed");
  });

  it("locks a project", async () => {
    const calls = mockApi(() => ({ project: { id: "p1", locked_at: "2026-09-09" } }));
    const out = await projects.lockProject("p1");
    expect(calls[0].path).toContain("lock");
    expect(out.locked_at).toBeTruthy();
  });

  it("propagates a backend error rather than returning a broken project", async () => {
    vi.spyOn(backendClient, "apiFetch").mockRejectedValue(new Error("Not a project member"));
    await expect(projects.getProject("p1")).rejects.toThrow(/Not a project member/);
  });
});

describe("membership and invites", () => {
  it("creates an invite with the requested role", async () => {
    const calls = mockApi(() => ({ invite: { token: "tok", role: "reviewer" } }));
    const out = await projects.createInvite("p1", "reviewer");
    expect(JSON.stringify(calls[0].init)).toContain("reviewer");
    expect(out.token).toBe("tok");
  });

  it("previews an invite without accepting it", async () => {
    const calls = mockApi(() => ({ invite: { token: "tok", role: "reviewer" },
                                   project: { id: "p1", name: "R" } }));
    const out = await projects.previewInvite("tok");
    expect(calls[0].path).toContain("tok");
    expect(out.project?.name).toBe("R");
    // A preview must not be a POST that joins the project as a side effect.
    expect(String(calls[0].init?.method ?? "GET").toUpperCase()).toBe("GET");
  });

  it("accepts an invite and reports the granted role", async () => {
    mockApi(() => ({ project_id: "p1", role: "reviewer" }));
    const out = await projects.acceptInvite("tok");
    expect(out.role).toBe("reviewer");
  });

  it("reports an already-member accept without treating it as an error", async () => {
    mockApi(() => ({ project_id: "p1", role: "reviewer", already_member: true }));
    await expect(projects.acceptInvite("tok")).resolves.toHaveProperty("already_member", true);
  });

  it("changes a member's role", async () => {
    const calls = mockApi(() => ({ member: { user_id: "u2", role: "adjudicator" } }));
    const out = await projects.setMemberRole("p1", "u2", "adjudicator");
    expect(JSON.stringify(calls[0].init)).toContain("adjudicator");
    expect(out.role).toBe("adjudicator");
  });

  it("adds and removes a participant", async () => {
    const calls = mockApi(() => ({ participant: { id: "rev1", name: "Dr Smith" } }));
    await projects.addParticipant("p1", "Dr Smith", "reviewer", 1);
    expect(JSON.stringify(calls[0].init)).toContain("Dr Smith");
    mockApi(() => ({}));
    await expect(projects.removeParticipant("p1", "rev1")).resolves.toBeUndefined();
  });
});

describe("papers, tags and assignment", () => {
  it("lists project papers", async () => {
    mockApi(() => ({ papers: [{ paper_id: "p1", Title: "T" }] }));
    await expect(projects.listProjectPapers("p1")).resolves.toHaveLength(1);
  });

  it("returns an empty list rather than undefined when there are no papers", async () => {
    mockApi(() => ({}));
    await expect(projects.listProjectPapers("p1")).resolves.toEqual([]);
  });

  it("sets papers and reports how many were stored", async () => {
    const calls = mockApi(() => ({ count: 3 }));
    const out = await projects.setProjectPapers("p1", [
      { paper_id: "a" }, { paper_id: "b" }, { paper_id: "c" },
    ] as any);
    expect(JSON.stringify(calls[0].init)).toContain("paper_id");
    expect(out).toBe(3);
  });

  it("sets project tags", async () => {
    mockApi(() => ({ tags: ["pilot"] }));
    await expect(projects.setProjectTags("p1", ["pilot"])).resolves.toEqual(["pilot"]);
  });

  it("sets per-paper tags", async () => {
    const calls = mockApi(() => ({ tags: ["maybe"] }));
    await projects.setPaperTags("p1", "paper1", ["maybe"]);
    expect(calls[0].path).toContain("paper1");
  });

  it("auto-assigns", async () => {
    const calls = mockApi(() => ({ assignments: [] }));
    await projects.autoAssign("p1", { strategy: "even", reviewers_per_paper: 2 } as any);
    expect(calls[0].path).toContain("auto-assign");
  });
});

describe("extraction", () => {
  it("reads the extraction template", async () => {
    mockApi(() => ({ fields: [{ id: "sample_size", label: "Sample size", type: "number" }] }));
    const out: any = await projects.getExtractionTemplate("p1");
    const fields = out.fields ?? out;
    expect(JSON.stringify(fields)).toContain("sample_size");
  });

  it("writes the extraction template", async () => {
    const calls = mockApi(() => ({ fields: [] }));
    await projects.setExtractionTemplate("p1", [
      { id: "sample_size", label: "Sample size", type: "number" } as any,
    ]);
    expect(JSON.stringify(calls[0].init)).toContain("sample_size");
  });

  it("lists extraction conflicts", async () => {
    mockApi(() => ({ conflicts: [{ paper_id: "p1", fields: ["sample_size"] }] }));
    await expect(projects.listExtractionConflicts("p1")).resolves.toHaveLength(1);
  });

  it("returns no conflicts rather than undefined", async () => {
    mockApi(() => ({}));
    await expect(projects.listExtractionConflicts("p1")).resolves.toEqual([]);
  });
});

// --------------------------------------------------------------------------
// reproReport: pure, and it is what accompanies a published review.
// --------------------------------------------------------------------------

const bundle = (over: any = {}) => ({
  bundle_version: 1,
  generated_at: "2026-09-09T00:00:00Z",
  project: {
    id: "p1", name: "Dental AI review", screening_mode: "dual",
    pico: { population: "adults", intervention: "AI", comparator: "none", outcome: "accuracy" },
  },
  papers: [{ paper_id: "s1", Title: "Study one" }, { paper_id: "s2", Title: "Study two" }],
  decisions: [
    { paper_id: "s1", reviewer_user_id: "u1", decision: "include", stage: "abstract" },
    { paper_id: "s1", reviewer_user_id: "u2", decision: "exclude", stage: "abstract" },
  ],
  adjudications: [{ paper_id: "s1", final_decision: "include", rationale: "lead call" }],
  extraction_template: [{ id: "sample_size", label: "Sample size", type: "number" }],
  extraction_finals: [{ paper_id: "s1", values: { sample_size: 128 } }],
  ...over,
}) as any;

describe("reproReport", () => {
  it("produces markdown with a title", () => {
    expect(projects.reproReport(bundle())).toMatch(/^# /m);
  });

  it("names the review", () => {
    expect(projects.reproReport(bundle())).toContain("Dental AI review");
  });

  it("records the screening mode, which determines how decisions were made", () => {
    expect(projects.reproReport(bundle())).toContain("dual");
  });

  it("records the generation timestamp", () => {
    expect(projects.reproReport(bundle())).toContain("2026-09-09");
  });

  it("reports every PICO element", () => {
    const out = projects.reproReport(bundle());
    for (const label of ["Population", "Intervention", "Comparator", "Outcome"]) {
      expect(out).toContain(label);
    }
  });

  it("marks an absent PICO element rather than leaving a blank line", () => {
    const out = projects.reproReport(bundle({
      project: { name: "R", screening_mode: "single", pico: {} },
    }));
    expect(out).toContain("n/a");
  });

  it("handles a bundle with no decisions", () => {
    expect(() => projects.reproReport(bundle({ decisions: [], adjudications: [] }))).not.toThrow();
  });

  it("handles an almost-empty bundle without crashing", () => {
    expect(typeof projects.reproReport({ generated_at: "x" } as any)).toBe("string");
  });

  it("handles missing collections entirely", () => {
    expect(() => projects.reproReport({
      project: { name: "R" }, generated_at: "x", bundle_version: 1,
    } as any)).not.toThrow();
  });

  it("is deterministic for the same bundle", () => {
    const b = bundle();
    expect(projects.reproReport(b)).toBe(projects.reproReport(b));
  });
});

// --------------------------------------------------------------------------
// PDF import gate
// --------------------------------------------------------------------------

const file = (name: string, type = "") => new File(["x"], name, { type });

describe("isAccepted", () => {
  it.each(["paper.pdf", "Paper.PDF", "notes.txt", "table.csv"])(
    "accepts the importable file %s",
    (name) => expect(isAccepted(file(name))).toBe(true),
  );

  it.each(["image.png", "video.mp4", "archive.zip", "binary.exe"])(
    "rejects the non-importable file %s",
    (name) => expect(isAccepted(file(name))).toBe(false),
  );

  it("accepts by MIME type when the extension is missing", () => {
    expect(isAccepted(file("document", "application/pdf"))).toBe(true);
  });

  it("does not crash on a file with no name", () => {
    expect(() => isAccepted(file(""))).not.toThrow();
  });
});
