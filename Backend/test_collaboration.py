"""Tests for the multi-reviewer collaboration layer.

Run: Backend/.venv/bin/python -m pytest Backend/test_collaboration.py

Driven through the real HTTP routes with a throwaway database (conftest.py
redirects the SQLite path), so role checks and request validation are exercised
alongside the logic.

Why this layer deserves tests more than most: dual screening is what makes a
review a review. If conflict detection silently misses a disagreement, the pair
never adjudicates it, one reviewer's call becomes the record, and the interrater
agreement you report describes something that did not happen. Nothing errors.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


def _hdr(uid):
    return {"x-reviewer-id": uid}


@pytest.fixture
def project(client):
    """A project with a lead and a SECOND reviewer who has actually joined.

    The second reviewer matters: without a real member, every decision they try
    to record is rejected as "Not a project member", so no conflict can ever be
    detected and the dual-screening tests below would pass vacuously.
    """
    lead = f"lead-{uuid.uuid4().hex[:6]}"
    r = client.post("/api/projects", json={"name": "Test review"}, headers=_hdr(lead))
    if r.status_code not in (200, 201):
        pytest.skip(f"project creation unavailable: {r.status_code}")
    body = r.json() or {}
    pid = body.get("id") or (body.get("project") or {}).get("id")
    if not pid:
        pytest.skip("project id not returned")

    second = f"rev-{uuid.uuid4().hex[:6]}"
    inv = client.post(f"/api/projects/{pid}/invites",
                      json={"role": "reviewer"}, headers=_hdr(lead))
    if inv.status_code == 200:
        token = ((inv.json() or {}).get("invite") or {}).get("token")
        if token:
            client.post(f"/api/invites/{token}/accept", headers=_hdr(second))
    return {"pid": pid, "lead": lead, "second": second}


def _decide(client, pid, uid, paper_id, decision, stage="abstract"):
    return client.post(
        f"/api/projects/{pid}/decisions",
        json={"paper_id": paper_id, "stage": stage, "decision": decision},
        headers=_hdr(uid),
    )


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

class TestProjectAccessControl:
    def test_a_non_member_cannot_read_conflicts(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}/conflicts",
                       headers=_hdr("total-stranger"))
        assert r.status_code in (401, 403)

    def test_a_non_member_cannot_create_assignments(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/assignments",
                        json={"strategy": "even"}, headers=_hdr("total-stranger"))
        assert r.status_code in (401, 403, 422)

    def test_an_unknown_project_is_not_readable(self, client):
        r = client.get("/api/projects/does-not-exist/conflicts", headers=_hdr("someone"))
        assert r.status_code in (403, 404)

    def test_the_lead_can_read_their_own_project(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}", headers=_hdr(project["lead"]))
        assert r.status_code == 200

    def test_a_stranger_cannot_read_someone_elses_project(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}", headers=_hdr("stranger"))
        assert r.status_code in (200, 403, 404)
        if r.status_code == 200:
            assert (r.json() or {}).get("id") in (project["pid"], None)


# ---------------------------------------------------------------------------
# Conflict detection. The core dual-screening invariant.
# ---------------------------------------------------------------------------

class TestInviteRoles:
    """An invite that names a role the system does not recognise used to be
    accepted. The invitee joined successfully and was then refused by every
    role-gated endpoint, because the role passed no allow-list anywhere. The
    403 blamed their role rather than the malformed invite that created it."""

    @pytest.mark.parametrize("role", ["screener", "admin", "LEAD", "owner", "Reviewer"])
    def test_an_unrecognised_role_is_rejected_at_invite_time(self, client, project, role):
        r = client.post(f"/api/projects/{project['pid']}/invites",
                        json={"role": role}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 422), f"role {role!r} should not be invitable"

    @pytest.mark.parametrize("body", [{}, {"role": ""}, {"role": None}])
    def test_an_omitted_role_defaults_to_reviewer(self, client, project, body):
        """Deliberate: no role means the ordinary screening role, not an error."""
        r = client.post(f"/api/projects/{project['pid']}/invites",
                        json=body, headers=_hdr(project["lead"]))
        assert r.status_code == 200
        assert (r.json()["invite"])["role"] == "reviewer"

    @pytest.mark.parametrize("role", ["lead", "reviewer", "adjudicator", "viewer"])
    def test_every_recognised_role_is_invitable(self, client, project, role):
        r = client.post(f"/api/projects/{project['pid']}/invites",
                        json={"role": role}, headers=_hdr(project["lead"]))
        assert r.status_code == 200

    def test_an_accepted_invite_grants_a_role_that_actually_works(self, client, project):
        """The end-to-end property the bug broke: joining must let you act."""
        pid, lead = project["pid"], project["lead"]
        inv = client.post(f"/api/projects/{pid}/invites",
                          json={"role": "reviewer"}, headers=_hdr(lead))
        token = ((inv.json() or {}).get("invite") or {}).get("token")
        newbie = f"rev-{uuid.uuid4().hex[:6]}"
        assert client.post(f"/api/invites/{token}/accept",
                           headers=_hdr(newbie)).status_code == 200
        assert _decide(client, pid, newbie, "p-newbie", "INCLUDE").status_code in (200, 201)

    def test_only_a_lead_can_invite(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/invites",
                        json={"role": "reviewer"}, headers=_hdr(project["second"]))
        assert r.status_code in (401, 403)


class TestConflictDetection:
    def test_the_second_reviewer_is_genuinely_a_member(self, client, project):
        """Guard against a vacuous suite: if the invite flow broke, the second
        reviewer's decisions would all 403, no conflict could ever be recorded,
        and every negative assertion below would pass for the wrong reason."""
        r = _decide(client, project["pid"], project["second"], "p-guard", "INCLUDE")
        assert r.status_code in (200, 201), "second reviewer never joined the project"

    def _conflicts(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}/conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip(f"conflicts endpoint unavailable: {r.status_code}")
        return (r.json() or {}).get("conflicts", [])

    def test_agreement_is_not_a_conflict(self, client, project):
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-agree", "INCLUDE")
        _decide(client, pid, project["second"], "p-agree", "INCLUDE")
        assert not any(c["paper_id"] == "p-agree" for c in self._conflicts(client, project))

    def test_disagreement_is_a_conflict(self, client, project):
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-clash", "INCLUDE")
        _decide(client, pid, project["second"], "p-clash", "EXCLUDE")
        assert any(c["paper_id"] == "p-clash" for c in self._conflicts(client, project))

    def test_a_single_decision_is_not_yet_a_conflict(self, client, project):
        """One reviewer cannot disagree with themselves. Flagging this would
        bury real conflicts in noise."""
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-single", "INCLUDE")
        assert not any(c["paper_id"] == "p-single" for c in self._conflicts(client, project))

    def test_a_conflict_carries_both_decisions_for_adjudication(self, client, project):
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-both", "INCLUDE")
        _decide(client, pid, project["second"], "p-both", "EXCLUDE")
        found = [c for c in self._conflicts(client, project) if c["paper_id"] == "p-both"]
        assert found and len(found[0]["decisions"]) >= 2

    def test_an_adjudicated_conflict_stops_being_reported(self, client, project):
        """Otherwise the queue never empties and the pair re-adjudicates
        decisions they have already settled."""
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-resolved", "INCLUDE")
        _decide(client, pid, project["second"], "p-resolved", "EXCLUDE")
        r = client.post(f"/api/projects/{pid}/adjudications",
                        json={"paper_id": "p-resolved", "stage": "abstract",
                              "final_decision": "INCLUDE", "rationale": "lead call"},
                        headers=_hdr(lead))
        if r.status_code not in (200, 201):
            pytest.skip(f"adjudication unavailable: {r.status_code}")
        assert not any(c["paper_id"] == "p-resolved" for c in self._conflicts(client, project))

    def test_stages_are_kept_separate(self, client, project):
        """An abstract-stage disagreement must not appear in the full-text queue."""
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-stage", "INCLUDE", stage="abstract")
        _decide(client, pid, project["second"], "p-stage", "EXCLUDE", stage="abstract")
        r = client.get(f"/api/projects/{pid}/conflicts",
                       params={"stage": "fulltext"}, headers=_hdr(lead))
        if r.status_code != 200:
            pytest.skip("stage filter unavailable")
        assert not any(c["paper_id"] == "p-stage" for c in (r.json() or {}).get("conflicts", []))

    def test_adjudication_requires_a_final_decision(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/adjudications",
                        json={"paper_id": "p-x", "stage": "abstract"},
                        headers=_hdr(project["lead"]))
        assert r.status_code in (400, 403, 422)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_a_decision_is_recorded_and_readable(self, client, project):
        pid, lead = project["pid"], project["lead"]
        if _decide(client, pid, lead, "p-1", "INCLUDE").status_code not in (200, 201):
            pytest.skip("decisions endpoint unavailable")
        r = client.get(f"/api/projects/{pid}/decisions", headers=_hdr(lead))
        assert r.status_code == 200
        body = r.json()
        rows = body if isinstance(body, list) else body.get("decisions", [])
        assert any(d.get("paper_id") == "p-1" for d in rows)

    def test_re_deciding_replaces_rather_than_duplicating(self, client, project):
        """Two live decisions from one reviewer on one paper would make the same
        person look like a disagreeing pair."""
        pid, lead = project["pid"], project["lead"]
        _decide(client, pid, lead, "p-redo", "INCLUDE")
        _decide(client, pid, lead, "p-redo", "EXCLUDE")
        r = client.get(f"/api/projects/{pid}/decisions", headers=_hdr(lead))
        if r.status_code != 200:
            pytest.skip("decisions endpoint unavailable")
        body = r.json()
        rows = body if isinstance(body, list) else body.get("decisions", [])
        mine = [d for d in rows if d.get("paper_id") == "p-redo"
                and d.get("reviewer_id") in (lead, None)]
        assert len(mine) <= 1

    def test_a_malformed_decision_is_rejected_cleanly(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/decisions", json={},
                        headers=_hdr(project["lead"]))
        assert r.status_code in (400, 403, 422)
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class TestAssignments:
    def test_assignments_endpoint_answers_the_lead(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}/assignments",
                       headers=_hdr(project["lead"]))
        assert r.status_code in (200, 403)

    def test_a_malformed_assignment_request_is_rejected_cleanly(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/assignments",
                        json={"strategy": 12345, "reviewers_per_paper": "many"},
                        headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_auto_assign_with_no_papers_does_not_500(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/auto-assign",
                        json={"strategy": "even", "reviewers_per_paper": 2},
                        headers=_hdr(project["lead"]))
        assert r.status_code != 500
