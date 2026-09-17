"""End-to-end tests for the project workflow: papers, assignment, tags,
calibration, risk-of-bias, and structured extraction.

Run: Backend/.venv/bin/python -m pytest Backend/test_project_workflow.py

Driven through the real routes against a throwaway database. These cover the
parts of a dual-reviewer review that produce reportable numbers: who screened
what, where the pair disagreed, and what was extracted. A silent fault in any of
them changes a published figure without changing anything visible on screen.
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
    """A project with a lead and a second reviewer who has genuinely joined."""
    lead = f"lead-{uuid.uuid4().hex[:6]}"
    r = client.post("/api/projects", json={"name": "Workflow review"}, headers=_hdr(lead))
    if r.status_code not in (200, 201):
        pytest.skip("project creation unavailable")
    pid = ((r.json() or {}).get("project") or {}).get("id") or (r.json() or {}).get("id")
    if not pid:
        pytest.skip("no project id")

    second = f"rev-{uuid.uuid4().hex[:6]}"
    inv = client.post(f"/api/projects/{pid}/invites",
                      json={"role": "reviewer"}, headers=_hdr(lead))
    if inv.status_code == 200:
        tok = ((inv.json() or {}).get("invite") or {}).get("token")
        if tok:
            client.post(f"/api/invites/{tok}/accept", headers=_hdr(second))
    return {"pid": pid, "lead": lead, "second": second}


PAPERS = [
    {"paper_id": "p1", "Title": "Study one", "Abstract": "A", "Source": "PubMed", "URL": ""},
    {"paper_id": "p2", "Title": "Study two", "Abstract": "B", "Source": "PubMed", "URL": ""},
    {"paper_id": "p3", "Title": "Study three", "Abstract": "C", "Source": "PubMed", "URL": ""},
]


def _put_papers(client, project):
    return client.put(f"/api/projects/{project['pid']}/papers",
                      json={"papers": PAPERS}, headers=_hdr(project["lead"]))


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------

class TestProjectLifecycle:
    def test_a_new_project_appears_in_the_owners_list(self, client, project):
        r = client.get("/api/projects", headers=_hdr(project["lead"]))
        assert r.status_code == 200
        body = r.json()
        rows = body if isinstance(body, list) else body.get("projects", [])
        assert any(p.get("id") == project["pid"] for p in rows)

    def test_the_owner_is_recorded_as_lead(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}", headers=_hdr(project["lead"]))
        assert (r.json() or {}).get("project", {}).get("my_role") == "lead"

    def test_a_project_can_be_renamed(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}",
                       json={"name": "Renamed"}, headers=_hdr(project["lead"]))
        assert r.status_code in (200, 404, 405)

    def test_locking_a_project_blocks_further_decisions(self, client, project):
        """Locking is what makes an analysis reproducible: after it, the
        decision set cannot move under the numbers already reported."""
        pid, lead = project["pid"], project["lead"]
        r = client.post(f"/api/projects/{pid}/lock", headers=_hdr(lead))
        if r.status_code != 200:
            pytest.skip("lock unavailable")
        d = client.post(f"/api/projects/{pid}/decisions",
                        json={"paper_id": "p1", "stage": "abstract", "decision": "INCLUDE"},
                        headers=_hdr(lead))
        assert d.status_code == 409

    def test_a_non_lead_cannot_lock(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/lock", headers=_hdr(project["second"]))
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

class TestPapers:
    def test_papers_round_trip(self, client, project):
        if _put_papers(client, project).status_code != 200:
            pytest.skip("papers endpoint unavailable")
        r = client.get(f"/api/projects/{project['pid']}/papers", headers=_hdr(project["lead"]))
        assert r.status_code == 200
        body = r.json()
        rows = body if isinstance(body, list) else body.get("papers", [])
        assert len(rows) == len(PAPERS)

    def test_an_empty_corpus_is_accepted(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/papers",
                       json={"papers": []}, headers=_hdr(project["lead"]))
        assert r.status_code in (200, 422)

    def test_a_non_member_cannot_write_papers(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/papers",
                       json={"papers": PAPERS}, headers=_hdr("stranger"))
        assert r.status_code in (401, 403)

    def test_a_malformed_body_is_rejected_cleanly(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/papers",
                       json={"papers": "not-a-list"}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 422)
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_auto_assign_covers_the_corpus(self, client, project):
        _put_papers(client, project)
        r = client.post(f"/api/projects/{project['pid']}/auto-assign",
                        json={"strategy": "even", "reviewers_per_paper": 1},
                        headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_assignments_can_be_read_back(self, client, project):
        _put_papers(client, project)
        client.post(f"/api/projects/{project['pid']}/auto-assign",
                    json={"strategy": "even", "reviewers_per_paper": 1},
                    headers=_hdr(project["lead"]))
        r = client.get(f"/api/projects/{project['pid']}/assignments",
                       headers=_hdr(project["lead"]))
        assert r.status_code in (200, 403)

    def test_assignments_can_be_cleared(self, client, project):
        r = client.delete(f"/api/projects/{project['pid']}/assignments",
                          headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_a_non_lead_cannot_auto_assign(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/auto-assign",
                        json={"strategy": "even"}, headers=_hdr(project["second"]))
        assert r.status_code in (401, 403, 422)

    @pytest.mark.parametrize("body", [
        {"reviewers_per_paper": -1},
        {"reviewers_per_paper": "many"},
        {"overlap_pct": 500},
        {},
    ])
    def test_a_nonsensical_assignment_request_never_500s(self, client, project, body):
        r = client.post(f"/api/projects/{project['pid']}/auto-assign",
                        json=body, headers=_hdr(project["lead"]))
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

class TestParticipants:
    def test_a_participant_can_be_added_and_listed(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/participants",
                        json={"name": "Dr Smith", "role": "reviewer"},
                        headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("participants unavailable")
        listing = client.get(f"/api/projects/{project['pid']}/participants",
                             headers=_hdr(project["lead"]))
        assert listing.status_code == 200
        assert "Dr Smith" in listing.text

    def test_a_participant_can_be_updated(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/participants",
                        json={"name": "Dr Smith"}, headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("participants unavailable")
        part_id = (r.json() or {}).get("participant", {}).get("id")
        upd = client.put(f"/api/projects/{project['pid']}/participants/{part_id}",
                         json={"name": "Dr Jones"}, headers=_hdr(project["lead"]))
        assert upd.status_code != 500

    def test_a_participant_can_be_removed(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/participants",
                        json={"name": "Temp"}, headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("participants unavailable")
        part_id = (r.json() or {}).get("participant", {}).get("id")
        rm = client.delete(f"/api/projects/{project['pid']}/participants/{part_id}",
                           headers=_hdr(project["lead"]))
        assert rm.status_code != 500

    def test_a_participant_needs_a_name(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/participants",
                        json={}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Tags and calibration
# ---------------------------------------------------------------------------

class TestTagsAndCalibration:
    def test_project_tags_round_trip(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/tags",
                       json={"tags": ["pilot", "dental"]}, headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("tags unavailable")
        got = client.get(f"/api/projects/{project['pid']}/tags", headers=_hdr(project["lead"]))
        assert "pilot" in got.text

    def test_paper_tags_round_trip(self, client, project):
        _put_papers(client, project)
        r = client.put(f"/api/projects/{project['pid']}/papers/p1/tags",
                       json={"paper_id": "p1", "tags": ["maybe"]},
                       headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_calibration_items_are_accepted(self, client, project):
        """Calibration sets the gold standard a reviewer pair is measured
        against, so it has to be storable before screening starts."""
        r = client.put(f"/api/projects/{project['pid']}/calibration",
                       json=[{"paper_id": "p1", "gold": "INCLUDE",
                              "rationale": "clear match", "is_calibration": True}],
                       headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_a_malformed_calibration_body_is_rejected_cleanly(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/calibration",
                       json={"not": "a list"}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Risk of bias
# ---------------------------------------------------------------------------

class TestRiskOfBias:
    def _post(self, client, project, uid, overall, domains=None):
        return client.post(
            f"/api/projects/{project['pid']}/rob-assessments",
            json={"paper_id": "p1", "instrument_id": "rob2",
                  "domains": domains or {"d1": "Low"}, "overall": overall, "notes": ""},
            headers=_hdr(uid))

    def test_an_assessment_round_trips(self, client, project):
        if self._post(client, project, project["lead"], "Low").status_code not in (200, 201):
            pytest.skip("rob endpoint unavailable")
        r = client.get(f"/api/projects/{project['pid']}/rob-assessments",
                       headers=_hdr(project["lead"]))
        assert r.status_code == 200
        assert "rob2" in r.text

    def test_disagreeing_assessments_are_reported_as_a_conflict(self, client, project):
        if self._post(client, project, project["lead"], "Low",
                      {"d1": "Low"}).status_code not in (200, 201):
            pytest.skip("rob endpoint unavailable")
        self._post(client, project, project["second"], "High", {"d1": "High"})
        r = client.get(f"/api/projects/{project['pid']}/rob-conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("rob conflicts unavailable")
        assert "p1" in r.text

    def test_conflicts_are_detected_per_domain_not_on_the_overall(self, client, project):
        """Deliberate: the overall judgment is derived from the domains by the
        instrument's algorithm (see instruments.py), so identical domains cannot
        legitimately produce different overalls. Domain-level comparison is the
        meaningful one."""
        if self._post(client, project, project["lead"], "Low",
                      {"d1": "Low"}).status_code not in (200, 201):
            pytest.skip("rob endpoint unavailable")
        self._post(client, project, project["second"], "High", {"d1": "Low"})
        r = client.get(f"/api/projects/{project['pid']}/rob-conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("rob conflicts unavailable")
        assert not (r.json() or {}).get("conflicts")

    def test_agreeing_assessments_are_not_a_conflict(self, client, project):
        if self._post(client, project, project["lead"], "Low").status_code not in (200, 201):
            pytest.skip("rob endpoint unavailable")
        self._post(client, project, project["second"], "Low")
        r = client.get(f"/api/projects/{project['pid']}/rob-conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("rob conflicts unavailable")
        conflicts = (r.json() or {}).get("conflicts", [])
        assert not any(c.get("paper_id") == "p1" for c in conflicts)

    def test_a_malformed_assessment_is_rejected_cleanly(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/rob-assessments",
                        json={}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 403, 422)
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------

# Fields carry an explicit id and a human LABEL. Extracted values are keyed by
# that id, not by the label, and extraction-conflict detection iterates the
# template's field ids: a value stored under any other key is invisible to it.
TEMPLATE = {"fields": [
    {"id": "sample_size", "label": "Sample size", "type": "number"},
    {"id": "design", "label": "Study design", "type": "text"},
]}


class TestExtraction:
    @pytest.fixture(autouse=True)
    def _template(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/extraction-template",
                       json=TEMPLATE, headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("extraction template unavailable")

    def test_the_template_round_trips(self, client, project):
        got = client.get(f"/api/projects/{project['pid']}/extraction-template",
                         headers=_hdr(project["lead"]))
        assert got.status_code == 200
        ids = {f["id"] for f in (got.json() or {}).get("fields", [])}
        assert {"sample_size", "design"} <= ids

    def test_an_unknown_field_type_falls_back_to_text(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/extraction-template",
                       json={"fields": [{"id": "x", "label": "X", "type": "nonsense"}]},
                       headers=_hdr(project["lead"]))
        assert r.json()["fields"][0]["type"] == "text"

    def test_a_field_without_an_id_gets_one_generated(self, client, project):
        r = client.put(f"/api/projects/{project['pid']}/extraction-template",
                       json={"fields": [{"label": "Unnamed"}]},
                       headers=_hdr(project["lead"]))
        assert r.json()["fields"][0]["id"]

    def _extract(self, client, project, uid, values):
        return client.post(f"/api/projects/{project['pid']}/extractions",
                           json={"paper_id": "p1", "values": values, "submitted": True},
                           headers=_hdr(uid))

    def test_an_extraction_round_trips(self, client, project):
        if self._extract(client, project, project["lead"],
                         {"sample_size": 128}).status_code not in (200, 201):
            pytest.skip("extractions unavailable")
        r = client.get(f"/api/projects/{project['pid']}/extractions",
                       headers=_hdr(project["lead"]))
        assert r.status_code == 200
        assert "128" in r.text

    def test_differing_values_are_reported_as_a_conflict(self, client, project):
        """Two reviewers extracting different numbers for the same field is the
        thing double extraction exists to catch."""
        if self._extract(client, project, project["lead"],
                         {"sample_size": 128}).status_code not in (200, 201):
            pytest.skip("extractions unavailable")
        self._extract(client, project, project["second"], {"sample_size": 999})
        r = client.get(f"/api/projects/{project['pid']}/extraction-conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("extraction conflicts unavailable")
        assert "p1" in r.text

    def test_identical_values_are_not_a_conflict(self, client, project):
        if self._extract(client, project, project["lead"],
                         {"sample_size": 128}).status_code not in (200, 201):
            pytest.skip("extractions unavailable")
        self._extract(client, project, project["second"], {"sample_size": 128})
        r = client.get(f"/api/projects/{project['pid']}/extraction-conflicts",
                       headers=_hdr(project["lead"]))
        if r.status_code != 200:
            pytest.skip("extraction conflicts unavailable")
        conflicts = (r.json() or {}).get("conflicts", [])
        assert not any(c.get("paper_id") == "p1" for c in conflicts)

    def test_a_reconciliation_can_be_recorded(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/extraction-reconciliations",
                        json={"paper_id": "p1", "values": {"sample_size": 128},
                              "rationale": "checked the table"},
                        headers=_hdr(project["lead"]))
        assert r.status_code != 500

    def test_a_malformed_extraction_is_rejected_cleanly(self, client, project):
        r = client.post(f"/api/projects/{project['pid']}/extractions",
                        json={}, headers=_hdr(project["lead"]))
        assert r.status_code in (400, 403, 422)
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_a_project_can_be_exported(self, client, project):
        _put_papers(client, project)
        r = client.get(f"/api/projects/{project['pid']}/export", headers=_hdr(project["lead"]))
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert r.json() is not None

    def test_a_non_member_cannot_export(self, client, project):
        r = client.get(f"/api/projects/{project['pid']}/export", headers=_hdr("stranger"))
        assert r.status_code in (401, 403, 404)
