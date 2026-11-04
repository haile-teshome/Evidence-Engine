"""The screening pack: one self-contained file a reviewer opens in any browser, and the
decisions it exports merging back into the lead's project.

This is the route that works when the app is not reachable at any address the reviewer can
open, which is the normal case for a locally run instance, so the round trip is tested without
a browser: the pack's embedded data is read, the export the page would produce is rebuilt, and
the real import endpoint consumes it.
"""
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
import review_pack

client = TestClient(app)
LEAD = "pack_lead"


def _hdr(uid=LEAD):
    return {"x-reviewer-id": uid}


@pytest.fixture
def project():
    pid = client.post("/api/projects", json={"name": "Pack test review", "screening_mode": "dual_blinded",
                                             "inclusion": ["RCT", "adults"], "exclusion": ["animal study"],
                                             "pico": {"population": "adults", "intervention": "drug",
                                                      "comparator": "placebo", "outcome": "mortality"}},
                      headers=_hdr()).json()["project"]["id"]
    papers = [{"paper_id": f"p{i}", "title": f"Study {i}", "abstract": f"Abstract for study {i}",
               "journal": "J Test", "year": 2020 + i, "pmid": f"1000{i}"} for i in range(3)]
    assert client.put(f"/api/projects/{pid}/papers", json={"papers": papers}, headers=_hdr()).status_code == 200
    return pid


def _embedded(html: str) -> dict:
    m = re.search(r'<script id="pack" type="application/json">(.*?)</script>', html, re.S)
    assert m, "the pack must embed its data"
    return json.loads(m.group(1))


def test_pack_is_a_download_with_a_sensible_filename(project):
    r = client.get(f"/api/projects/{project}/review-pack?email=rev@example.edu", headers=_hdr())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"]
    assert "pack-test-review" in r.headers["content-disposition"]
    assert "rev-example-edu" in r.headers["content-disposition"]


def test_pack_carries_the_criteria_and_every_paper(project):
    d = _embedded(client.get(f"/api/projects/{project}/review-pack?email=rev@example.edu",
                             headers=_hdr()).text)
    assert d["project_name"] == "Pack test review"
    assert d["inclusion"] == ["RCT", "adults"] and d["exclusion"] == ["animal study"]
    assert d["pico"]["population"] == "adults"
    assert [p["paper_id"] for p in d["papers"]] == ["p0", "p1", "p2"]
    assert d["papers"][0]["abstract"].startswith("Abstract for study 0")


def test_pack_has_no_external_references(project):
    """It must work from a mail attachment with no network at all."""
    html = client.get(f"/api/projects/{project}/review-pack", headers=_hdr()).text
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    assert "<script" in html and "localStorage" in html      # self-contained behaviour


def test_pack_identity_comes_from_the_invitation(project):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer", "email": "Invited@Example.edu"},
                      headers=_hdr()).json()["invite"]["token"]
    d = _embedded(client.get(f"/api/projects/{project}/review-pack?token={tok}", headers=_hdr()).text)
    assert d["reviewer_email"] == "invited@example.edu"
    assert d["reviewer_user_id"] == "pack:invited@example.edu"   # stable across repeated packs


def test_decisions_exported_by_the_pack_import_back(project):
    """The exact bundle the page's export button builds, consumed by the real import route."""
    d = _embedded(client.get(f"/api/projects/{project}/review-pack?email=rev@example.edu", headers=_hdr()).text)
    decisions = [{"paper_id": p["paper_id"], "stage": d["stage"], "reviewer_user_id": d["reviewer_user_id"],
                  "decision": v, "reason": f"because {v}", "decided_at": "2026-10-01T00:00:00+00:00",
                  "per_pico_verdict": None, "ai_decision": None, "is_override": False}
                 for p, v in zip(d["papers"], ["include", "exclude", "maybe"])]
    bundle = {"kind": d["kind"], "bundle_version": d["bundle_version"],
              "project": {"id": d["project_id"], "name": d["project_name"]}, "decisions": decisions}
    merged = client.post(f"/api/projects/{project}/import", json={"bundle": bundle}, headers=_hdr())
    assert merged.status_code == 200, merged.text
    assert merged.json()["merged"]["decisions"] == 3
    back = client.get(f"/api/projects/{project}/decisions?stage=abstract", headers=_hdr()).json()
    rows = [r for r in (back.get("decisions") or back) if r["reviewer_user_id"] == d["reviewer_user_id"]]
    assert {r["decision"] for r in rows} == {"include", "exclude", "maybe"}
    assert all(r["reason"].startswith("because") for r in rows)


def test_importing_the_same_pack_twice_does_not_duplicate(project):
    d = _embedded(client.get(f"/api/projects/{project}/review-pack?email=twice@example.edu", headers=_hdr()).text)
    bundle = {"kind": d["kind"], "bundle_version": d["bundle_version"],
              "project": {"id": d["project_id"], "name": d["project_name"]},
              "decisions": [{"paper_id": "p0", "stage": "abstract", "reviewer_user_id": d["reviewer_user_id"],
                             "decision": "include", "reason": "first pass"}]}
    client.post(f"/api/projects/{project}/import", json={"bundle": bundle}, headers=_hdr())
    bundle["decisions"][0]["decision"] = "exclude"           # they changed their mind and resent
    client.post(f"/api/projects/{project}/import", json={"bundle": bundle}, headers=_hdr())
    back = client.get(f"/api/projects/{project}/decisions?stage=abstract", headers=_hdr()).json()
    rows = [r for r in (back.get("decisions") or back)
            if r["reviewer_user_id"] == d["reviewer_user_id"] and r["paper_id"] == "p0"]
    assert len(rows) == 1 and rows[0]["decision"] == "exclude"   # replaced, not duplicated


def test_pack_is_lead_only(project):
    tok = client.post(f"/api/projects/{project}/invites", json={"role": "reviewer"},
                      headers=_hdr()).json()["invite"]["token"]
    client.post(f"/api/invites/{tok}/accept", headers=_hdr("pack_plain_reviewer"))
    assert client.get(f"/api/projects/{project}/review-pack",
                      headers=_hdr("pack_plain_reviewer")).status_code == 403


def test_hostile_paper_text_cannot_inject_markup(project):
    """A title from an upstream database is untrusted. It must not be able to close the data
    block early, and it must never reach the page as markup: the pack sets text, not HTML."""
    nasty = '</script><img src=x onerror=alert(1)>'
    client.put(f"/api/projects/{project}/papers",
               json={"papers": [{"paper_id": "x1", "title": nasty, "abstract": nasty}]}, headers=_hdr())
    html = client.get(f"/api/projects/{project}/review-pack", headers=_hdr()).text
    # exactly the two legitimate closing tags: the hostile ones are escaped to <\/script>
    assert html.count("</script>") == 2
    assert "<\\/script>" in html or "<\/script>" in html
    body = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert "<img" not in body                                 # nothing live in the rendered HTML
    d = _embedded(html)
    assert d["papers"][0]["title"] == nasty                   # preserved verbatim as data
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    assert "textContent = p.title" in js                      # rendered as text
    assert not re.search(r"innerHTML\s*=\s*[^;]*p\.", js)      # never as HTML


def test_hostile_project_fields_are_escaped(project):
    """Project name and criteria are interpolated into the document, so they are escaped."""
    nasty = '<img src=x onerror=alert(1)>'
    html = review_pack.build_pack({"id": "p", "name": nasty, "pico": {"population": nasty},
                                   "inclusion": [nasty], "exclusion": []}, [], "a@b.co", "pack:a@b.co")
    body = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert "<img" not in body and "&lt;img" in body


def test_builder_handles_a_project_with_no_papers():
    html = review_pack.build_pack({"id": "p", "name": "Empty", "pico": {}, "inclusion": [], "exclusion": []},
                                  [], "a@b.co", "pack:a@b.co")
    assert "Empty" in html and "none recorded" in html
