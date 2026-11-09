"""Downloading and staging a release.

Run: pytest Backend/test_self_update.py

The dangerous parts are not the download. They are writing outside the updates
directory, where the reviewer's database lives, and staging bytes that failed
their checksum, which would replace a working app with a damaged one.
"""
import hashlib
import json

import pytest

import self_update


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EE_UPDATE_DIR", str(tmp_path / "updates"))
    yield


def test_asset_matches_the_machine(monkeypatch):
    monkeypatch.setattr(self_update.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(self_update.platform, "machine", lambda: "arm64")
    assert self_update.asset_name("1.0.11") == "EvidenceEngine-1.0.11-darwin-arm64.dmg"

    monkeypatch.setattr(self_update.platform, "machine", lambda: "x86_64")
    assert self_update.asset_name("1.0.11") == "EvidenceEngine-1.0.11-darwin-x64.dmg"

    monkeypatch.setattr(self_update.platform, "system", lambda: "Windows")
    assert self_update.asset_name("v1.0.11") == "EvidenceEngine-1.0.11-win-x64-Setup.exe"

    monkeypatch.setattr(self_update.platform, "system", lambda: "Linux")
    assert self_update.asset_name("1.0.11") == "EvidenceEngine-1.0.11-linux-x64.zip"


def test_download_url_points_at_the_tag():
    url = self_update._download_url("1.0.11", "SHA256SUMS")
    assert url.endswith("/releases/download/v1.0.11/SHA256SUMS")
    assert self_update._download_url("v1.0.11", "x") == self_update._download_url("1.0.11", "x")


def test_reads_the_checksum_for_this_asset(monkeypatch):
    body = (
        "aaa  EvidenceEngine-1.0.11-darwin-arm64.dmg\n"
        "bbb  EvidenceEngine-1.0.11-linux-x64.zip\n"
    )

    class Res:
        text = body
        def raise_for_status(self): pass

    monkeypatch.setattr(self_update.requests, "get", lambda *a, **k: Res())
    assert self_update.expected_sha("1.0.11", "EvidenceEngine-1.0.11-linux-x64.zip") == "bbb"
    # An asset the release does not list is not a reason to invent a checksum.
    assert self_update.expected_sha("1.0.11", "nothing-like-it.zip") == ""


def test_a_missing_sums_file_is_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(self_update.requests, "get", boom)
    assert self_update.expected_sha("1.0.11", "anything") == ""


def test_everything_it_writes_stays_in_the_updates_directory(tmp_path):
    # The reviews, the database and the cached text are the siblings of this
    # directory. Nothing here may address them.
    assert str(self_update.updates_dir()).endswith("updates")
    assert self_update.updates_dir() == tmp_path / "updates"


def test_a_damaged_download_is_never_staged(tmp_path, monkeypatch):
    work = self_update.updates_dir()
    work.mkdir(parents=True)

    def fake_download(url, out):
        out.write_bytes(b"not the release you were promised")

    monkeypatch.setattr(self_update, "_download", fake_download)
    monkeypatch.setattr(self_update, "expected_sha", lambda v, n: "a" * 64)
    staged = []
    monkeypatch.setattr(self_update, "_stage", lambda *a: staged.append(a) or tmp_path)

    self_update._run("1.0.11")

    assert self_update.status()["state"] == "failed"
    assert "checksum" in self_update.status()["error"]
    assert staged == [], "a download that failed its checksum must not be unpacked"
    assert not (work / "staged.json").exists()


def test_a_good_download_is_staged_and_recorded(tmp_path, monkeypatch):
    payload = b"the real thing"
    digest = hashlib.sha256(payload).hexdigest()
    work = self_update.updates_dir()

    monkeypatch.setattr(self_update, "_download", lambda url, out: out.write_bytes(payload))
    monkeypatch.setattr(self_update, "expected_sha", lambda v, n: digest)
    monkeypatch.setattr(self_update, "_stage", lambda archive, w: w / "Evidence Engine.app")

    self_update._run("1.0.11")

    assert self_update.status()["state"] == "ready"
    recorded = json.loads((work / "staged.json").read_text())
    assert recorded["version"] == "1.0.11"
    assert recorded["sha256"] == digest
    # The archive is 500 MB. Keeping it after unpacking would quietly fill the
    # reviewer's disk one release at a time.
    assert not (work / self_update.asset_name("1.0.11")).exists()


def test_one_update_at_a_time(monkeypatch):
    monkeypatch.setattr(self_update.threading, "Thread", lambda *a, **k: type(
        "T", (), {"start": lambda self: None})())
    first = self_update.start("1.0.11")
    assert first["state"] == "downloading"
    # A second click while one is running must not start another download.
    again = self_update.start("1.0.12")
    assert again["version"] == "1.0.11"
