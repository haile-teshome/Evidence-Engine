"""Shared pytest setup.

Two jobs, both about isolation:

1. Point the SQLite store at a throwaway file BEFORE anything imports it, so a
   test run can never read, write or corrupt the user's real projects and
   accounts in ~/.evidence-engine/evidence.db.

2. Make sure no test can reach the network. Every outbound call in this suite is
   meant to be stubbed; if one is missed, it must fail loudly rather than quietly
   hitting a live scholarly API, which would make the suite slow, flaky, and
   rude to services that rate-limit us.
"""
import os
import socket
import tempfile
from pathlib import Path

import pytest

# Must happen before `store` is imported anywhere, since it resolves its path at
# import time. pytest loads conftest before collecting test modules.
_TMP_DB = Path(tempfile.gettempdir()) / "evidence-engine-test" / "test.db"
_TMP_DB.parent.mkdir(parents=True, exist_ok=True)
os.environ["EE_DB_PATH"] = str(_TMP_DB)


@pytest.fixture(autouse=True)
def _no_accidental_network(monkeypatch, request):
    """Fail any test that opens a real socket.

    Opt out with @pytest.mark.network for a test that deliberately needs it.
    """
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "This test tried to open a network connection. Stub the call, or "
            "mark the test with @pytest.mark.network if it genuinely needs one."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def temp_store(monkeypatch):
    """A store backed by its own empty database, isolated per test."""
    import store

    db = Path(tempfile.mkdtemp(prefix="ee-store-")) / "t.db"
    monkeypatch.setattr(store, "_DB_PATH", db)
    monkeypatch.setattr(store, "_initialized", False)
    return store
