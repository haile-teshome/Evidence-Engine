"""Tests for the persistence and auth layer. No network, throwaway database.

Run: Backend/.venv/bin/python -m pytest Backend/test_store.py

`conftest.py` redirects the SQLite path so a test run can never touch the real
~/.evidence-engine/evidence.db. The password and token cases are the ones worth
having: a silent weakening there (a hash that always verifies, a token that
collides) is invisible in normal use and only shows up as a breach.
"""
import pytest

import store


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_is_not_the_plaintext(self):
        assert "hunter2" not in store._hash_password("hunter2")

    def test_hash_is_salted_so_two_hashes_differ(self):
        """Unsalted hashes let one rainbow table crack every account at once."""
        assert store._hash_password("same") != store._hash_password("same")

    def test_correct_password_verifies(self):
        assert store._verify_password("hunter2", store._hash_password("hunter2"))

    def test_wrong_password_does_not_verify(self):
        assert not store._verify_password("wrong", store._hash_password("hunter2"))

    def test_uses_pbkdf2_with_a_real_iteration_count(self):
        algo, iters, salt, digest = store._hash_password("x").split("$")
        assert algo == "pbkdf2_sha256"
        assert int(iters) >= 100_000
        assert len(salt) == 32          # 16 salt bytes, hex encoded
        assert len(digest) == 64        # sha256, hex encoded

    @pytest.mark.parametrize("stored", [
        "", "garbage", "notpbkdf2$1$aa$bb", "pbkdf2_sha256$only$three",
        "md5$1000$aa$bb", None,
    ])
    def test_malformed_stored_hash_fails_closed(self, stored):
        """A corrupt record must deny access, never grant it."""
        assert store._verify_password("anything", stored) is False

    def test_empty_password_still_round_trips(self):
        assert store._verify_password("", store._hash_password(""))

    def test_unicode_password_round_trips(self):
        pw = "pässwörd-日本語-🔐"
        assert store._verify_password(pw, store._hash_password(pw))

    def test_password_is_case_sensitive(self):
        assert not store._verify_password("HUNTER2", store._hash_password("hunter2"))


# ---------------------------------------------------------------------------
# Tokens and bearer parsing
# ---------------------------------------------------------------------------

class TestTokens:
    def test_tokens_are_unique_per_call(self):
        uid = "rev_1"
        assert store._create_token(uid) != store._create_token(uid)

    def test_token_resolves_back_to_its_user(self):
        token = store._create_token("rev_42")
        assert store._uid_for_token(token) == "rev_42"

    @pytest.mark.parametrize("bad", ["", "not-a-token", "rev_42", None])
    def test_unknown_token_resolves_to_nothing(self, bad):
        assert not store._uid_for_token(bad)

    def test_token_is_long_enough_to_resist_guessing(self):
        assert len(store._create_token("u")) >= 24

    @pytest.mark.parametrize("header,expected", [
        ("Bearer abc123", "abc123"),
        ("bearer abc123", "abc123"),
        ("BEARER abc123", "abc123"),
        ("Bearer   abc123  ", "abc123"),
    ])
    def test_bearer_extraction(self, header, expected):
        assert store._bearer(header) == expected

    @pytest.mark.parametrize("header", [None, "", "abc123", "Basic abc123", "Bearer"])
    def test_non_bearer_headers_yield_nothing(self, header):
        assert store._bearer(header) in ("", None)


# ---------------------------------------------------------------------------
# Identifier generation
# ---------------------------------------------------------------------------

class TestIds:
    def test_ids_carry_their_prefix(self):
        assert store._new_id("proj").startswith("proj")

    def test_ids_are_unique(self):
        assert len({store._new_id("p") for _ in range(200)}) == 200


# ---------------------------------------------------------------------------
# Key/value persistence
# ---------------------------------------------------------------------------

class TestKeyValueStore:
    def test_round_trips_a_dict(self, temp_store):
        temp_store.kv_set("k1", {"a": 1, "b": [1, 2, 3]})
        assert temp_store.kv_get("k1") == {"a": 1, "b": [1, 2, 3]}

    def test_missing_key_returns_none(self, temp_store):
        assert temp_store.kv_get("never-written") is None

    def test_set_overwrites(self, temp_store):
        temp_store.kv_set("k", {"v": 1})
        temp_store.kv_set("k", {"v": 2})
        assert temp_store.kv_get("k") == {"v": 2}

    def test_delete_removes(self, temp_store):
        temp_store.kv_set("k", {"v": 1})
        temp_store.kv_del("k")
        assert temp_store.kv_get("k") is None

    def test_delete_of_a_missing_key_is_not_an_error(self, temp_store):
        temp_store.kv_del("never-written")

    def test_prefix_query_returns_only_matches(self, temp_store):
        temp_store.kv_set("proj:1", {"n": 1})
        temp_store.kv_set("proj:2", {"n": 2})
        temp_store.kv_set("other:1", {"n": 3})
        got = temp_store.kv_get_by_prefix("proj:")
        assert len(got) == 2
        assert {g["n"] for g in got} == {1, 2}

    def test_prefix_query_with_no_matches_is_empty(self, temp_store):
        assert temp_store.kv_get_by_prefix("nothing:") == []

    def test_values_survive_a_reconnect(self, temp_store):
        """State must live in the file, not in a process-local cache."""
        temp_store.kv_set("persisted", {"v": 99})
        temp_store._initialized = False
        assert temp_store.kv_get("persisted") == {"v": 99}

    def test_unicode_and_nesting_survive_the_json_round_trip(self, temp_store):
        payload = {"title": "Zähne 日本語 🦷", "nested": {"list": [1, {"x": None}]}}
        temp_store.kv_set("u", payload)
        assert temp_store.kv_get("u") == payload


# ---------------------------------------------------------------------------
# Isolation guarantees for the suite itself
# ---------------------------------------------------------------------------

class TestSuiteIsolation:
    def test_tests_do_not_point_at_the_real_database(self):
        assert ".evidence-engine/evidence.db" not in str(store._DB_PATH), \
            "the suite must never open the user's real database"

    def test_network_is_blocked_by_default(self):
        import socket
        with pytest.raises(RuntimeError, match="network connection"):
            socket.socket().connect(("example.com", 80))
