# ============================================================================
# FILE: request_creds.py
# Per-request credentials (LLM provider API keys, optional database keys).
#
# The React frontend stores keys ONLY in the user's browser and sends the
# relevant one with each request as an HTTP header. A middleware in api.py drops
# them into the ContextVar below for the lifetime of that request. Model and data
# services read from here first, then fall back to environment variables. Keys
# are never written to disk or logged by this backend.
#
# Starlette copies the active context into the threadpool it uses to run sync
# endpoint handlers, so a value set in the async middleware is visible to the
# get_cred() calls made inside those handlers.
# ============================================================================

import contextvars

_creds: "contextvars.ContextVar[dict]" = contextvars.ContextVar("request_creds", default={})


def set_request_creds(d: dict) -> None:
    _creds.set(d or {})


def get_cred(name: str) -> str:
    """Return the per-request credential for `name` (e.g. 'openai', 'anthropic',
    'google', 'core', 'semantic_scholar'), or '' if none was supplied."""
    try:
        return (_creds.get() or {}).get(name) or ""
    except Exception:
        return ""


def openalex_auth() -> dict:
    """Query params that authenticate a call to OpenAlex, or {} without a key.

    Since February 2026 OpenAlex meters unkeyed callers: the request counts
    against a free daily budget shared by every machine on the same public IP,
    which on a university network is spent long before the day starts. The call
    still returns 200 with an error body, so an unkeyed caller sees empty
    results rather than a failure. Every OpenAlex request must go through here.
    """
    import os
    key = get_cred("openalex") or os.getenv("OPENALEX_API_KEY", "")
    return {"api_key": key} if key else {}


# Header name <-> credential key mapping, shared by the middleware.
HEADER_TO_CRED = {
    "x-llm-openai-key": "openai",
    "x-llm-anthropic-key": "anthropic",
    "x-llm-google-key": "google",
    "x-db-openalex-key": "openalex",
    "x-db-core-key": "core",
    "x-db-semantic-scholar-key": "semantic_scholar",
    "x-db-scopus-key": "scopus",
    "x-db-ncbi-key": "ncbi",
    "x-db-springer-key": "springer",
    "x-db-ieee-key": "ieee",
    "x-db-wos-key": "wos",
    # Publisher text-and-data-mining credentials. These reach SUBSCRIPTION
    # content through the interface each publisher provides for text mining;
    # entitlement comes from the user's institution, not from this app, and
    # with no key the corresponding tier makes no request at all.
    "x-tdm-wiley-token": "wiley",
    "x-tdm-elsevier-insttoken": "elsevier_insttoken",
    # LibKey (Third Iron): maps a DOI to the copy this library is entitled to.
    # Needs the library's numeric id as well as the token.
    "x-libkey-id": "libkey_id",
    "x-libkey-token": "libkey",
    # Not a key: the user's OWN contact address. Several scholarly APIs require
    # a real, working email as their politeness policy — Unpaywall hard-rejects
    # calls without one (HTTP 422). Supplied per user from the browser like any
    # other credential, so no address is ever baked into the codebase.
    "x-user-contact-email": "contact_email",
}
