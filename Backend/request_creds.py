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


# Header name <-> credential key mapping, shared by the middleware.
HEADER_TO_CRED = {
    "x-llm-openai-key": "openai",
    "x-llm-anthropic-key": "anthropic",
    "x-llm-google-key": "google",
    "x-db-core-key": "core",
    "x-db-semantic-scholar-key": "semantic_scholar",
}
