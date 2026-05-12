"""Headless Streamlit shim.

Replaces st.session_state with a plain dict and turns UI calls (st.error,
st.write, st.empty, st.spinner, ...) into no-ops or log lines so that
modules originally written for `streamlit run` can be imported and called
from a FastAPI process.

Usage (must run before importing utils / data_services / app):

    from streamlit_shim import install
    install()
    from utils import AIService           # safe now
"""

from __future__ import annotations

import streamlit as st


class _SessionState(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]


class _NoopCtx:
    """Stand-in for context managers like st.spinner / st.status / st.empty."""

    def __getattr__(self, _name):
        return self

    def __call__(self, *_a, **_k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def write(self, *_a, **_k):
        return None

    def markdown(self, *_a, **_k):
        return None

    def update(self, *_a, **_k):
        return None


_session_state = _SessionState()


def _logger(level: str):
    def _fn(*args, **_kwargs):
        try:
            msg = " | ".join(str(a)[:300] for a in args)
            if msg:
                print(f"[st.{level}] {msg}")
        except Exception:
            pass

    return _fn


def install() -> _SessionState:
    """Monkey-patch streamlit into a headless mode. Returns the session state dict."""
    st.session_state = _session_state

    for name in (
        "error",
        "warning",
        "info",
        "success",
        "write",
        "markdown",
        "header",
        "subheader",
        "title",
        "caption",
        "code",
        "text",
        "json",
        "toast",
        "divider",
        "exception",
    ):
        setattr(st, name, _logger(name))

    for name in (
        "empty",
        "spinner",
        "status",
        "container",
        "expander",
        "form",
        "placeholder",
        "progress",
        "metric",
        "sidebar",
        "tabs",
        "columns",
        "popover",
        "chat_message",
    ):
        setattr(st, name, lambda *_a, **_k: _NoopCtx())

    setattr(st, "set_page_config", lambda *_a, **_k: None)
    setattr(st, "stop", lambda: None)
    setattr(st, "rerun", lambda: None)
    setattr(st, "cache_data", lambda *a, **k: (lambda f: f))
    setattr(st, "cache_resource", lambda *a, **k: (lambda f: f))

    return _session_state


session_state = _session_state
