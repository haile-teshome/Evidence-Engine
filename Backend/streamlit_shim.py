"""Headless Streamlit shim.

Registers a FAKE ``streamlit`` module so modules originally written for
``streamlit run`` (data_services, state_manager, utils, ui_components, ...) can
be imported and called from the FastAPI process WITHOUT installing the real,
heavy, and install-fragile ``streamlit`` package.

All UI calls become no-ops or log lines; ``st.session_state`` is a plain dict.
Import this before anything that does ``import streamlit`` (api.py does, at the
top) so the fake is registered first and shadows any real install.

    from streamlit_shim import install
    install()
    from data_services import DataAggregator   # safe: sees the fake streamlit
"""

from __future__ import annotations

import sys
import types


class _SessionState(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]


class _NoopCtx:
    """Stand-in for context managers / layout objects (st.spinner, st.empty, ...)."""

    def __getattr__(self, _name):
        return self

    def __call__(self, *_a, **_k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


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


def _build_fake_streamlit() -> types.ModuleType:
    m = types.ModuleType("streamlit")
    m.session_state = _session_state

    # Message calls → log lines.
    for name in (
        "error", "warning", "info", "success", "write", "markdown", "header",
        "subheader", "title", "caption", "code", "text", "json", "toast",
        "divider", "exception",
    ):
        setattr(m, name, _logger(name))

    # Layout / widget calls → no-op context objects.
    for name in (
        "empty", "spinner", "status", "container", "expander", "form",
        "placeholder", "progress", "metric", "sidebar", "tabs", "columns",
        "popover", "chat_message",
    ):
        setattr(m, name, lambda *_a, **_k: _NoopCtx())

    m.set_page_config = lambda *_a, **_k: None
    m.stop = lambda: None
    m.rerun = lambda: None
    m.experimental_rerun = lambda: None
    m.cache_data = lambda *a, **k: (lambda f: f)
    m.cache_resource = lambda *a, **k: (lambda f: f)

    # Catch-all for any other streamlit attribute: a universal no-op callable
    # that also works as a context manager / factory. Dunder lookups (e.g.
    # __file__, __path__) must raise so import machinery treats this as a plain
    # module, not a package.
    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *_a, **_k: _NoopCtx()

    m.__getattr__ = __getattr__   # PEP 562 module-level __getattr__
    return m


# Register the fake immediately so any later `import streamlit` resolves to it.
# If streamlit is already in sys.modules (unlikely — the shim imports first),
# leave it; otherwise the fake shadows any real install without importing it.
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _build_fake_streamlit()


def install() -> _SessionState:
    """Back-compat entry point. The fake module is already active on import;
    this just returns the shared headless session-state dict."""
    sys.modules.setdefault("streamlit", _build_fake_streamlit())
    return _session_state


session_state = _session_state
