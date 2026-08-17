# ============================================================================
# FILE: keychain.py
# Optional storage of LLM provider keys in the operating system keychain
# (macOS Keychain, Windows Credential Manager, Linux Secret Service) via the
# `keyring` library. This is the strongest at-rest option: the secret never
# lives in the browser or the project, only in the OS credential store, and the
# local backend reads it directly when building a model.
#
# All operations are guarded: if `keyring` or an OS backend is unavailable, the
# app silently falls back to the browser (passphrase-encrypted) storage mode.
# ============================================================================

SERVICE = "evidence-engine"
PROVIDERS = ("openai", "anthropic", "google")

try:
    import keyring
    from keyring.backends.fail import Keyring as _FailKeyring
    _kr = keyring.get_keyring()
    _AVAILABLE = not isinstance(_kr, _FailKeyring)
except Exception:
    keyring = None
    _AVAILABLE = False

# Small in-process cache so we don't hit the OS keychain on every model build
# (a read per request would be wasteful and, on some platforms, prompt-happy).
_cache: "dict[str, str]" = {}


def available() -> bool:
    return _AVAILABLE


def get_key(provider: str) -> str:
    if not _AVAILABLE:
        return ""
    if provider in _cache:
        return _cache[provider]
    try:
        val = keyring.get_password(SERVICE, provider) or ""
    except Exception:
        val = ""
    _cache[provider] = val
    return val


def set_key(provider: str, key: str) -> None:
    if not _AVAILABLE:
        raise RuntimeError("OS keychain is not available on this system")
    keyring.set_password(SERVICE, provider, key)
    _cache[provider] = key


def delete_key(provider: str) -> None:
    _cache.pop(provider, None)
    if not _AVAILABLE:
        return
    try:
        keyring.delete_password(SERVICE, provider)
    except Exception:
        pass


def status() -> dict:
    """Which providers currently have a key stored in the OS keychain."""
    return {p: bool(get_key(p)) for p in PROVIDERS}
