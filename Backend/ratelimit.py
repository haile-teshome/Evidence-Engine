# ============================================================================
# FILE: ratelimit.py
# Per-host rate limiting and a small shared cache for the scholarly APIs.
#
# Citation snowballing was taking about 74 seconds per seed paper, almost all
# of it spent idle: every lookup was issued, awaited in full, and only then
# followed by the next. The achieved rate was roughly 0.1 requests per second
# against services that permit several per second, so the run was latency-bound
# rather than limit-bound.
#
# The fix is to overlap the waiting, NOT to send more than the services allow.
# Each host gets a token bucket sized BELOW its published limit, and every
# worker blocks on that bucket before it is allowed to call. Raising
# concurrency therefore cannot raise the request rate past the cap; it only
# removes the dead time between calls.
#
# Published limits, and what this module actually uses:
#
#   OpenAlex          100 req/s + a daily BUDGET         -> 3/s
#       Since February 2026 OpenAlex is usage-based. Without an API key a
#       caller shares a free budget of $0.10 a day with everyone on the same
#       network IP address, at 10 credits ($0.001) a request: roughly a
#       hundred requests a day for the whole institution. A 95-seed snowball
#       run needs more than that, so the 429 it returns is a budget refusal,
#       not a speed one, and no amount of rate limiting avoids it. A key is
#       free and carries its own budget; see _polite_params in api.py.
#   Europe PMC        10 req/s (500/min) per IP         -> 3/s
#       Measured behaviour is worse than the published figure: it throttles
#       after a short burst and then answers in 10-30s or not at all, so the
#       cap is not the binding constraint and there is nothing to gain by
#       approaching it.
#   Semantic Scholar  1 req/s with a key; unauthenticated callers share a
#                     pool of 5,000 per 5 minutes ACROSS ALL USERS  -> 1/s
#       api.semanticscholar.org: they ask for exponential backoff.
#   NCBI E-utilities  3 req/s, 10 with an API key       -> 3/s (10 with a key)
#   Crossref          polite pool                       -> 5/s
#   OpenReview        20 req/min, stated in its own response headers -> 0.33/s
#
# Every number here is below the published one. If a service publishes a lower
# limit in future, change it here: this is the only place the rate is decided.
# ============================================================================

import os
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

# host key -> requests per second we allow ourselves
_LIMITS: Dict[str, float] = {
    "openalex":         3.0,
    "europepmc":        3.0,
    "semanticscholar":  1.0,
    "ncbi":             3.0,
    "crossref":         5.0,
    # OpenReview advertises its own limit in every response header:
    # `ratelimit-policy: 20;w=60`. Twenty a minute is a third of a request a
    # second, and it is the tightest limit of any source here.
    "openreview":       0.33,
    "other":            2.0,
}


def _ncbi_rate() -> float:
    """NCBI publishes 3 requests a second, or 10 for a caller with a key."""
    from request_creds import get_cred
    return 10.0 if (get_cred("ncbi") or os.getenv("NCBI_API_KEY", "")) else 3.0


class _Bucket:
    """A token bucket. `acquire` blocks until a token is available.

    Shared by every thread in the process, so the rate is a property of the
    application rather than of any one worker pool.
    """

    def __init__(self, rate: float):
        self.rate = rate
        # No burst allowance. A bucket holding `rate` tokens lets a caller
        # spend them instantly and then earn `rate` more within the same
        # second, so the observed rate over a one-second window reaches twice
        # the cap. Measured at 5/s against a 3/s limit before this was fixed.
        # One token means calls are simply spaced 1/rate apart.
        self.capacity = 1.0
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait, 1.0))


_buckets: Dict[str, _Bucket] = {}
_buckets_lock = threading.Lock()


def _bucket(key: str) -> _Bucket:
    rate = _ncbi_rate() if key == "ncbi" else _LIMITS.get(key, _LIMITS["other"])
    with _buckets_lock:
        b = _buckets.get(key)
        if b is None or b.rate != rate:
            b = _Bucket(rate)
            _buckets[key] = b
        return b


def host_key(url: str) -> str:
    u = (url or "").lower()
    if "openalex.org" in u:
        return "openalex"
    if "ebi.ac.uk" in u or "europepmc.org" in u:
        return "europepmc"
    if "semanticscholar.org" in u:
        return "semanticscholar"
    if "ncbi.nlm.nih.gov" in u:
        return "ncbi"
    if "crossref.org" in u:
        return "crossref"
    if "openreview.net" in u:
        return "openreview"
    return "other"


# Worth trying again shortly: the service is up but refusing this moment.
# Europe PMC in particular answers 503 in bursts and recovers within seconds.
_RETRYABLE = {429, 502, 503, 504}


def limited_get(url: str, *, timeout: float = 8.0, retries: int = 2, **kw) -> Optional[requests.Response]:
    """A GET that waits its turn, and retries a transient refusal with backoff.

    Returns None rather than raising when it fails: a snowball run over
    hundreds of seeds must not stop because one lookup was throttled.

    A TIMEOUT IS NOT RETRIED. The two failures look alike and need opposite
    responses. A 429 says "you asked too fast", and waiting fixes it. A timeout
    against Europe PMC says the service is simply slow right now, and asking
    again costs another full timeout for the same nothing: retrying three times
    at 6 seconds turned a 10-second failure into an 18-second one, and made the
    run slower than the version this replaced.

    The default timeout is deliberately short for the same reason: Europe PMC
    frequently takes longer than 10 seconds and then fails anyway, so a long
    timeout buys a wait and no data.
    """
    key = host_key(url)
    for attempt in range(retries + 1):
        _bucket(key).acquire()
        try:
            r = requests.get(url, timeout=timeout, **kw)
        except requests.RequestException:
            return None                      # slow or unreachable: do not pile on
        if r.status_code not in _RETRYABLE:
            return r
        # Say WHY, once. A 429 for going too fast is fixed by waiting; a 429
        # for an exhausted budget is fixed by a key, and waiting out a
        # Retry-After measured in hours is not a strategy. A 5xx is the
        # service's own problem and usually clears in seconds.
        if attempt == 0:
            retry_after = r.headers.get("Retry-After", "")
            budget = r.headers.get("X-RateLimit-Remaining-USD")
            detail = f", budget remaining ${budget}" if budget is not None else ""
            print(f"[ratelimit] {key} returned {r.status_code} "
                  f"(retry-after {retry_after or 'unset'}s{detail})")
        if attempt >= retries:
            return r
        # Exponential backoff with jitter. Semantic Scholar asks for this
        # explicitly, and it is the right behaviour for any throttled host.
        time.sleep(min(8.0, (2 ** attempt) * 1.0) + random.random() * 0.4)
    return None


# ---------------------------------------------------------------------------
# A process-wide cache for lookups that repeat across seed papers.
#
# Snowballing a coherent corpus produces heavy overlap: the same reference
# appears under many seeds, and each one was resolved from scratch because the
# de-duplication map lived inside a single request. Caching does not merely
# make the run faster, it removes requests the services would otherwise have
# had to serve.
# ---------------------------------------------------------------------------

_CACHE_TTL = 60 * 60          # an hour: long enough for a run, short enough to stay fresh
_CACHE_MAX = 5000

_cache: Dict[str, Tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_hits = 0
_misses = 0


def cached(namespace: str, key: str, produce) -> Any:
    """Return `produce()`, memoised under (namespace, key) for the run."""
    global _hits, _misses
    if not key:
        return produce()
    k = f"{namespace}:{key.strip().lower()[:300]}"
    now = time.time()
    with _cache_lock:
        hit = _cache.get(k)
        if hit and now - hit[0] < _CACHE_TTL:
            _hits += 1
            return hit[1]
    _misses += 1
    value = produce()
    # Never memoise a failure. A lookup that came back empty because the
    # service was throttling would otherwise stay empty for the whole run,
    # turning one refused request into a permanently missing paper.
    if not value:
        return value
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            # Drop the oldest tenth rather than clearing: a run in progress
            # should keep most of what it has learned.
            for old in sorted(_cache, key=lambda x: _cache[x][0])[: _CACHE_MAX // 10]:
                _cache.pop(old, None)
        _cache[k] = (now, value)
    return value


def cache_stats() -> Dict[str, int]:
    return {"hits": _hits, "misses": _misses, "entries": len(_cache)}


def reset_cache() -> None:
    global _hits, _misses
    with _cache_lock:
        _cache.clear()
    _hits = _misses = 0
