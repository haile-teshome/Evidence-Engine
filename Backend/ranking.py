"""Screening prioritisation: the ranker behind active-learning mode.

This is a port of the benchmarked harness loop, not a re-derivation of it. The
manuscript reports WSS@95 0.679 over the 26 SYNERGY reviews for a BGE-large
embedding logistic ensembled with a TF-IDF logistic, and for that number to say
anything about the shipped product the shipped product has to run that
algorithm. Every hyperparameter below is pinned to the harness
(``ranking_cluster.py``) and changing one invalidates the reported figure:

    embeddings   BAAI/bge-large-en-v1.5, CLS pooling, L2-normalised, max_len 256
    lexical      TF-IDF, 1-2 grams, min_df=2, sublinear_tf, max_features=40000
    classifier   LogisticRegression(class_weight="balanced", max_iter=200), one
                 per view, probabilities summed
    selection    relevance-greedy (CAL): label the top-scoring unlabelled batch,
                 refit, repeat, with batch = max(1, N // 100)

Tiers run most to least faithful, and the tier that actually ran is always
returned. That matters more than it looks: a ranker that silently degrades to a
cheaper model when a dependency is missing is exactly how the benchmarked
component and the shipped component came apart, so the caller is told which one
produced the ordering and the UI names it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")

EMB_MODEL = "BAAI/bge-large-en-v1.5"
MAX_LEN = 256
EMB_BATCH = 64

# Below this many labels, or with only one class present, a supervised ranker
# has nothing to learn and the caller should stay on its cold-start ordering.
# Two is the harness's seed round (one include, one exclude); raising it would
# quietly change the protocol the 0.679 figure was measured under.
MIN_LABELS = 2

_CACHE_DIR = os.environ.get(
    "EE_RANK_CACHE", os.path.join(os.path.dirname(__file__), ".rank_cache")
)

# Embedding a corpus is the expensive step and depends only on the text, so it
# is computed once per corpus and reused for every re-rank as labels accumulate.
_EMB_CACHE: Dict[str, np.ndarray] = {}
_TFIDF_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_MODEL: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Capability probing
# ---------------------------------------------------------------------------

def _have(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def _sklearn_ready() -> bool:
    return _have("sklearn")


def _torch_ready() -> bool:
    return _have("torch") and _have("transformers")


# Exactly the files _encode() loads. Probing for these rather than for a
# complete snapshot keeps the readiness check honest: the repo also carries
# ONNX and pytorch_model.bin copies we never touch, and demanding those would
# report "not downloaded" for a cache that is in fact ready to serve.
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "vocab.txt")


def _model_cached_locally() -> bool:
    """True when the BGE weights are already on disk, so ranking will not
    silently start a 1.3 GB download in the middle of a screening session."""
    if not _torch_ready():
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(
            isinstance(try_to_load_from_cache(EMB_MODEL, f), str) for f in MODEL_FILES
        )
    except Exception:
        return False


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _corpus_key(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _encode(texts: List[str], key: str) -> np.ndarray:
    """CLS-pooled, L2-normalised BGE embeddings, cached per corpus.

    The harness ran fp16 on CUDA. MPS keeps fp32: half precision there has bitten
    us with silent NaNs, and a ranker that returns NaN scores degrades to random
    order without erroring, which is the failure mode hardest to notice.
    """
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"bge_{key}.npy")
    if os.path.exists(path):
        E = np.load(path)
        if len(E) == len(texts):
            _EMB_CACHE[key] = E
            return E

    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    with _MODEL_LOCK:
        if "tok" not in _MODEL:
            # local_files_only: rank() only reaches this branch once
            # _model_cached_locally() passed, so any hub call here would be a
            # pointless round-trip that hangs a reviewer working offline.
            # Downloading is rank_warm()'s job and nothing else's.
            _MODEL["tok"] = AutoTokenizer.from_pretrained(EMB_MODEL, local_files_only=True)
            dtype = torch.float16 if dev == "cuda" else torch.float32
            # transformers 5 renamed torch_dtype -> dtype; accept either so the
            # backend works across the 4.x/5.x split people actually have.
            try:
                m = AutoModel.from_pretrained(EMB_MODEL, dtype=dtype, local_files_only=True)
            except TypeError:
                m = AutoModel.from_pretrained(EMB_MODEL, torch_dtype=dtype, local_files_only=True)
            _MODEL["model"] = m.to(dev).eval()
    tok, model = _MODEL["tok"], _MODEL["model"]

    out: List[np.ndarray] = []
    for i in range(0, len(texts), EMB_BATCH):
        enc = tok(
            texts[i : i + EMB_BATCH],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        v = torch.nn.functional.normalize(h[:, 0], dim=1)
        out.append(v.float().cpu().numpy())

    E = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
    _EMB_CACHE[key] = E
    try:
        np.save(path, E)
    except OSError:
        pass  # a read-only cache dir must not fail the request
    return E


def _tfidf(texts: List[str], key: str):
    if key in _TFIDF_CACHE:
        return _TFIDF_CACHE[key]
    from sklearn.feature_extraction.text import TfidfVectorizer

    def build(**override):
        kw = dict(
            min_df=2,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=40000,
        )
        kw.update(override)
        return TfidfVectorizer(**kw).fit_transform(texts)

    try:
        X = build()
    except ValueError:
        # Empty vocabulary: too few documents to clear min_df=2, or nothing but
        # stop words. No SYNERGY review looks like this, but a reviewer screening
        # a handful of records with missing abstracts does, and sklearn raises
        # rather than returning an empty matrix. A weaker lexical view beats a
        # 500 in the middle of screening.
        try:
            X = build(min_df=1, stop_words=None)
        except ValueError:
            from scipy.sparse import csr_matrix

            X = csr_matrix((len(texts), 1), dtype=float)
    _TFIDF_CACHE[key] = X
    return X


def _fit_score(X, labeled: List[int], y: np.ndarray) -> np.ndarray:
    """One balanced logistic over one view, scored across the whole corpus."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(class_weight="balanced", max_iter=200)
    clf.fit(X[labeled], y[labeled])
    return clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------

class RankRecord(BaseModel):
    id: str
    title: str = ""
    text: str = ""


class RankRequest(BaseModel):
    records: List[RankRecord]
    # Reviewer decisions so far: {record id: 1 include / 0 exclude}. Records not
    # present here are the unlabelled pool being ranked.
    labels: Dict[str, int] = Field(default_factory=dict)
    # "auto" takes the most faithful tier available; "bge" refuses to silently
    # degrade and errors instead; "tfidf" pins the lexical-only tier.
    tier: str = "auto"


class RankResponse(BaseModel):
    order: List[str]
    scores: Dict[str, float]
    tier: str
    trained: bool
    reviewed: int
    includes_found: int
    predicted_remaining: int
    est_recall: Optional[float]
    batch: int
    detail: str = ""


@router.get("/rank/status")
def rank_status() -> Dict[str, Any]:
    """What the ranker can actually run right now.

    The UI calls this before offering active-learning mode so it can name the
    tier honestly, and so the 1.3 GB weight download is something the reviewer
    agrees to rather than something that happens to them mid-session.
    """
    torch_ok, model_ok = _torch_ready(), _model_cached_locally()
    return {
        "benchmarked_tier_available": torch_ok and model_ok,
        "sklearn": _sklearn_ready(),
        "torch": torch_ok,
        "model_downloaded": model_ok,
        "model": EMB_MODEL,
        "device": _device() if torch_ok else "none",
        "tier": (
            "bge+tfidf" if (torch_ok and model_ok and _sklearn_ready())
            else "tfidf" if _sklearn_ready()
            else "nb"
        ),
        "download_mb": 1340,
    }


@router.post("/rank/warm")
def rank_warm() -> Dict[str, Any]:
    """Fetch the BGE weights. Explicit, so the download is never a surprise."""
    if not _torch_ready():
        raise HTTPException(
            status_code=501,
            detail="torch and transformers are not installed in the backend "
                   "environment; run: pip install -r requirements.txt",
        )
    from huggingface_hub import snapshot_download

    snapshot_download(EMB_MODEL, allow_patterns=list(MODEL_FILES))
    return {"ok": True, "model": EMB_MODEL, "device": _device()}


@router.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    records = req.records
    n = len(records)
    if n == 0:
        raise HTTPException(status_code=400, detail="no records to rank")

    ids = [r.id for r in records]
    texts = [f"{r.title or ''} {r.text or ''}".strip() for r in records]
    idx = {rid: i for i, rid in enumerate(ids)}
    batch = max(1, n // 100)

    y = np.full(n, -1, dtype=int)
    for rid, lab in (req.labels or {}).items():
        i = idx.get(rid)
        if i is not None:
            y[i] = 1 if int(lab) == 1 else 0
    labeled = [i for i in range(n) if y[i] >= 0]
    unlabeled = [i for i in range(n) if y[i] < 0]
    includes_found = int((y == 1).sum())

    # A supervised ranker needs both classes and a few examples. Short of that
    # there is nothing to report and the client keeps its cold-start ordering,
    # which is the same position the harness is in before its seed round.
    both = bool((y == 1).any() and (y == 0).any())
    if not both or len(labeled) < MIN_LABELS:
        return RankResponse(
            order=[ids[i] for i in unlabeled],
            scores={},
            tier="cold",
            trained=False,
            reviewed=len(labeled),
            includes_found=includes_found,
            predicted_remaining=0,
            est_recall=None,
            batch=batch,
            detail=f"needs {MIN_LABELS} labels including at least one of each class",
        )

    want = (req.tier or "auto").lower()
    if not _sklearn_ready():
        raise HTTPException(
            status_code=501,
            detail="scikit-learn is not installed in the backend environment",
        )

    use_bge = want in ("auto", "bge") and _torch_ready() and _model_cached_locally()
    if want == "bge" and not use_bge:
        raise HTTPException(
            status_code=409,
            detail="the benchmarked BGE tier was requested but its weights are "
                   "not downloaded; POST /api/rank/warm first",
        )

    key = _corpus_key(texts)
    Xt = _tfidf(texts, key)
    score = _fit_score(Xt, labeled, y)
    tier, detail = "tfidf", "lexical only; not the benchmarked configuration"

    if use_bge:
        E = _encode(texts, key)
        score = _fit_score(E, labeled, y) + score  # harness sums the two views
        tier, detail = "bge+tfidf", f"benchmarked configuration on {_device()}"

    order_idx = sorted(unlabeled, key=lambda i: -float(score[i]))
    scores = {ids[i]: float(score[i]) for i in unlabeled}

    # Scores are a sum of two probabilities under bge+tfidf, so the "probably
    # relevant" line sits at 0.5 per view.
    thresh = 1.0 if tier == "bge+tfidf" else 0.5
    predicted_remaining = int(sum(1 for i in unlabeled if score[i] >= thresh))
    denom = includes_found + predicted_remaining
    est_recall = (includes_found / denom) if denom > 0 else None

    return RankResponse(
        order=[ids[i] for i in order_idx],
        scores=scores,
        tier=tier,
        trained=True,
        reviewed=len(labeled),
        includes_found=includes_found,
        predicted_remaining=predicted_remaining,
        est_recall=est_recall,
        batch=batch,
        detail=detail,
    )
