"""Tests for the screening-prioritisation ranker.

Run: pytest test_ranking.py

This module is unusual in that its constants are load-bearing published facts.
The manuscript reports WSS@95 0.679 over the 26 SYNERGY reviews, and that number
describes one specific configuration: BGE-large CLS embeddings, TF-IDF 1-2 grams
at min_df=2 with sublinear scaling, a balanced logistic per view, probabilities
summed, and a batch of N//100. Change any of those and the figure no longer
describes the shipped system.

That is not hypothetical. The app previously shipped an in-browser Naive Bayes
while the paper described the ensemble, and nothing failed, because nothing
checked. So alongside the ordinary behavioural tests there are tests that simply
assert the hyperparameters still say what the benchmark ran with, and tests that
the tier is reported honestly when a dependency is missing.
"""
import numpy as np
import pytest
from fastapi import HTTPException

import ranking
from ranking import RankRecord, RankRequest, rank, rank_status


# A tiny two-topic corpus. Diabetes records are the "relevant" class; the
# structural-engineering ones are background. Lexically separable on purpose, so
# a working ranker has to surface the unlabelled diabetes records.
DIABETES = [
    "Metformin therapy in type 2 diabetes randomised controlled trial",
    "Glycaemic control and HbA1c outcomes in diabetic adults",
    "Insulin glargine versus metformin for type 2 diabetes mellitus",
    "Diabetes prevention through metformin in prediabetic patients",
    "HbA1c reduction with metformin monotherapy diabetes cohort",
]
BRIDGES = [
    "Fatigue cracking in welded steel bridge girders",
    "Concrete creep and shrinkage modelling for bridge decks",
    "Seismic retrofit of reinforced concrete bridge piers",
    "Corrosion of prestressing tendons in post-tensioned bridges",
    "Load rating of steel truss bridges under traffic",
]


def corpus():
    recs, truth = [], {}
    for i, t in enumerate(DIABETES):
        recs.append(RankRecord(id=f"d{i}", title=t, text="")); truth[f"d{i}"] = 1
    for i, t in enumerate(BRIDGES):
        recs.append(RankRecord(id=f"b{i}", title=t, text="")); truth[f"b{i}"] = 0
    return recs, truth


def do(records, labels, tier="tfidf"):
    return rank(RankRequest(records=records, labels=labels, tier=tier))


# ---------------------------------------------------------------------------
# The published configuration. These assertions are the point of the module.
# ---------------------------------------------------------------------------

class TestBenchmarkedConfiguration:
    def test_embedding_model_is_the_benchmarked_one(self):
        assert ranking.EMB_MODEL == "BAAI/bge-large-en-v1.5"

    def test_sequence_length_matches_the_harness(self):
        assert ranking.MAX_LEN == 256

    def test_seed_round_is_two_labels(self):
        """The harness fits from one include plus one exclude. A higher floor
        would silently change the protocol the 0.679 was measured under."""
        assert ranking.MIN_LABELS == 2

    def test_tfidf_hyperparameters_are_pinned(self):
        import inspect
        src = inspect.getsource(ranking._tfidf)
        for token in ("min_df=2", "ngram_range=(1, 2)", 'stop_words="english"',
                      "sublinear_tf=True", "max_features=40000"):
            assert token in src, f"TF-IDF no longer matches the harness: {token}"

    def test_classifier_hyperparameters_are_pinned(self):
        import inspect
        src = inspect.getsource(ranking._fit_score)
        assert 'class_weight="balanced"' in src
        assert "max_iter=200" in src

    def test_batch_is_one_percent_of_the_corpus(self):
        recs, _ = corpus()
        big = [RankRecord(id=f"x{i}", title=f"record {i}", text="") for i in range(500)]
        r = do(big, {"x0": 1, "x1": 0})
        assert r.batch == 5

    def test_batch_never_drops_to_zero_on_a_small_corpus(self):
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0}).batch == 1


# ---------------------------------------------------------------------------
# Does it actually learn?
# ---------------------------------------------------------------------------

class TestRanking:
    def test_surfaces_relevant_records_from_one_example_of_each(self):
        recs, truth = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        top_half = r.order[: len(r.order) // 2]
        assert sum(truth[i] for i in top_half) > len(top_half) / 2

    def test_the_single_top_ranked_record_is_relevant(self):
        recs, truth = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        assert truth[r.order[0]] == 1

    def test_returns_only_unlabelled_records(self):
        """A reviewer must never be handed a card they already decided."""
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0, "d1": 1})
        assert "d0" not in r.order and "b0" not in r.order and "d1" not in r.order
        assert len(r.order) == 7

    def test_every_unlabelled_record_appears_exactly_once(self):
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        assert sorted(r.order) == sorted([f"d{i}" for i in range(1, 5)] + [f"b{i}" for i in range(1, 5)])

    def test_scores_are_ordered_consistently_with_the_order(self):
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        vals = [r.scores[i] for i in r.order]
        assert vals == sorted(vals, reverse=True)

    def test_scores_cover_exactly_the_unlabelled_pool(self):
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        assert set(r.scores) == set(r.order)

    def test_more_labels_do_not_break_the_ordering(self):
        recs, truth = corpus()
        r = do(recs, {"d0": 1, "d1": 1, "d2": 1, "b0": 0, "b1": 0, "b2": 0})
        assert truth[r.order[0]] == 1

    def test_is_deterministic(self):
        recs, _ = corpus()
        a = do(recs, {"d0": 1, "b0": 0})
        b = do(recs, {"d0": 1, "b0": 0})
        assert a.order == b.order


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_when_no_labels(self):
        recs, _ = corpus()
        assert do(recs, {}).tier == "cold"

    def test_cold_when_only_includes_are_labelled(self):
        """A one-class fit has no decision boundary to learn."""
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "d1": 1}).tier == "cold"

    def test_cold_when_only_excludes_are_labelled(self):
        recs, _ = corpus()
        assert do(recs, {"b0": 0, "b1": 0}).tier == "cold"

    def test_cold_response_is_not_marked_trained(self):
        recs, _ = corpus()
        assert do(recs, {}).trained is False

    def test_cold_still_returns_the_unlabelled_pool(self):
        recs, _ = corpus()
        assert len(do(recs, {"d0": 1}).order) == 9

    def test_cold_explains_what_is_missing(self):
        recs, _ = corpus()
        assert "label" in do(recs, {}).detail.lower()

    def test_one_of_each_is_enough_to_train(self):
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0}).trained is True


# ---------------------------------------------------------------------------
# Tier reporting. The property that keeps benchmark and product in step.
# ---------------------------------------------------------------------------

class TestTierHonesty:
    def test_tfidf_tier_is_named_as_such(self):
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0}, tier="tfidf").tier == "tfidf"

    def test_tfidf_tier_says_it_is_not_the_benchmarked_configuration(self):
        recs, _ = corpus()
        assert "not the benchmarked" in do(recs, {"d0": 1, "b0": 0}, tier="tfidf").detail

    def test_auto_degrades_to_tfidf_when_torch_is_absent(self, monkeypatch):
        monkeypatch.setattr(ranking, "_torch_ready", lambda: False)
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0}, tier="auto").tier == "tfidf"

    def test_auto_degrades_to_tfidf_when_the_weights_are_not_downloaded(self, monkeypatch):
        monkeypatch.setattr(ranking, "_model_cached_locally", lambda: False)
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0}, tier="auto").tier == "tfidf"

    def test_explicitly_requesting_bge_refuses_to_degrade_silently(self, monkeypatch):
        """The whole failure mode in one test: asking for the benchmarked ranker
        and quietly receiving a different one is what put the paper out of step
        with the product."""
        monkeypatch.setattr(ranking, "_model_cached_locally", lambda: False)
        recs, _ = corpus()
        with pytest.raises(HTTPException) as e:
            do(recs, {"d0": 1, "b0": 0}, tier="bge")
        assert e.value.status_code == 409
        assert "warm" in e.value.detail

    def test_missing_sklearn_is_an_error_not_a_downgrade(self, monkeypatch):
        monkeypatch.setattr(ranking, "_sklearn_ready", lambda: False)
        recs, _ = corpus()
        with pytest.raises(HTTPException) as e:
            do(recs, {"d0": 1, "b0": 0})
        assert e.value.status_code == 501

    def test_a_trained_response_always_names_a_tier(self, monkeypatch):
        """Stubbed embeddings, because what is under test is the tier bookkeeping
        rather than BGE itself, and a unit test should not need 1.2 GB of
        weights on disk to run."""
        recs, _ = corpus()
        monkeypatch.setattr(ranking, "_torch_ready", lambda: True)
        monkeypatch.setattr(
            ranking, "_encode",
            lambda texts, key: np.random.default_rng(0).normal(size=(len(texts), 8)))
        for cached in (True, False):
            monkeypatch.setattr(ranking, "_model_cached_locally", lambda c=cached: c)
            r = do(recs, {"d0": 1, "b0": 0}, tier="auto")
            assert r.tier == ("bge+tfidf" if cached else "tfidf")


# ---------------------------------------------------------------------------
# Recall estimate. It drives a "safe to stop" prompt, so its arithmetic matters.
# ---------------------------------------------------------------------------

class TestRecallEstimate:
    def test_counts_includes_found(self):
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "d1": 1, "b0": 0}).includes_found == 2

    def test_estimate_is_found_over_found_plus_predicted(self):
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        expected = r.includes_found / (r.includes_found + r.predicted_remaining)
        assert r.est_recall == pytest.approx(expected)

    def test_estimate_is_a_probability(self):
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0})
        assert 0.0 <= r.est_recall <= 1.0

    def test_estimate_is_none_when_there_is_nothing_to_divide(self):
        recs, _ = corpus()
        r = do(recs, {})
        assert r.est_recall is None

    def test_reviewed_counts_every_label_not_only_includes(self):
        recs, _ = corpus()
        assert do(recs, {"d0": 1, "b0": 0, "b1": 0}).reviewed == 3


# ---------------------------------------------------------------------------
# Caching. Embeddings are the expensive part and must be reused, but never
# across different corpora.
# ---------------------------------------------------------------------------

class TestCorpusKey:
    def test_same_text_gives_the_same_key(self):
        assert ranking._corpus_key(["a", "b"]) == ranking._corpus_key(["a", "b"])

    def test_different_text_gives_a_different_key(self):
        assert ranking._corpus_key(["a", "b"]) != ranking._corpus_key(["a", "c"])

    def test_order_is_part_of_the_key(self):
        """Embeddings are positional, so a reordered corpus must not reuse them."""
        assert ranking._corpus_key(["a", "b"]) != ranking._corpus_key(["b", "a"])

    def test_boundaries_are_unambiguous(self):
        """Without a separator, ["ab","c"] and ["a","bc"] would collide and one
        corpus would silently be scored with another's embeddings."""
        assert ranking._corpus_key(["ab", "c"]) != ranking._corpus_key(["a", "bc"])

    def test_tfidf_matrix_is_cached_per_corpus(self):
        texts = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
        key = ranking._corpus_key(texts)
        ranking._TFIDF_CACHE.pop(key, None)
        first = ranking._tfidf(texts, key)
        assert ranking._tfidf(texts, key) is first


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_empty_record_list_is_rejected(self):
        with pytest.raises(HTTPException) as e:
            do([], {})
        assert e.value.status_code == 400

    def test_labels_for_unknown_ids_are_ignored(self):
        """Stale ids arrive whenever a reviewer removes records mid-session."""
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0, "ghost": 1})
        assert r.reviewed == 2

    def test_title_and_text_are_both_used(self):
        recs = [RankRecord(id="a", title="", text=t) for t in DIABETES]
        recs += [RankRecord(id=f"b{i}", title="", text=t) for i, t in enumerate(BRIDGES)]
        for i, r in enumerate(recs[:5]):
            r.id = f"d{i}"
        out = do(recs, {"d0": 1, "b0": 0})
        assert out.trained is True

    def test_missing_text_does_not_crash(self):
        recs = [RankRecord(id=f"r{i}") for i in range(6)]
        assert do(recs, {"r0": 1, "r1": 0}).trained is True

    def test_non_binary_label_values_are_coerced_to_exclude(self):
        """Anything that is not exactly 1 is a negative, so a stray 2 cannot
        quietly become a positive training example."""
        recs, _ = corpus()
        r = do(recs, {"d0": 1, "b0": 0, "b1": 7})
        assert r.includes_found == 1


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

class TestStatus:
    def test_reports_every_field_the_ui_needs(self):
        s = rank_status()
        for k in ("benchmarked_tier_available", "sklearn", "torch",
                  "model_downloaded", "model", "device", "tier", "download_mb"):
            assert k in s

    def test_names_the_benchmarked_model(self):
        assert rank_status()["model"] == "BAAI/bge-large-en-v1.5"

    def test_benchmarked_tier_requires_both_torch_and_weights(self, monkeypatch):
        monkeypatch.setattr(ranking, "_torch_ready", lambda: True)
        monkeypatch.setattr(ranking, "_model_cached_locally", lambda: False)
        assert rank_status()["benchmarked_tier_available"] is False

    def test_advertises_the_download_size(self, monkeypatch):
        """The reviewer agrees to 1.3 GB rather than discovering it mid-session."""
        assert rank_status()["download_mb"] > 1000

    def test_tier_falls_back_through_the_chain(self, monkeypatch):
        monkeypatch.setattr(ranking, "_torch_ready", lambda: False)
        monkeypatch.setattr(ranking, "_sklearn_ready", lambda: True)
        assert rank_status()["tier"] == "tfidf"
        monkeypatch.setattr(ranking, "_sklearn_ready", lambda: False)
        assert rank_status()["tier"] == "nb"
