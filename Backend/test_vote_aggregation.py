"""Tests for multi-agent vote aggregation.

Run: Backend/.venv/bin/python -m pytest Backend/test_vote_aggregation.py

This is the other place a verdict is decided. `_decide_from_votes` handles the
per-element panel; these two aggregators turn a set of independent criterion
agents into one include/exclude call.

The governing design is deliberately permissive: default to INCLUDE, and exclude
only on an explicit exclusion violation or on total failure across every PICO and
inclusion criterion. That is correct for a first-pass screen, where a false
exclude is unrecoverable and a false include costs one full-text read. These
tests pin that asymmetry so it cannot be tightened by accident.
"""
import pytest

from models import Paper, PICOCriteria
from utils import AgentVote, FullTextAgentVote, ScreeningOrchestrator, FullTextOrchestrator


def vote(paper_id="p1", agent_type="PICO_P", met=True, criterion="c",
         confidence=0.9, evidence="quote", reasoning="because"):
    return AgentVote(agent_name=f"agent-{agent_type}", agent_type=agent_type,
                     criterion=criterion, paper_id=paper_id, met=met,
                     confidence=confidence, evidence=evidence, reasoning=reasoning)


def ft_vote(paper_id="p1", agent_type="INCLUSION", met=True, criterion="c",
            confidence=0.9, evidence="quote", reasoning="because"):
    return FullTextAgentVote(agent_name=f"agent-{agent_type}", agent_type=agent_type,
                             criterion=criterion, paper_id=paper_id, met=met,
                             confidence=confidence, evidence=evidence, reasoning=reasoning)


@pytest.fixture
def orch():
    return ScreeningOrchestrator(PICOCriteria(population="adults", intervention="drug"),
                                 ["Published after 2010"], ["Animal studies"], "stub-model")


PAPER = Paper(source="PubMed", id="p1", title="T", abstract="A", url="")


class TestAbstractAggregation:
    def test_all_criteria_met_includes(self, orch):
        out = orch._aggregate_results([PAPER], [vote(met=True), vote(agent_type="INCLUSION")])
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_an_explicit_exclusion_violation_excludes(self, orch):
        """A matched exclusion criterion is the one decisive signal."""
        out = orch._aggregate_results([PAPER], [
            vote(met=True), vote(agent_type="EXCLUSION", met=True, criterion="Animal studies"),
        ])
        assert out[0]["Decision"].upper() == "EXCLUDE"

    def test_an_unmatched_exclusion_does_not_exclude(self, orch):
        out = orch._aggregate_results([PAPER], [
            vote(met=True), vote(agent_type="EXCLUSION", met=False),
        ])
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_total_failure_across_pico_and_inclusion_excludes(self, orch):
        out = orch._aggregate_results([PAPER], [
            vote(agent_type="PICO_P", met=False),
            vote(agent_type="PICO_I", met=False),
            vote(agent_type="INCLUSION", met=False),
        ])
        assert out[0]["Decision"].upper() == "EXCLUDE"

    def test_partial_pico_failure_still_includes(self, orch):
        """One criterion failing is not enough. Borderline records go forward."""
        out = orch._aggregate_results([PAPER], [
            vote(agent_type="PICO_P", met=True),
            vote(agent_type="PICO_I", met=False),
        ])
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_pico_failure_alone_does_not_exclude_when_inclusion_passes(self, orch):
        out = orch._aggregate_results([PAPER], [
            vote(agent_type="PICO_P", met=False),
            vote(agent_type="PICO_I", met=False),
            vote(agent_type="INCLUSION", met=True),
        ])
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_no_votes_at_all_still_includes(self, orch):
        """A total agent failure must not silently drop the paper. Recall is the
        property worth protecting at abstract stage."""
        out = orch._aggregate_results([PAPER], [])
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_every_paper_gets_exactly_one_result(self, orch):
        papers = [Paper(source="s", id=f"p{i}", title="T", abstract="A", url="")
                  for i in range(5)]
        out = orch._aggregate_results(papers, [vote(paper_id="p2", met=False)])
        assert len(out) == 5
        assert {r["paper_id"] if "paper_id" in r else r.get("ID") for r in out}

    def test_votes_are_matched_to_the_right_paper(self, orch):
        """A misrouted vote would exclude the wrong study."""
        a = Paper(source="s", id="pa", title="A", abstract="", url="")
        b = Paper(source="s", id="pb", title="B", abstract="", url="")
        out = orch._aggregate_results([a, b], [
            vote(paper_id="pa", agent_type="EXCLUSION", met=True),
        ])
        by_title = {r.get("Title"): r["Decision"].upper() for r in out}
        assert by_title["A"] == "EXCLUDE"
        assert by_title["B"] == "INCLUDE"

    def test_result_carries_a_reason(self, orch):
        out = orch._aggregate_results([PAPER], [
            vote(agent_type="EXCLUSION", met=True, criterion="Animal studies"),
        ])
        assert (out[0].get("Reason") or out[0].get("reason") or "").strip()

    def test_a_vote_for_an_unknown_paper_is_ignored_safely(self, orch):
        out = orch._aggregate_results([PAPER], [vote(paper_id="ghost", met=False)])
        assert len(out) == 1
        assert out[0]["Decision"].upper() == "INCLUDE"

    def test_aggregation_is_deterministic(self, orch):
        votes = [vote(agent_type="PICO_P", met=False), vote(agent_type="INCLUSION", met=True)]
        first = orch._aggregate_results([PAPER], votes)[0]["Decision"]
        for _ in range(3):
            assert orch._aggregate_results([PAPER], votes)[0]["Decision"] == first

    def test_empty_paper_list_yields_no_results(self, orch):
        assert orch._aggregate_results([], [vote()]) == []


class TestFullTextAggregation:
    @pytest.fixture
    def ft(self):
        return FullTextOrchestrator(["Reports an outcome"], ["Not primary research"], "stub")

    PAPER_D = {"paper_id": "p1", "Title": "T", "Abstract": "A"}

    def test_all_met_includes(self, ft):
        out = ft._aggregate_results(self.PAPER_D, [ft_vote(met=True)])
        assert str(out.get("decision", out.get("Decision", ""))).lower().startswith("inc")

    def test_an_exclusion_violation_excludes(self, ft):
        out = ft._aggregate_results(self.PAPER_D, [
            ft_vote(agent_type="EXCLUSION", met=True, criterion="Not primary research"),
        ])
        assert str(out.get("decision", out.get("Decision", ""))).lower().startswith("exc")

    def test_no_votes_does_not_crash(self, ft):
        assert isinstance(ft._aggregate_results(self.PAPER_D, []), dict)

    def test_result_carries_a_reason(self, ft):
        out = ft._aggregate_results(self.PAPER_D, [ft_vote(met=False)])
        assert (out.get("reason") or out.get("Reason") or "").strip()

    def test_is_deterministic(self, ft):
        votes = [ft_vote(met=False), ft_vote(agent_type="EXCLUSION", met=False)]
        first = ft._aggregate_results(self.PAPER_D, votes)
        assert ft._aggregate_results(self.PAPER_D, votes) == first


# ---------------------------------------------------------------------------
# Agent response parsing. A malformed reply must never become a confident vote.
# ---------------------------------------------------------------------------

class TestAgentResponseParsing:
    @pytest.fixture
    def agent(self):
        from utils import CriterionAgent
        return CriterionAgent("population-agent", "PICO_P", "adults", "stub-model")

    def test_error_vote_does_not_claim_the_criterion_failed(self, agent):
        """An agent that crashed knows nothing. Recording "not met" would turn a
        transport error into an exclusion."""
        v = agent._error_vote(PAPER)
        assert isinstance(v, AgentVote)
        assert v.confidence == 0 or v.met is True

    @pytest.mark.parametrize("content", ["", "not json", "{}", "[]", None])
    def test_unparseable_response_yields_votes_not_an_exception(self, agent, content):
        out = agent._parse_response(content, [PAPER])
        assert isinstance(out, list)
        assert all(isinstance(v, AgentVote) for v in out)

    def test_one_vote_per_paper_even_from_a_partial_response(self, agent):
        papers = [Paper(source="s", id=f"p{i}", title="T", abstract="A", url="")
                  for i in range(3)]
        out = agent._parse_response('[{"paper_id":"p0","met":true}]', papers)
        assert len(out) == len(papers)

    def test_a_well_formed_response_is_parsed(self, agent):
        out = agent._parse_single_response(
            '{"met": true, "confidence": 0.8, "evidence": "q", "reasoning": "r"}', PAPER)
        assert isinstance(out, AgentVote)
        assert out.paper_id == PAPER.id
