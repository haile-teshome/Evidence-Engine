"""Tests for turning an extraction workbook into tabbed form.

Run: Backend/.venv/bin/python -m pytest Backend/test_extraction_tabs.py

A real template is several tabs: characteristics for every study, tabs that only
apply to one design, outcomes for every study again. The case worth guarding is
the rule on a conditional tab. It is written by a model from tab names, and a
rule whose values do not match what gets extracted empties the tab silently:
the reviewer sees an RCT tab with no RCTs in it and no error anywhere.
"""
import pytest

import api


class FakeModel:
    """Returns canned JSON, so no network and no real model is needed."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        body = self.replies.pop(0) if self.replies else "{}"
        return type("R", (), {"content": body})()


SHEETS = [
    {"name": "Study characteristics", "columns": [
        {"header": "Study design", "samples": ["RCT", "RCT", "Cohort", "RCT", "Cohort"]},
        {"header": "N enrolled", "samples": ["573", "637"]}]},
    {"name": "RCT details", "columns": [
        {"header": "Blinding", "samples": ["Open", "Double"]}]},
    {"name": "Outcomes", "columns": [
        {"header": "Median OS (months)", "samples": ["14.6", "16.1"]}]},
]


def _post(client, sheets):
    return client.post("/api/extraction/infer-form", json={"sheets": sheets}).json()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(api.app)


class TestShape:
    def test_each_sheet_becomes_a_tab(self, client, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda m: None))
        out = _post(client, SHEETS)
        assert [t["name"] for t in out["tabs"]] == \
            ["Study characteristics", "RCT details", "Outcomes"]

    def test_fields_are_also_returned_flat(self, client, monkeypatch):
        """So a caller that only wants one form keeps working."""
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda m: None))
        out = _post(client, SHEETS)
        assert len(out["fields"]) == sum(len(t["fields"]) for t in out["tabs"])

    def test_a_single_sheet_request_still_works(self, client, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda m: None))
        out = client.post("/api/extraction/infer-form", json={
            "columns": [{"header": "Sample size", "samples": ["12", "40"]}],
            "sheet_name": "my sheet",
        }).json()
        assert len(out["tabs"]) == 1
        assert out["tabs"][0]["name"] == "my sheet"

    def test_nothing_usable_is_not_an_error(self, client, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda m: None))
        out = _post(client, [{"name": "Empty", "columns": []}])
        assert out == {"tabs": [], "fields": [], "source": "empty"}

    def test_without_a_model_no_tab_gets_a_rule(self, client, monkeypatch):
        """A rule is a judgement; the heuristic path does not make one up."""
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda m: None))
        out = _post(client, SHEETS)
        assert all(t["when"] is None for t in out["tabs"])
        assert out["source"] == "heuristic"


class TestTabRules:
    def _run(self, monkeypatch, rule_json):
        # _suggest_tab_rules is called directly here, so the only reply it asks
        # for is the rule one. (The endpoint consumes a field reply per tab
        # first; that path is covered by TestShape.)
        model = FakeModel(rule_json)
        tabs = [
            {"name": "Study characteristics", "when": None, "fields": [
                {"label": "Study design", "type": "select",
                 "options": ["RCT", "Cohort"], "description": "", "include": True}]},
            {"name": "RCT details", "when": None, "fields": [
                {"label": "Blinding", "type": "text",
                 "options": [], "description": "", "include": True}]},
        ]
        api._suggest_tab_rules(model, tabs)
        return tabs, model

    def test_a_rule_is_attached_to_the_conditional_tab(self, monkeypatch):
        tabs, _ = self._run(monkeypatch, '{"tabs": [{"name": "RCT details", '
                            '"all_studies": false, "deciding_column": "Study design", '
                            '"values": ["RCT"]}]}')
        assert tabs[0]["when"] is None                     # all-studies tab untouched
        assert tabs[1]["when"] == {"label": "Study design", "equals": ["RCT"]}

    def test_a_value_the_field_cannot_take_is_snapped_to_one_it_can(self, monkeypatch):
        """The model writes "rct"; the field says "RCT". Same thing, must match."""
        tabs, _ = self._run(monkeypatch, '{"tabs": [{"name": "RCT details", '
                            '"all_studies": false, "deciding_column": "Study design", '
                            '"values": ["rct"]}]}')
        assert tabs[1]["when"]["equals"] == ["RCT"]

    def test_a_rule_that_could_never_match_is_dropped(self, monkeypatch):
        """"Randomised controlled trial" is not one of the options, so a tab
        gated on it would be permanently empty. No rule is the safer answer."""
        tabs, _ = self._run(monkeypatch, '{"tabs": [{"name": "RCT details", '
                            '"all_studies": false, "deciding_column": "Study design", '
                            '"values": ["Randomised controlled trial"]}]}')
        assert tabs[1]["when"] is None

    def test_a_rule_on_a_column_that_does_not_exist_is_dropped(self, monkeypatch):
        tabs, _ = self._run(monkeypatch, '{"tabs": [{"name": "RCT details", '
                            '"all_studies": false, "deciding_column": "Allocation", '
                            '"values": ["Adequate"]}]}')
        assert tabs[1]["when"] is None

    def test_the_options_are_shown_to_the_model(self, monkeypatch):
        """Otherwise it invents its own wording for the same category."""
        _, model = self._run(monkeypatch, '{"tabs": []}')
        assert "RCT" in model.prompts[-1] and "KNOWN COLUMN VALUES" in model.prompts[-1]

    def test_a_broken_reply_leaves_every_tab_unrestricted(self, monkeypatch):
        tabs, _ = self._run(monkeypatch, "not json at all")
        assert all(t["when"] is None for t in tabs)

    def test_one_tab_is_never_given_a_rule(self, monkeypatch):
        """With a single tab there is nothing to be conditional against."""
        model = FakeModel('{"tabs": [{"name": "Only", "all_studies": false, '
                          '"deciding_column": "x", "values": ["y"]}]}')
        tabs = [{"name": "Only", "fields": [], "when": None}]
        api._suggest_tab_rules(model, tabs)
        assert tabs[0]["when"] is None
        assert not model.prompts        # it does not even ask
