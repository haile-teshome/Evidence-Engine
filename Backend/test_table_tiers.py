"""Tests for the table-extraction ladder. No network: every rung is stubbed.

Run: Backend/.venv/bin/python -m pytest Backend/test_table_tiers.py

The ladder exists because no single reader works on every paper, and the risks
are asymmetric. A rung that invents a table is worse than one that finds none,
so the deterministic readers run first and the model runs last; and a reviewer
who sees "no tables" has to be able to tell whether the paper has none or
whether nothing ever managed to look, which is what `tried` is for.
"""
import pytest
from fastapi.testclient import TestClient

import api
import docling_tables


class TestDoclingNaming:
    """Docling finds the tables; naming them is a separate judgement, and the
    layout model gets caption-to-table wrong on a page that holds two."""

    def _t(self, caption):
        return {"caption": caption, "data": [["a"], ["b"]], "page": 1}

    def test_a_clean_sequence_of_labels_is_believed(self):
        out = docling_tables.name_tables([self._t("Table 1 Characteristics"),
                                          self._t("Table 2 Outcomes")])
        assert [t["title"] for t in out] == ["Table 1", "Table 2"]
        assert out[0]["caption"] == "Table 1 Characteristics"

    def test_two_tables_claiming_the_same_label_fall_back_to_position(self):
        out = docling_tables.name_tables([self._t("Table 2 Univariate"),
                                          self._t("Table 4 Adverse effects"),
                                          self._t("Table 4 Adverse effects")])
        assert [t["title"] for t in out] == ["Table 1", "Table 2", "Table 3"]

    def test_a_caption_that_might_be_on_the_wrong_table_is_dropped(self):
        # Worse than no caption: it names a table the reader is not looking at.
        out = docling_tables.name_tables([self._t(""), self._t("Table 4 Adverse effects")])
        assert all(t["caption"] == "" for t in out)

    def test_out_of_order_labels_are_not_believed(self):
        out = docling_tables.name_tables([self._t("Table 3 Later"), self._t("Table 1 Earlier")])
        assert [t["title"] for t in out] == ["Table 1", "Table 2"]


class TestCaptionWindows:
    """What the model is shown. A 40,000-character paper does not fit, and the
    first 3,000 characters of it are the abstract, not the tables."""

    def test_windows_start_at_the_captions(self):
        text = ("intro " * 500) + "\nTable 1\nSex: Male 167 (61.4%)\n" + ("discussion " * 500)
        wins = api._table_windows(text, size=200)
        assert len(wins) == 1
        assert "Sex: Male 167" in wins[0]

    def test_every_caption_gets_a_window(self):
        text = "\n".join(["Table 1", "a 1", "x" * 6000, "Table 2", "b 2", "y" * 6000, "Table 3", "c 3"])
        wins = api._table_windows(text, size=500)
        assert len(wins) == 3
        assert "a 1" in wins[0] and "b 2" in wins[1] and "c 3" in wins[2]

    def test_captions_close_together_share_one_window(self):
        text = "Table 1\nrow\nTable 2\nrow\n" + "z" * 100
        assert len(api._table_windows(text, size=4000)) == 1

    def test_a_paper_with_no_captions_falls_back_to_the_head(self):
        text = "No tabular material at all. " * 100
        wins = api._table_windows(text, size=120)
        assert wins == [text[:120]]

    def test_nothing_in_nothing_out(self):
        assert api._table_windows("") == []


class TestLayoutFilter:
    """A two-column page read by the text-alignment strategy is one grid per
    page. It passes every structural test, so it has to be rejected on shape."""

    def test_a_two_column_page_is_not_a_table(self):
        rows = [["investigate the effect on OS by comparing patients", "the chemoradiation cycle was"],
                ["used it for a portion of the cycle or never, and", "radiotherapy was initiated"],
                ["the total duration of treatment varied with the", "protocol used in each case"]]
        assert api._looks_like_page_layout(rows, 0.87) is True

    def test_a_real_table_in_one_column_survives(self):
        rows = [["Sample Characteristics", "n = 272"],
                ["Sex: Male", "167 (61.4%)"],
                ["Age (years)", "63 (55-71)"]]
        assert api._looks_like_page_layout(rows, 0.38) is False

    def test_a_wide_table_of_values_survives(self):
        """Single-column papers do have full-width tables; width alone is not a
        verdict, which is why both tests have to fail it."""
        rows = [["Group", "n", "Median OS", "p"],
                ["LEV", "77", "15.10", "0.37"],
                ["LAC", "20", "12.58", "0.37"]]
        assert api._looks_like_page_layout(rows, 0.92) is False


class TestTitleMatch:
    def test_the_paper_matches_itself(self):
        t = "Effects of Levetiracetam and Lacosamide on survival in glioblastoma"
        assert api._same_title(t, t.lower().replace("-", " ")) is True

    def test_a_letter_about_the_paper_does_not(self):
        """The commonest wrong answer a title search gives, and extracting the
        letter's tables would be worse than finding none."""
        real = "Effects of Levetiracetam and Lacosamide on survival and seizure control in glioblastoma"
        letter = 'Letter to the editor "Effects of levetiracetam and lacosamide on survival in glioblastoma"'
        assert api._same_title(letter, real) is False

    def test_blank_titles_never_match(self):
        assert api._same_title("", "") is False
        assert api._same_title("A trial", "") is False


class TestResolveId:
    def _hits(self, results):
        class R:
            ok = True
            status_code = 200

            def json(self_inner):
                return {"resultList": {"result": results}}
        return R()

    def test_picks_the_matching_record(self, monkeypatch):
        title = "Dopamine Receptor Antagonists, Radiation, and Cholesterol Biosynthesis"
        monkeypatch.setattr(api, "limited_get", lambda *a, **k: self._hits([
            {"pmcid": "PMC999", "title": "Letter to the editor: " + title},
            {"pmcid": "PMC8328983", "pmid": "33561287", "title": title},
        ]))
        monkeypatch.setattr(api, "cached", lambda ns, key, produce: produce())
        assert api._resolve_epmc_id(title) == "PMC8328983"

    def test_no_match_is_empty_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(api, "limited_get", lambda *a, **k: self._hits([
            {"pmcid": "PMC1", "title": "A completely different paper"},
        ]))
        monkeypatch.setattr(api, "cached", lambda ns, key, produce: produce())
        assert api._resolve_epmc_id("Dopamine receptor antagonists and radiation") == ""

    def test_a_title_too_short_to_identify_is_not_searched(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("should not have searched")

        monkeypatch.setattr(api, "limited_get", boom)
        assert api._resolve_epmc_id("Trial") == ""


class TestLadder:
    """Order and reporting. The rungs themselves are tested above and by the
    PDF readers' own suites; what matters here is which ones get to run."""

    def _client(self):
        return TestClient(api.app)

    def _req(self, **over):
        body = {"Title": "A trial of something", "URL": "", "Source": "Local PDFs",
                "full_text": "Background. " * 50}
        body.update(over)
        return body

    def _silence(self, monkeypatch, *, epmc=None, pdf=None, ai=None):
        monkeypatch.setattr(api, "_resolve_epmc_id", lambda t: "")
        monkeypatch.setattr(api, "_extract_epmc_tables", lambda i, t: epmc or [])
        monkeypatch.setattr(api, "_read_cached_pdf", lambda i, u: pdf)
        monkeypatch.setattr(api, "_fulltext_fetch_impl", lambda r: {"status": "missing"})
        monkeypatch.setattr(api.AITableExtractor, "extract_from_text",
                            staticmethod(lambda *a, **k: ai or []))

    def test_the_structured_source_wins_and_the_model_never_runs(self, monkeypatch):
        jats = [{"title": "Table 1", "type": "other", "data": [["a", "b"], ["1", "2"]],
                 "caption": "", "page": None, "tier": "jats", "deterministic": True}]

        def never(*a, **k):
            raise AssertionError("the model ran although a deterministic tier answered")

        self._silence(monkeypatch, epmc=jats)
        monkeypatch.setattr(api, "_resolve_epmc_id", lambda t: "PMC1")
        monkeypatch.setattr(api.AITableExtractor, "extract_from_text", staticmethod(never))

        body = self._client().post("/api/extract/tables", json=self._req()).json()
        assert [t["tier"] for t in body["tables"]] == ["jats"]
        assert any("Europe PMC" in line for line in body["tried"])

    def test_the_model_runs_only_when_the_readers_come_back_empty(self, monkeypatch):
        self._silence(monkeypatch, ai=[{"label": "Table 1", "caption": "",
                                        "headers": ["Group", "n"],
                                        "rows": [["LEV", "77"], ["LAC", "20"]]}])
        body = self._client().post("/api/extract/tables", json=self._req()).json()
        assert [t["tier"] for t in body["tables"]] == ["llm"]
        assert body["tables"][0]["deterministic"] is False

    def test_an_invented_one_row_table_is_dropped(self, monkeypatch):
        """The model's output goes through the same shape test as every other
        tier's, because this is the rung that can make things up."""
        self._silence(monkeypatch, ai=[{"label": "Table 1", "headers": ["Group"], "rows": []}])
        body = self._client().post("/api/extract/tables", json=self._req()).json()
        assert body["tables"] == []

    def test_it_says_what_it_tried_when_it_finds_nothing(self, monkeypatch):
        self._silence(monkeypatch)
        body = self._client().post("/api/extract/tables", json=self._req()).json()
        assert body["tables"] == []
        assert body["tried"], "a reviewer cannot tell an empty paper from a failed look"
        assert any("no PDF" in line for line in body["tried"])

    def test_a_paper_with_no_text_is_reported_as_such(self, monkeypatch):
        self._silence(monkeypatch)
        body = self._client().post("/api/extract/tables",
                                   json=self._req(full_text="", abstract="")).json()
        assert any("no text" in line for line in body["tried"])


class TestReadingAPdf:
    """Reading a document the browser could not.

    A scanned PDF imports cleanly and contains no words, so every stage that
    reads text would skip it. The file is already on the server, so the server
    reads it, OCR included. The reviewer is never asked to run OCR themselves.
    """

    def _client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "_PDF_CACHE_DIR", tmp_path)
        return TestClient(api.app)

    def test_a_text_layer_is_used_as_is(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        monkeypatch.setattr(api, "_extract_text_from_pdf", lambda b: "Methods. " * 60)
        c.post("/api/fulltext/pdf?paper_id=p1", content=b"%PDF-1.7 x")

        body = c.post("/api/fulltext/read?paper_id=p1").json()
        assert body["method"] == "text-layer"
        assert "Methods." in body["text"]

    def test_a_scanned_file_is_read_from_its_pages(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        monkeypatch.setattr(api, "_extract_text_from_pdf", lambda b: "")
        import docling_tables
        monkeypatch.setattr(docling_tables, "available", lambda: True)
        monkeypatch.setattr(docling_tables, "extract_text", lambda b: "Sample Characteristics n = 272")
        c.post("/api/fulltext/pdf?paper_id=p1", content=b"%PDF-1.7 x")

        body = c.post("/api/fulltext/read?paper_id=p1").json()
        assert body["method"] == "ocr"
        assert "272" in body["text"]

    def test_it_says_so_when_nothing_can_read_the_file(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        monkeypatch.setattr(api, "_extract_text_from_pdf", lambda b: "")
        import docling_tables
        monkeypatch.setattr(docling_tables, "available", lambda: False)
        c.post("/api/fulltext/pdf?paper_id=p1", content=b"%PDF-1.7 x")

        body = c.post("/api/fulltext/read?paper_id=p1").json()
        assert body["method"] == "none"
        assert body["text"] == ""
        assert "OCR" in body["detail"]

    def test_a_paper_with_no_file_here_is_a_404(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        assert c.post("/api/fulltext/read?paper_id=nothing").status_code == 404

    def test_the_file_can_be_sent_in_the_request(self, tmp_path, monkeypatch):
        # No upload first: a caller that already has the bytes can just send them.
        c = self._client(tmp_path, monkeypatch)
        monkeypatch.setattr(api, "_extract_text_from_pdf", lambda b: "Results. " * 60)
        body = c.post("/api/fulltext/read", content=b"%PDF-1.7 body").json()
        assert body["method"] == "text-layer"


class TestPdfUpload:
    """The reviewer's own file has to reach the server, or every reader that
    works on the file is skipped for exactly the papers they supplied."""

    def test_a_pdf_is_cached_under_its_paper_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "_PDF_CACHE_DIR", tmp_path)
        c = TestClient(api.app)
        r = c.post("/api/fulltext/pdf?paper_id=p1", content=b"%PDF-1.7 body")
        assert r.status_code == 200
        assert api._read_cached_pdf("p1", None) == b"%PDF-1.7 body"

    def test_anything_that_is_not_a_pdf_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "_PDF_CACHE_DIR", tmp_path)
        c = TestClient(api.app)
        r = c.post("/api/fulltext/pdf?paper_id=p1", content=b"<html>not a pdf</html>")
        assert r.status_code == 400
        assert not list(tmp_path.iterdir())
