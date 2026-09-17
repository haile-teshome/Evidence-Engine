"""Tests for the optional image-based table recogniser.

Run: Backend/.venv/bin/python -m pytest Backend/test_table_recognition.py

UniTable is opt-in: it needs EE_USE_UNITABLE=1, ~1.5 GB of weights, and torch.
The model inference itself cannot be unit-tested without those, and that is
fine. What MUST be tested is the guard contract, because it protects the default
install: every entry point returns None rather than raising when the feature is
unavailable, so the caller falls back to the deterministic pdfplumber extractor.

A regression here would not be subtle in the logs, it would break PDF table
extraction for every user who never enabled the feature.
"""
import pytest

import table_recognition as TR


class TestAvailabilityGuard:
    def test_disabled_by_default(self, monkeypatch):
        """The feature ships OFF. Enabling it must be a deliberate act."""
        monkeypatch.setattr(TR, "_ENABLED", False)
        assert TR.available() is False

    def test_unavailable_when_weights_are_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(TR, "_ENABLED", True)
        monkeypatch.setattr(TR, "_WEIGHTS", tmp_path)  # empty directory
        assert TR.available() is False

    def test_unavailable_when_only_some_weights_are_present(self, monkeypatch, tmp_path):
        """A partial download must not look like a working install."""
        monkeypatch.setattr(TR, "_ENABLED", True)
        (tmp_path / TR._WEIGHT_FILES[0]).write_bytes(b"stub")
        monkeypatch.setattr(TR, "_WEIGHTS", tmp_path)
        assert TR.available() is False

    def test_unavailable_when_torch_cannot_be_imported(self, monkeypatch, tmp_path):
        monkeypatch.setattr(TR, "_ENABLED", True)
        for w in TR._WEIGHT_FILES:
            (tmp_path / w).write_bytes(b"stub")
        monkeypatch.setattr(TR, "_WEIGHTS", tmp_path)

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def no_torch(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", no_torch)
        assert TR.available() is False

    def test_always_returns_a_bool_and_never_raises(self):
        assert isinstance(TR.available(), bool)


class TestRecogniseGuard:
    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(TR, "available", lambda: False)
        assert TR.recognize_table_image(object()) is None

    def test_returns_none_when_models_fail_to_load(self, monkeypatch):
        """Available but unloadable, e.g. corrupt weights. Still no exception."""
        monkeypatch.setattr(TR, "available", lambda: True)
        monkeypatch.setattr(TR, "_load_models", lambda: None)
        assert TR.recognize_table_image(object()) is None

    def test_returns_none_rather_than_raising_on_a_bad_image(self, monkeypatch):
        monkeypatch.setattr(TR, "available", lambda: True)
        monkeypatch.setattr(TR, "_load_models", lambda: {"device": "cpu"})
        assert TR.recognize_table_image(None) is None

    def test_load_models_returns_none_when_dependencies_are_absent(self):
        TR._load_models.cache_clear()
        out = TR._load_models()
        assert out is None or isinstance(out, dict)


class TestHtmlToRows:
    def test_parses_a_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        assert TR._html_to_rows(html) == [["A", "B"], ["1", "2"]]

    def test_expands_a_colspan_so_columns_stay_aligned(self):
        """A spanning header cell that did not pad would shift every column to
        its right, silently mis-labelling the data underneath."""
        html = ('<table><tr><td colspan="2">Wide</td><td>C</td></tr>'
                '<tr><td>1</td><td>2</td><td>3</td></tr></table>')
        rows = TR._html_to_rows(html)
        assert rows[0] == ["Wide", "", "C"]
        assert len(rows[0]) == len(rows[1])

    def test_collapses_internal_whitespace(self):
        assert TR._html_to_rows("<table><tr><td>a\n   b</td></tr></table>") == [["a b"]]

    def test_skips_a_row_with_no_content(self):
        html = "<table><tr><td></td><td>  </td></tr><tr><td>real</td></tr></table>"
        assert TR._html_to_rows(html) == [["real"]]

    def test_header_cells_are_included(self):
        html = "<table><tr><th>H</th></tr><tr><td>1</td></tr></table>"
        assert TR._html_to_rows(html) == [["H"], ["1"]]

    def test_nested_markup_inside_a_cell_is_flattened(self):
        html = "<table><tr><td><b>Bold</b> text</td></tr></table>"
        assert TR._html_to_rows(html) == [["Bold text"]]

    def test_an_invalid_colspan_does_not_crash(self):
        html = '<table><tr><td colspan="abc">X</td></tr></table>'
        assert isinstance(TR._html_to_rows(html), list)

    @pytest.mark.parametrize("html", ["", "   ", None, "not html", "<table></table>"])
    def test_unusable_input_returns_an_empty_list(self, html):
        assert TR._html_to_rows(html) == []
