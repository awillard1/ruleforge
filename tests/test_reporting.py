"""Tests for ruleforge/reporting.py"""

import json
import pytest
from ruleforge.reporting import Report, Reporter, _html_escape


@pytest.fixture
def sample_report():
    return Report(
        title="Test Report",
        config={"workers": 4, "seed": 42},
        analysis={
            "total_lines": 1000,
            "valid_lines": 950,
            "invalid_lines": 50,
            "unique_count": 800,
            "entropy": 3.5,
        },
        generation={
            "kept_rules": 500,
            "rounds": 10,
            "worker_errors": 0,
        },
        top_rules=[
            {"rule": "l$1", "score": 1.5, "origin": "markov", "round": 1},
            {"rule": "u$!", "score": 1.2, "origin": "mutate", "round": 2},
        ],
    )


@pytest.fixture
def reporter(sample_report):
    return Reporter(sample_report)


class TestReporter:
    def test_to_json(self, reporter):
        j = reporter.to_json()
        data = json.loads(j)
        assert data["title"] == "Test Report"
        assert len(data["top_rules"]) == 2

    def test_write_json(self, reporter, tmp_path):
        path = tmp_path / "report.json"
        reporter.write_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["title"] == "Test Report"

    def test_to_csv(self, reporter):
        csv = reporter.to_csv()
        lines = csv.strip().split("\n")
        assert lines[0] == "rule,score,origin"
        assert "l$1" in csv

    def test_write_csv(self, reporter, tmp_path):
        path = tmp_path / "report.csv"
        reporter.write_csv(path)
        assert path.exists()

    def test_to_tsv(self, reporter):
        tsv = reporter.to_tsv()
        assert "rule\tscore\torigin" in tsv
        assert "l$1" in tsv

    def test_write_tsv(self, reporter, tmp_path):
        path = tmp_path / "report.tsv"
        reporter.write_tsv(path)
        assert path.exists()

    def test_to_markdown(self, reporter):
        md = reporter.to_markdown()
        assert "# Test Report" in md
        assert "l$1" in md

    def test_write_markdown(self, reporter, tmp_path):
        path = tmp_path / "report.md"
        reporter.write_markdown(path)
        assert path.exists()
        content = path.read_text()
        assert "Test Report" in content

    def test_to_html(self, reporter):
        html = reporter.to_html()
        assert "<!DOCTYPE html>" in html
        assert "l$1" in html

    def test_write_html(self, reporter, tmp_path):
        path = tmp_path / "report.html"
        reporter.write_html(path)
        assert path.exists()

    def test_write_all(self, reporter, tmp_path):
        reporter.write_all(tmp_path, stem="test_report")
        assert (tmp_path / "test_report.json").exists()
        assert (tmp_path / "test_report.csv").exists()
        assert (tmp_path / "test_report.tsv").exists()
        assert (tmp_path / "test_report.md").exists()
        assert (tmp_path / "test_report.html").exists()


class TestHtmlEscape:
    def test_escape_ampersand(self):
        assert _html_escape("a & b") == "a &amp; b"

    def test_escape_angle(self):
        assert _html_escape("<tag>") == "&lt;tag&gt;"

    def test_escape_quote(self):
        assert _html_escape('"hello"') == "&quot;hello&quot;"
