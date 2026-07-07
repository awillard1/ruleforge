"""Tests for ruleforge/analyzer.py"""

import json
import pytest
from pathlib import Path
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer, AnalysisResult


@pytest.fixture
def tmp_rule_file(tmp_path):
    f = tmp_path / "test.rule"
    f.write_text("l$1\nl$!\nlu\n# comment\n\nXXXINVALID\n", encoding="utf-8")
    return f


@pytest.fixture
def analyzer():
    return Analyzer(Parser())


class TestAnalyzer:
    def test_ingest_file(self, analyzer, tmp_rule_file):
        analyzer.ingest_file(tmp_rule_file)
        r = analyzer.result()
        assert r.valid_lines >= 3
        assert r.invalid_lines >= 1
        assert r.comment_or_empty >= 2
        assert r.unique_count >= 3

    def test_ingest_rules_in_memory(self, analyzer):
        analyzer.ingest_rules(["l", "u", "l$1", "l$1"])  # l$1 duplicated
        r = analyzer.result()
        assert r.unique_count == 3
        assert r.duplicate_count >= 1

    def test_cmd_freq_populated(self, analyzer):
        analyzer.ingest_rules(["l", "u", "l$1"])
        assert analyzer.cmd["l"] >= 2
        assert analyzer.cmd["u"] >= 1

    def test_start_freq(self, analyzer):
        analyzer.ingest_rules(["l$1", "l$2", "u"])
        assert analyzer.start["l"] >= 2

    def test_transition_populated(self, analyzer):
        analyzer.ingest_rules(["lu"])
        assert analyzer.trans["l"]["u"] >= 1

    def test_len_dist(self, analyzer):
        analyzer.ingest_rules(["l", "lu", "lur"])
        assert analyzer.len_dist[1] >= 1
        assert analyzer.len_dist[2] >= 1
        assert analyzer.len_dist[3] >= 1

    def test_entropy_positive(self, analyzer):
        analyzer.ingest_rules(["l", "u", "c", "r"])
        r = analyzer.result()
        assert r.entropy > 0

    def test_export_json(self, analyzer, tmp_path):
        analyzer.ingest_rules(["l$1", "u"])
        out = tmp_path / "analysis.json"
        analyzer.export_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "counts" in data
        assert "cmd_freq" in data


class TestAnalysisResult:
    def test_to_dict(self):
        r = AnalysisResult(total_lines=10, valid_lines=8)
        d = r.to_dict()
        assert d["counts"]["total_lines"] == 10
        assert d["counts"]["valid_lines"] == 8

    def test_to_json(self):
        r = AnalysisResult(entropy=1.5)
        j = r.to_json()
        data = json.loads(j)
        assert data["entropy"] == 1.5
