"""Tests for ruleforge/passwords.py"""

import pytest
from ruleforge.passwords import PasswordAnalyzer, PasswordStats


@pytest.fixture
def analyzer():
    return PasswordAnalyzer()


PASSWORDS = [
    "Football2025!",
    "password",
    "Password1",
    "abc123",
    "qwerty",
    "Admin2023",
    "janSummer!",
    "L33tH@x0r",
    "aaabbbccc",
]


class TestPasswordAnalyzer:
    def test_analyze_basic(self, analyzer):
        analyzer.analyze(PASSWORDS)
        r = analyzer.result()
        assert r.total == len(PASSWORDS)
        assert r.has_year > 0

    def test_length_dist(self, analyzer):
        analyzer.analyze(["hello", "world!"])
        r = analyzer.result()
        assert len(r.length_dist) > 0

    def test_has_keyboard_walk(self, analyzer):
        analyzer.analyze(["qwerty", "hello"])
        r = analyzer.result()
        assert r.has_keyboard_walk >= 1

    def test_has_leet(self, analyzer):
        analyzer.analyze(["L33tH@x0r", "normal"])
        r = analyzer.result()
        assert r.has_leet >= 1

    def test_has_repeat(self, analyzer):
        analyzer.analyze(["aaabbbccc", "hello"])
        r = analyzer.result()
        assert r.has_repeat >= 1

    def test_year_values(self, analyzer):
        analyzer.analyze(["Football2025!"])
        r = analyzer.result()
        assert "2025" in r.year_values

    def test_top_prefixes(self, analyzer):
        analyzer.analyze(["password", "pass123", "pass!"] * 3)
        r = analyzer.result()
        assert "pas" in r.top_prefixes

    def test_export_json(self, analyzer, tmp_path):
        analyzer.analyze(PASSWORDS)
        path = tmp_path / "passwords.json"
        analyzer.export_json(path)
        assert path.exists()

    def test_analyze_file(self, analyzer, tmp_path):
        f = tmp_path / "passwords.txt"
        f.write_text("\n".join(PASSWORDS), encoding="utf-8")
        analyzer.analyze_file(f)
        r = analyzer.result()
        assert r.total == len(PASSWORDS)

    def test_short_password_ignored(self, analyzer):
        a = PasswordAnalyzer(min_length=5)
        a.analyze(["ab"])  # too short
        r = a.result()
        assert r.total == 0


class TestPasswordStats:
    def test_to_dict(self):
        s = PasswordStats(total=10, has_year=3)
        d = s.to_dict()
        assert d["total"] == 10
        assert d["has_year"] == 3

    def test_to_json(self):
        import json
        s = PasswordStats(total=5)
        j = s.to_json()
        data = json.loads(j)
        assert data["total"] == 5
