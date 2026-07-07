"""Tests for ruleforge/semantics.py"""

import pytest
from ruleforge.semantics import SemanticAnalyzer, Category, _classify_heuristic


class TestHeuristicClassifier:
    def test_year(self):
        assert _classify_heuristic("2025") == Category.YEAR

    def test_number(self):
        assert _classify_heuristic("123456") == Category.NUMBER

    def test_keyboard(self):
        assert _classify_heuristic("qwerty") == Category.KEYBOARD

    def test_name(self):
        assert _classify_heuristic("Football") == Category.NAME

    def test_word(self):
        assert _classify_heuristic("password") == Category.WORD


class TestSemanticAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer(use_wordnet=False)

    def test_classify_word(self, analyzer):
        result = analyzer.classify("password")
        assert result.category in (Category.WORD, Category.NAME, Category.OTHER)
        assert result.source in ("heuristic", "word_list", "wordnet", "plugin")

    def test_classify_year(self, analyzer):
        result = analyzer.classify("2025")
        assert result.category == Category.YEAR

    def test_classify_number(self, analyzer):
        result = analyzer.classify("1234")
        assert result.category == Category.NUMBER

    def test_classify_many(self, analyzer):
        results = analyzer.classify_many(["password", "2025", "Football"])
        assert len(results) == 3

    def test_word_list_match(self, analyzer, tmp_path):
        wl = tmp_path / "cities.txt"
        wl.write_text("london\nparis\ntokyo\n", encoding="utf-8")
        analyzer.load_word_list(Category.CITY, wl)
        r = analyzer.classify("London")
        assert r.category == Category.CITY
        assert r.source == "word_list"

    def test_plugin(self, analyzer):
        def my_plugin(word: str):
            if word.lower() == "minecraft":
                return Category.GAME
            return None

        analyzer.register_plugin("game_detector", my_plugin)
        r = analyzer.classify("Minecraft")
        assert r.category == Category.GAME
        assert r.source == "game_detector"

    def test_plugin_fallthrough(self, analyzer):
        def noop(word: str):
            return None

        analyzer.register_plugin("noop", noop)
        r = analyzer.classify("qwerty")
        assert r.category == Category.KEYBOARD

    def test_analyze_passwords(self, analyzer):
        categories = analyzer.analyze_passwords([
            "Football2025!", "qwerty123", "Admin"
        ])
        assert isinstance(categories, dict)

    def test_cache(self, analyzer):
        analyzer.classify("password")
        # Second call uses cache
        r = analyzer.classify("password")
        assert r.word == "password"

    def test_stats(self, analyzer):
        analyzer.classify_many(["password", "2025"])
        s = analyzer.stats()
        assert s["total_classified"] == 2

    def test_export_json(self, analyzer, tmp_path):
        analyzer.classify_many(["password", "2025"])
        path = tmp_path / "semantics.json"
        analyzer.export_json(path)
        assert path.exists()
