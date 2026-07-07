"""Tests for ruleforge/plugins.py"""

import pytest
from pathlib import Path
from ruleforge.plugins import PluginRegistry, PluginType, PluginEntry


class SimpleScorer:
    """Minimal scorer plugin for testing."""
    name = "test_scorer"

    def score(self, rule: str) -> float:
        return float(len(rule))


class SimpleGenerator:
    name = "test_generator"

    def generate(self, n: int) -> list:
        return ["l$1"] * n


class TestPluginRegistry:
    @pytest.fixture
    def registry(self):
        return PluginRegistry()

    def test_register_and_get(self, registry):
        scorer = SimpleScorer()
        registry.register(PluginType.SCORER, scorer, name="test_scorer")
        retrieved = registry.get(PluginType.SCORER, "test_scorer")
        assert retrieved is scorer

    def test_register_invalid_type(self, registry):
        with pytest.raises(ValueError):
            registry.register("nonexistent_type", object())

    def test_list_plugins_empty(self, registry):
        assert registry.list_plugins() == []

    def test_list_plugins_filtered(self, registry):
        registry.register(PluginType.SCORER, SimpleScorer())
        registry.register(PluginType.GENERATOR, SimpleGenerator())
        scorers = registry.list_plugins(PluginType.SCORER)
        assert len(scorers) == 1
        assert scorers[0].plugin_type == PluginType.SCORER

    def test_get_all(self, registry):
        registry.register(PluginType.SCORER, SimpleScorer(), name="s1")
        registry.register(PluginType.SCORER, SimpleScorer(), name="s2")
        all_scorers = registry.get_all(PluginType.SCORER)
        assert len(all_scorers) == 2

    def test_get_missing_returns_none(self, registry):
        assert registry.get(PluginType.SCORER, "nonexistent") is None

    def test_stats(self, registry):
        registry.register(PluginType.SCORER, SimpleScorer(), name="s1")
        s = registry.stats()
        assert "scorer" in s
        assert "s1" in s["scorer"]

    def test_load_from_file(self, registry, tmp_path):
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(
            """
class MyScorer:
    name = "my_scorer"
    def score(self, rule):
        return 1.0

def register(registry):
    from ruleforge.plugins import PluginType
    registry.register(PluginType.SCORER, MyScorer(), name="my_scorer")
""",
            encoding="utf-8",
        )
        entries = registry.load_from_file(plugin_file)
        assert len(entries) == 1
        assert registry.get(PluginType.SCORER, "my_scorer") is not None

    def test_load_from_file_missing_register(self, registry, tmp_path):
        plugin_file = tmp_path / "bad_plugin.py"
        plugin_file.write_text("x = 1", encoding="utf-8")
        with pytest.raises(AttributeError):
            registry.load_from_file(plugin_file)

    def test_load_from_directory(self, registry, tmp_path):
        for i in range(3):
            pf = tmp_path / f"plugin_{i}.py"
            pf.write_text(
                f"""
def register(registry):
    from ruleforge.plugins import PluginType
    class Gen:
        name = "gen_{i}"
        def generate(self, n): return []
    registry.register(PluginType.GENERATOR, Gen(), name="gen_{i}")
""",
                encoding="utf-8",
            )
        entries = registry.load_from_directory(tmp_path)
        assert len(entries) == 3
