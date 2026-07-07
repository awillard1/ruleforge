"""Tests for ruleforge/grammar.py"""

import pytest
from ruleforge.grammar import (
    PCFGLearner, PCFG, GrammarRule,
    _classify_segment, _classify_password, TokType
)


class TestClassifySegment:
    def test_year(self):
        assert _classify_segment("2025") == TokType.YEAR
        assert _classify_segment("1999") == TokType.YEAR

    def test_number(self):
        assert _classify_segment("123") == TokType.NUMBER

    def test_symbol(self):
        assert _classify_segment("!") == TokType.SYMBOL

    def test_name(self):
        assert _classify_segment("Football") == TokType.NAME

    def test_lower(self):
        assert _classify_segment("password") == TokType.LOWER

    def test_upper(self):
        assert _classify_segment("PASSWORD") == TokType.UPPER


class TestClassifyPassword:
    def test_word_year(self):
        result = _classify_password("Football2025")
        assert TokType.NAME in result or TokType.WORD in result
        assert TokType.YEAR in result

    def test_simple_word(self):
        result = _classify_password("password")
        assert result  # not empty


class TestPCFGLearner:
    @pytest.fixture
    def learner(self):
        return PCFGLearner()

    def test_learn_builds_structure(self, learner):
        passwords = ["Football2025!", "password1", "Admin2023", "Test!"]
        learner.learn(passwords)
        assert learner._total_passwords > 0

    def test_top_structures(self, learner):
        passwords = ["Football2025!"] * 5 + ["password1"] * 3 + ["Test!"] * 2
        learner.learn(passwords)
        top = learner.top_structures(3)
        assert len(top) <= 3
        # Sorted by count descending
        counts = [c for _, c in top]
        assert counts == sorted(counts, reverse=True)

    def test_build_pcfg(self, learner):
        learner.learn(["Football2025!", "Admin2023", "Pass1!"])
        pcfg = learner.build()
        assert isinstance(pcfg, PCFG)
        assert len(pcfg._rules) > 0

    def test_learn_from_file(self, learner, tmp_path):
        f = tmp_path / "passwords.txt"
        f.write_text("Football2025!\npassword\nAdmin2023\n", encoding="utf-8")
        learner.learn_from_file(f)
        assert learner._total_passwords == 3


class TestPCFG:
    @pytest.fixture
    def pcfg(self):
        learner = PCFGLearner()
        learner.learn(["Football2025!", "Admin2023!", "Test1!"] * 3
                      + ["password", "hello123"] * 2)
        return learner.build()

    def test_top_n(self, pcfg):
        rules = pcfg.top_n(n=5)
        assert len(rules) <= 5

    def test_probabilities_sum_to_one(self, pcfg):
        rules = pcfg.top_n(n=50)
        total = sum(r.prob for r in rules)
        assert total <= 1.0 + 1e-6

    def test_sample_structure(self, pcfg):
        import random
        rng = random.Random(42)
        struct = pcfg.sample_structure(rng=rng)
        # May be None if no rules
        assert struct is None or isinstance(struct, tuple)

    def test_save_load(self, pcfg, tmp_path):
        path = tmp_path / "pcfg.json"
        pcfg.save(path)
        assert path.exists()
        pcfg2 = PCFG.load(path)
        assert len(pcfg2._rules) == len(pcfg._rules)
