"""Tests for ruleforge/runtime.py"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ruleforge.runtime import WordSampler, RuntimeEvaluator


class TestWordSampler:
    def test_load_words_basic(self, tmp_path):
        import random
        wf = tmp_path / "words.txt"
        wf.write_text("password\nletmein\nadmin\n123456\n", encoding="utf-8")
        rng = random.Random(0)
        words, stats = WordSampler.load_words(wf, sample_size=10, rng=rng)
        assert len(words) > 0
        assert stats["raw_count"] == 4

    def test_load_words_dedupe(self, tmp_path):
        import random
        wf = tmp_path / "words.txt"
        wf.write_text("password\npassword\nletmein\n", encoding="utf-8")
        rng = random.Random(0)
        words, stats = WordSampler.load_words(
            wf, sample_size=10, rng=rng, dedupe_exact=True
        )
        assert len(words) == 2

    def test_load_words_not_stratified(self, tmp_path):
        import random
        wf = tmp_path / "words.txt"
        lines = "\n".join(f"word{i}" for i in range(20))
        wf.write_text(lines, encoding="utf-8")
        rng = random.Random(0)
        words, stats = WordSampler.load_words(
            wf, sample_size=5, rng=rng, stratified=False
        )
        assert len(words) <= 5
        assert stats["stratified"] is False

    def test_shape_signature(self):
        sig = WordSampler.shape_signature("Password1!")
        assert isinstance(sig, str)
        assert len(sig) == len("Password1!")

    def test_len_bucket(self):
        b = WordSampler.len_bucket(5)
        assert isinstance(b, str)

    def test_alpha_stem(self):
        stem = WordSampler.alpha_stem("P4ssw0rd!")
        assert stem.isalpha() or stem == ""

    def test_signature(self):
        sig = WordSampler.signature("P4ss!")
        assert isinstance(sig, str)

    def test_empty_file(self, tmp_path):
        import random
        wf = tmp_path / "empty.txt"
        wf.write_text("", encoding="utf-8")
        rng = random.Random(0)
        words, stats = WordSampler.load_words(wf, sample_size=10, rng=rng)
        assert words == []
        assert stats["raw_count"] == 0


class TestRuntimeEvaluator:
    @pytest.fixture
    def evaluator(self):
        return RuntimeEvaluator(hashcat_bin="hashcat", timeout_sec=5)

    def test_constructor(self, evaluator):
        assert evaluator._bin == "hashcat"

    def test_stats_empty(self, evaluator):
        s = evaluator.stats()
        assert s["calls"] == 0
        assert s["cache_size"] == 0

    def test_clear_cache(self, evaluator):
        evaluator._cache["l$1"] = {"password1"}
        evaluator.clear_cache()
        assert evaluator._cache == {}

    def test_outputs_for_rule_binary_not_found(self, evaluator):
        ok, outputs, err = evaluator.outputs_for_rule("l$1", ["password"])
        # hashcat not installed → should fail gracefully
        assert not ok or isinstance(outputs, set)
        assert isinstance(err, str)

    def test_outputs_for_rule_cached(self, evaluator):
        # Pre-populate cache
        evaluator._cache["l$1"] = {"password1", "letmein1"}
        ok, outputs, err = evaluator.outputs_for_rule("l$1", ["password"])
        assert ok
        assert outputs == {"password1", "letmein1"}
        assert err == ""

    def test_novelty(self, evaluator):
        evaluator._cache["l$1"] = {"password1", "letmein1"}
        baseline = {"password1"}
        n = evaluator.novelty("l$1", ["password"], baseline)
        assert n == 1

    def test_novelty_fail(self, evaluator):
        # When outputs_for_rule fails, novelty returns 0
        evaluator._bin = "no_such_binary_xyz"
        n = evaluator.novelty("l$1", ["password"], set())
        assert n == 0
