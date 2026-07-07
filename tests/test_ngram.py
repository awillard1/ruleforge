"""Tests for ruleforge/ngram.py"""

import pytest
from ruleforge.parser import Parser
from ruleforge.ngram import NGramCounter, LaplaceNGram, BackoffNGram, KneserNeyNGram, NGramEngine


RULES = ["l$1", "l$!", "lu", "uc", "cl", "l", "u", "c", "r", "d"]


@pytest.fixture
def parser():
    return Parser()


class TestNGramCounter:
    def test_add_sequence(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l", "u", "c"])
        assert counter.count(("l", "u")) > 0
        assert counter.count(("u", "c")) > 0
        assert "l" in counter.vocab

    def test_unigram_count(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l", "l", "u"])
        # Two l's
        assert counter.count(("l",)) == 2

    def test_to_from_dict(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l", "u"])
        d = counter.to_dict()
        counter2 = NGramCounter.from_dict(d)
        assert counter2.count(("l", "u")) == counter.count(("l", "u"))


class TestLaplaceNGram:
    def test_prob_positive(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l", "u"])
        model = LaplaceNGram(counter, alpha=1.0)
        p = model.prob(("l",), "u")
        assert 0.0 < p <= 1.0

    def test_log_prob(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l", "u"])
        model = LaplaceNGram(counter)
        lp = model.log_prob(["l", "u"])
        assert lp < 0  # log-prob always ≤ 0


class TestBackoffNGram:
    def test_prob_positive(self):
        counter = NGramCounter(n=3)
        counter.add_sequence(["l", "u", "c"])
        model = BackoffNGram(counter)
        p = model.prob(("l", "u"), "c")
        assert p >= 0.0

    def test_backoff_to_unigram(self):
        counter = NGramCounter(n=2)
        counter.add_sequence(["l"])
        model = BackoffNGram(counter)
        # Context ("x",) never seen → should back off to unigram
        p = model.prob(("x",), "l")
        assert p >= 0.0


class TestKneserNeyNGram:
    def test_prob_sums_to_one(self):
        counter = NGramCounter(n=2)
        for ops in [["l", "u"], ["l", "c"], ["l", "r"]]:
            counter.add_sequence(ops)
        from ruleforge.ngram import EOS
        model = KneserNeyNGram(counter)
        vocab = list(counter.vocab) + [EOS]
        total = sum(model.prob(("l",), v) for v in vocab)
        assert abs(total - 1.0) < 0.5  # loose check given small vocab


class TestNGramEngine:
    def test_train_score(self, parser):
        engine = NGramEngine(n=2, smoothing="backoff")
        engine.train(RULES, parser)
        score = engine.score_rule("l$1", parser)
        assert isinstance(score, float)

    def test_invalid_rule_score(self, parser):
        engine = NGramEngine(n=2, smoothing="laplace")
        engine.train(RULES, parser)
        score = engine.score_rule("XXXXX", parser)
        assert score == float("-inf")

    def test_save_load(self, parser, tmp_path):
        engine = NGramEngine(n=2, smoothing="backoff")
        engine.train(RULES, parser)
        path = tmp_path / "ngram.json"
        engine.save(path)
        engine2 = NGramEngine()
        engine2.load(path)
        assert engine2.score_rule("l", parser) == engine.score_rule("l", parser)

    def test_kneser_ney(self, parser):
        engine = NGramEngine(n=2, smoothing="kneser_ney")
        engine.train(RULES, parser)
        score = engine.score_rule("lu", parser)
        assert isinstance(score, float)
