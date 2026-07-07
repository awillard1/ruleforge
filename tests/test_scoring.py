"""Tests for ruleforge/scoring.py"""

import pytest
from collections import Counter
from ruleforge.parser import Parser
from ruleforge.scoring import Scorer, ScoreWeights, ScoreBreakdown


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def scorer(parser):
    cmd_freq = Counter({"l": 10, "u": 8, "c": 5, "$": 7, "r": 3})
    param_freq = {"$": Counter({"1": 5, "!": 3, "2": 2})}
    trans = {"l": Counter({"u": 4, "$": 3}), "u": Counter({"c": 2})}
    start_freq = Counter({"l": 8, "u": 5})
    return Scorer(
        parser=parser,
        cmd_freq=cmd_freq,
        param_freq=param_freq,
        trans=trans,
        start_freq=start_freq,
    )


class TestScorer:
    def test_score_valid_rule(self, scorer):
        score = scorer.score("l$1")
        assert isinstance(score, float)

    def test_score_invalid_rule(self, scorer):
        score = scorer.score("XXXXX")
        assert score == float("-inf")

    def test_score_detailed(self, scorer):
        breakdown = scorer.score_detailed("l$1")
        assert isinstance(breakdown, ScoreBreakdown)
        assert "novelty" in breakdown.components
        assert "entropy" in breakdown.components
        assert "template_rarity" in breakdown.components

    def test_rank(self, scorer):
        rules = ["l$1", "u", "l", "c$!"]
        ranked = scorer.rank(rules)
        assert len(ranked) == 4
        # Should be sorted descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_known_rule_penalty(self, parser):
        cmd_freq = Counter({"l": 5})
        scorer = Scorer(
            parser=parser,
            cmd_freq=cmd_freq,
            param_freq={},
            trans={},
            start_freq=Counter({"l": 5}),
            known_rules={"l$1"},
        )
        breakdown = scorer.score_detailed("l$1")
        assert breakdown.components.get("uniqueness", 0) < 0

    def test_score_weights(self, parser):
        w = ScoreWeights(novelty=0.0, template_rarity=0.0)
        cmd_freq = Counter({"l": 5})
        scorer = Scorer(
            parser=parser,
            cmd_freq=cmd_freq,
            param_freq={},
            trans={},
            start_freq=Counter({"l": 5}),
            weights=w,
        )
        score = scorer.score("l")
        assert isinstance(score, float)
