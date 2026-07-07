"""Tests for ruleforge/generator.py"""

import random
import pytest
from collections import Counter
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator, MixtureWeights


RULES = ["l$1", "l$!", "lu", "uc", "cl", "l", "u", "c", "r", "d",
         "l$2", "u^!", "c$0", "lu$1", "l$1u"]


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def analyzer(parser):
    a = Analyzer(parser)
    a.ingest_rules(RULES)
    return a


@pytest.fixture
def generator(parser, analyzer):
    rng = random.Random(42)
    return RuleGenerator(
        parser=parser,
        source_rules=list(analyzer.unique_rules),
        cmd_freq=analyzer.cmd,
        start_freq=analyzer.start,
        end_freq=analyzer.end,
        trans=analyzer.trans,
        param_freq=analyzer.params,
        len_dist=analyzer.len_dist,
        allow_param_fallback=True,
        rng=rng,
    )


class TestMixtureWeights:
    def test_normalize(self):
        w = MixtureWeights(markov=2.0, mutate=2.0)
        n = w.normalize()
        total = n.markov + n.ngram + n.template + n.random_explore + n.mutate
        assert abs(total - 1.0) < 1e-9

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            MixtureWeights(0, 0, 0, 0, 0).normalize()


class TestRuleGenerator:
    def test_generate_one_valid(self, generator, parser):
        for _ in range(20):
            rule = generator.generate_one(max_ops=10)
            if rule:
                assert parser.validate(rule)

    def test_generate_batch(self, generator, parser):
        batch = generator.generate_batch(batch_size=20, max_ops=10)
        assert len(batch) > 0
        for rule in batch:
            assert parser.validate(rule)

    def test_batch_unique(self, generator):
        batch = generator.generate_batch(batch_size=50)
        assert len(batch) == len(set(batch))

    def test_mutate_returns_valid_or_empty(self, generator, parser):
        for rule in RULES:
            mutated = generator.mutate(rule, max_ops=10)
            if mutated:
                assert parser.validate(mutated)

    def test_offline_score_finite(self, generator):
        score = generator.offline_score("l$1")
        assert isinstance(score, float)
        assert score != float("-inf")

    def test_offline_score_invalid(self, generator):
        score = generator.offline_score("XXXXX")
        assert score == float("-inf")

    def test_build_markov(self, generator, parser):
        for _ in range(20):
            rule = generator._build_markov(max_ops=8)
            if rule:
                assert parser.validate(rule, max_ops=8)
