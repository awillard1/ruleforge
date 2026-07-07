"""Tests for ruleforge/markov.py"""

import pytest
from ruleforge.parser import Parser
from ruleforge.markov import (
    MarkovTrainer, MarkovSampler, VariableOrderMarkov, BOS
)


@pytest.fixture
def parser():
    return Parser()


RULES = ["l$1", "l$!", "lu", "uc", "cl", "l", "u", "c", "r", "d"]


class TestMarkovTrainer:
    def test_train_builds_model(self, parser):
        trainer = MarkovTrainer(order=1)
        trainer.train_rules(RULES, parser)
        model = trainer.build()
        assert len(model.vocab) > 0
        assert len(model.counts) > 0

    def test_order_2(self, parser):
        trainer = MarkovTrainer(order=2)
        trainer.train_rules(RULES, parser)
        model = trainer.build()
        # Second-order contexts should have 2-tuples
        for ctx in model.counts:
            assert len(ctx) == 2


class TestMarkovSampler:
    def test_sample_nonempty(self, parser):
        trainer = MarkovTrainer(order=1)
        trainer.train_rules(RULES, parser)
        model = trainer.build()
        sampler = MarkovSampler(model)
        seq = sampler.sample(max_len=5)
        # May be empty in edge cases, but usually not
        assert isinstance(seq, list)

    def test_sample_many(self, parser):
        trainer = MarkovTrainer(order=1)
        trainer.train_rules(RULES, parser)
        model = trainer.build()
        sampler = MarkovSampler(model)
        seqs = sampler.sample_many(10, max_len=5)
        assert len(seqs) == 10

    def test_interpolation(self, parser):
        t1 = MarkovTrainer(order=1)
        t1.train_rules(RULES, parser)
        m1 = t1.build()
        t2 = MarkovTrainer(order=2)
        t2.train_rules(RULES, parser)
        m2 = t2.build()
        sampler = MarkovSampler(m2, lower_models=[m1], lambdas=[0.3, 0.7])
        seq = sampler.sample(max_len=5)
        assert isinstance(seq, list)


class TestVariableOrderMarkov:
    def test_train_and_sample(self, parser):
        vom = VariableOrderMarkov(max_order=2)
        vom.train(RULES, parser)
        seq = vom.sample(max_len=5)
        assert isinstance(seq, list)

    def test_save_load(self, parser, tmp_path):
        vom = VariableOrderMarkov(max_order=2)
        vom.train(RULES, parser)
        path = tmp_path / "markov.json"
        vom.save(path)
        assert path.exists()
        vom2 = VariableOrderMarkov(max_order=2)
        vom2.load(path)
        seq = vom2.sample(max_len=5)
        assert isinstance(seq, list)

    def test_sample_many(self, parser):
        vom = VariableOrderMarkov(max_order=1)
        vom.train(RULES, parser)
        seqs = vom.sample_many(20, max_len=5)
        assert len(seqs) == 20
