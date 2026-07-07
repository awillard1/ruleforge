"""Tests for ruleforge/bayesian.py"""

import random
import pytest
from ruleforge.bayesian import BayesianOptimizer, BayesianConfig
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator


SOURCE_RULES = ["l$1", "lu", "u", "c", "r", "d", "l", "u^!", "c$0"]


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def generator(parser):
    analyzer = Analyzer(parser)
    analyzer.ingest_rules(SOURCE_RULES)
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
        rng=random.Random(0),
    )


@pytest.fixture
def fitness_fn():
    def fn(rule: str) -> float:
        return min(1.0, len(rule) * 0.1)
    return fn


@pytest.fixture
def config():
    return BayesianConfig(
        n_initial=3,
        n_iterations=2,
        kappa=2.0,
        batch_size_per_iter=3,
    )


class TestBayesianConfig:
    def test_defaults(self):
        cfg = BayesianConfig()
        assert cfg.n_initial > 0
        assert cfg.n_iterations > 0
        assert cfg.kappa > 0

    def test_custom(self, config):
        assert config.n_initial == 3
        assert config.n_iterations == 2
        assert config.kappa == pytest.approx(2.0)


class TestBayesianOptimizer:
    def test_init(self, config, parser, generator, fitness_fn):
        opt = BayesianOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=random.Random(0),
        )
        assert opt is not None

    def test_run_returns_list(self, config, parser, generator, fitness_fn):
        opt = BayesianOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=random.Random(0),
        )
        results = opt.run()
        assert isinstance(results, list)

    def test_best_rules(self, config, parser, generator, fitness_fn):
        opt = BayesianOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=random.Random(0),
        )
        opt.run()
        best = opt.best_rules(n=3)
        assert isinstance(best, list)
        assert len(best) <= 3

    def test_save_load(self, config, parser, generator, fitness_fn, tmp_path):
        opt = BayesianOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=random.Random(0),
        )
        opt.run()
        path = tmp_path / "bayesian.json"
        opt.save(path)
        assert path.exists()
        opt2 = BayesianOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=random.Random(0),
        )
        opt2.load(path)  # should not raise
