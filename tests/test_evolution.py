"""Tests for ruleforge/evolution.py"""

import random
import pytest
from unittest.mock import MagicMock

from ruleforge.evolution import (
    GeneticOptimizer,
    EvolutionConfig,
    Individual,
)
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator


SOURCE_RULES = [
    "l$1", "l$!", "l$@", "lu", "uc", "cl", "l", "u", "c", "r", "d",
    "l$2", "u^!", "c$0",
]


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def generator(parser, rng):
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
        rng=rng,
    )


@pytest.fixture
def fitness_fn(parser):
    def fn(rule: str) -> float:
        return min(1.0, len(rule) * 0.1)
    return fn


@pytest.fixture
def config():
    return EvolutionConfig(
        population_size=20,
        max_generations=5,
        elite_fraction=0.1,
        tournament_size=3,
        crossover_prob=0.7,
        mutation_prob=0.3,
        adaptive_mutation=True,
        stagnation_limit=3,
        checkpoint_interval=2,
    )


class TestIndividual:
    def test_create(self):
        ind = Individual(rule="l$1", fitness=0.5)
        assert ind.rule == "l$1"
        assert ind.fitness == pytest.approx(0.5)

    def test_to_dict(self):
        ind = Individual(rule="l$1", fitness=0.5, generation=1)
        d = ind.to_dict()
        assert d["rule"] == "l$1"
        assert d["fitness"] == pytest.approx(0.5)

    def test_from_dict(self):
        ind = Individual.from_dict({"rule": "lu", "fitness": 0.7, "generation": 2})
        assert ind.rule == "lu"
        assert ind.fitness == pytest.approx(0.7)


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.population_size > 0
        assert 0 < cfg.elite_fraction < 1
        assert 0 < cfg.crossover_prob <= 1
        assert 0 < cfg.mutation_prob <= 1

    def test_custom(self, config):
        assert config.population_size == 20
        assert config.max_generations == 5


class TestGeneticOptimizer:
    def test_init(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        assert opt is not None

    def test_run_returns_list(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        results = opt.run()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_individuals_have_fitness(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        pop = opt.run()
        for ind in pop:
            assert isinstance(ind, Individual)
            assert ind.fitness >= 0

    def test_top_rules(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        opt.run()
        top = opt.top_rules(n=5)
        assert isinstance(top, list)
        assert len(top) <= 5

    def test_initialize(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        opt.initialize(seed_rules=SOURCE_RULES[:5])
        assert len(opt._population) > 0

    def test_stats(self, config, parser, generator, fitness_fn, rng):
        opt = GeneticOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        opt.run()
        s = opt.stats()
        assert isinstance(s, dict)

