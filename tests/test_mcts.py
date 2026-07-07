"""Tests for ruleforge/mcts.py"""

import random
import pytest
from ruleforge.mcts import MCTSNode, MCTSConfig, MCTSOptimizer
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator


SOURCE_RULES = ["l$1", "lu", "u", "c", "r", "d", "l", "u^!", "c$0"]


@pytest.fixture
def rng():
    return random.Random(7)


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
def fitness_fn():
    def fn(rule: str) -> float:
        return min(1.0, len(rule) * 0.08)
    return fn


@pytest.fixture
def config():
    return MCTSConfig(
        n_simulations=10,
        max_depth=4,
        exploration_constant=1.414,
        rollout_depth=3,
    )


class TestMCTSNode:
    def test_create(self):
        node = MCTSNode(rule="", parent=None)
        assert node.visits == 0
        assert node.value == pytest.approx(0.0)

    def test_update(self):
        node = MCTSNode(rule="l", parent=None)
        node.update(0.8)
        assert node.visits == 1
        assert node.value == pytest.approx(0.8)

    def test_is_fully_expanded_untried_none(self):
        node = MCTSNode(rule="", parent=None)
        # _untried is None → not fully expanded
        assert not node.is_fully_expanded(["l", "u"])

    def test_is_fully_expanded_empty(self):
        node = MCTSNode(rule="", parent=None)
        node._untried = []
        assert node.is_fully_expanded(["l", "u"])

    def test_best_child(self):
        parent = MCTSNode(rule="", parent=None)
        parent.visits = 5
        child = MCTSNode(rule="l", parent=parent)
        child.visits = 3
        child.value = 2.4
        parent.children.append(child)
        best = parent.best_child(c=1.414)
        assert best is child


class TestMCTSConfig:
    def test_defaults(self):
        cfg = MCTSConfig()
        assert cfg.n_simulations > 0
        assert cfg.max_depth > 0

    def test_custom(self, config):
        assert config.n_simulations == 10
        assert config.max_depth == 4


class TestMCTSOptimizer:
    def test_init(self, config, parser, generator, fitness_fn, rng):
        opt = MCTSOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        assert opt is not None

    def test_search_returns_list(self, config, parser, generator, fitness_fn, rng):
        opt = MCTSOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        results = opt.search()
        assert isinstance(results, list)

    def test_top_rules(self, config, parser, generator, fitness_fn, rng):
        opt = MCTSOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        opt.search()
        top = opt.top_rules(n=5)
        assert isinstance(top, list)
        assert len(top) <= 5

    def test_stats(self, config, parser, generator, fitness_fn, rng):
        opt = MCTSOptimizer(
            parser=parser,
            generator=generator,
            fitness_fn=fitness_fn,
            config=config,
            rng=rng,
        )
        opt.search()
        s = opt.stats()
        assert isinstance(s, dict)
