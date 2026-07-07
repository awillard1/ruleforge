"""Tests for ruleforge/reinforcement.py"""

import random
import pytest
from ruleforge.reinforcement import RuleEnvironment, QTableConfig, QTableAgent
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator


SOURCE_RULES = ["l$1", "lu", "u", "c", "r", "d"]


@pytest.fixture
def rng():
    return random.Random(0)


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
def env(parser, generator):
    return RuleEnvironment(parser=parser, generator=generator, max_ops=6)


@pytest.fixture
def q_config():
    return QTableConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.5,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        max_episodes=10,
        max_steps=10,
    )


class TestRuleEnvironment:
    def test_reset(self, env):
        state = env.reset()
        assert isinstance(state, str)

    def test_step(self, env):
        env.reset()
        actions = env.action_space
        assert len(actions) > 0
        obs, reward, done = env.step(actions[0])
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_current_rule(self, env):
        env.reset()
        rule = env.current_rule
        assert isinstance(rule, str)

    def test_episode(self, env):
        env.reset()
        for _ in range(10):
            actions = env.action_space
            if not actions:
                break
            obs, reward, done = env.step(actions[0])
            if done:
                break


class TestQTableConfig:
    def test_defaults(self):
        cfg = QTableConfig()
        assert 0 < cfg.alpha <= 1
        assert 0 < cfg.gamma <= 1
        assert 0 <= cfg.epsilon <= 1

    def test_custom(self, q_config):
        assert q_config.alpha == pytest.approx(0.1)
        assert q_config.max_episodes == 10


class TestQTableAgent:
    def test_init(self, env, q_config, rng):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        assert agent is not None

    def test_train_no_error(self, env, q_config, rng):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        agent.train(seed_rules=SOURCE_RULES)  # should not raise

    def test_generate_rules(self, env, q_config, rng):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        agent.train(seed_rules=SOURCE_RULES)
        rules = agent.generate_rules(n=5)
        assert isinstance(rules, list)

    def test_epsilon_decay_after_train(self, env, q_config, rng):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        initial_eps = agent._epsilon
        agent.train(seed_rules=SOURCE_RULES)
        assert agent._epsilon <= initial_eps

    def test_stats(self, env, q_config, rng):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        agent.train(seed_rules=SOURCE_RULES)
        s = agent.stats()
        assert isinstance(s, dict)

    def test_save_load(self, env, q_config, rng, tmp_path):
        agent = QTableAgent(env=env, config=q_config, rng=rng)
        agent.train(seed_rules=SOURCE_RULES)
        path = tmp_path / "qtable.json"
        agent.save(path)
        assert path.exists()
        agent2 = QTableAgent(env=env, config=q_config, rng=rng)
        agent2.load(path)

