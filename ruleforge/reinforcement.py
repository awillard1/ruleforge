"""
ruleforge/reinforcement.py
--------------------------
Reinforcement Learning engine for Hashcat rule optimization.

State:   Current rule (token sequence).
Actions: Insert, Delete, Replace, Swap, Modify parameter.
Reward:  New passwords generated, coverage, unique outputs, runtime efficiency.

Implements a simple Q-learning / policy gradient sketch that can be trained
offline against the heuristic scorer or online against hashcat --stdout.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .parser import Parser, Token, _arity, _ok_char, MAX_OPS
from .generator import RuleGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

ACTION_INSERT = "insert"
ACTION_DELETE = "delete"
ACTION_REPLACE_CMD = "replace_cmd"
ACTION_SWAP = "swap"
ACTION_MODIFY_PARAM = "modify_param"

ALL_ACTIONS = (ACTION_INSERT, ACTION_DELETE, ACTION_REPLACE_CMD, ACTION_SWAP, ACTION_MODIFY_PARAM)


# ---------------------------------------------------------------------------
# Reward function type
# ---------------------------------------------------------------------------

RewardFn = Callable[[str, str], float]  # (old_rule, new_rule) → reward


def _default_reward(generator: RuleGenerator, old: str, new: str) -> float:
    """Heuristic reward: improvement in offline score."""
    old_score = generator.offline_score(old)
    new_score = generator.offline_score(new)
    return new_score - old_score


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RuleEnvironment:
    """Single-agent environment wrapping a Hashcat rule.

    Observations are simple rule strings; actions are rule edits.

    Args:
        parser:    A :class:`~ruleforge.parser.Parser` instance.
        generator: A :class:`~ruleforge.generator.RuleGenerator` for
                   parameter sampling and mutation helpers.
        max_ops:   Maximum rule length.
        reward_fn: Callable ``(old_rule, new_rule) → float``.
    """

    def __init__(
        self,
        parser: Parser,
        generator: RuleGenerator,
        max_ops: int = MAX_OPS,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self._parser = parser
        self._gen = generator
        self._max_ops = max_ops
        self._reward_fn: RewardFn = reward_fn or (
            lambda old, new: _default_reward(generator, old, new)
        )
        self._current_rule: str = ":"
        self._step_count: int = 0

    def reset(self, initial_rule: str | None = None) -> str:
        """Reset environment to *initial_rule* (random if not provided)."""
        if initial_rule and self._parser.validate(initial_rule):
            self._current_rule = initial_rule
        else:
            cand = self._gen.generate_one(max_ops=min(6, self._max_ops))
            self._current_rule = cand if cand else ":"
        self._step_count = 0
        return self._current_rule

    def step(self, action: str) -> tuple[str, float, bool]:
        """Apply *action* to current rule.

        Returns:
            (new_rule, reward, done)
        """
        old_rule = self._current_rule
        new_rule = self._apply_action(action, old_rule)

        if not new_rule or not self._parser.validate(new_rule, max_ops=self._max_ops):
            new_rule = old_rule  # invalid action → no change

        reward = self._reward_fn(old_rule, new_rule)
        self._current_rule = new_rule
        self._step_count += 1

        done = self._step_count >= 20  # episode length
        return new_rule, reward, done

    def _apply_action(self, action: str, rule: str) -> str:
        return self._gen.mutate(rule, max_ops=self._max_ops) if action else rule

    @property
    def current_rule(self) -> str:
        return self._current_rule

    @property
    def action_space(self) -> tuple[str, ...]:
        return ALL_ACTIONS


# ---------------------------------------------------------------------------
# Q-Table agent (tabular ε-greedy Q-learning)
# ---------------------------------------------------------------------------


@dataclass
class QTableConfig:
    """Hyperparameters for the tabular Q-learning agent."""

    alpha: float = 0.1          # learning rate
    gamma: float = 0.95         # discount factor
    epsilon: float = 1.0        # initial exploration rate
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    max_episodes: int = 1000
    max_steps: int = 20


class QTableAgent:
    """Tabular Q-learning agent.

    The state representation is a compact *shape* string (operation
    sequence without parameters), which keeps the state space manageable.

    Args:
        env:    A :class:`RuleEnvironment` instance.
        config: :class:`QTableConfig`.
        rng:    Random source.
    """

    def __init__(
        self,
        env: RuleEnvironment,
        config: QTableConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._env = env
        self._cfg = config or QTableConfig()
        self._rng = rng or random.Random()
        # Q[state][action] → value
        self._Q: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._epsilon = self._cfg.epsilon
        self._episode_rewards: list[float] = []

    def _state(self, rule: str) -> str:
        """Map rule to state representation (operation shape)."""
        toks = self._env._parser.try_parse(rule)
        if not toks:
            return ""
        return "".join(t.cmd for t in toks)

    def _choose_action(self, state: str) -> str:
        """ε-greedy action selection."""
        if self._rng.random() < self._epsilon:
            return self._rng.choice(list(self._env.action_space))
        q_vals = self._Q[state]
        if not q_vals:
            return self._rng.choice(list(self._env.action_space))
        return max(q_vals, key=q_vals.__getitem__)

    def train(self, seed_rules: list[str] | None = None) -> None:
        """Run Q-learning for max_episodes episodes."""
        cfg = self._cfg
        for ep in range(cfg.max_episodes):
            start = (seed_rules[self._rng.randint(0, len(seed_rules) - 1)]
                     if seed_rules else None)
            rule = self._env.reset(start)
            state = self._state(rule)
            total_reward = 0.0

            for _ in range(cfg.max_steps):
                action = self._choose_action(state)
                new_rule, reward, done = self._env.step(action)
                new_state = self._state(new_rule)

                # Q-update
                old_q = self._Q[state][action]
                next_max = max(self._Q[new_state].values(), default=0.0)
                self._Q[state][action] = old_q + cfg.alpha * (
                    reward + cfg.gamma * next_max - old_q
                )

                state = new_state
                total_reward += reward
                if done:
                    break

            self._episode_rewards.append(total_reward)

            # Decay epsilon
            self._epsilon = max(cfg.epsilon_min, self._epsilon * cfg.epsilon_decay)

            if ep % 100 == 0:
                logger.debug(
                    "Episode %d: total_reward=%.4f epsilon=%.3f",
                    ep, total_reward, self._epsilon,
                )

        logger.info(
            "Q-learning complete: %d episodes, final ε=%.4f",
            cfg.max_episodes, self._epsilon,
        )

    def generate_rules(self, n: int, max_steps: int = 20) -> list[str]:
        """Generate *n* rules by following the learned greedy policy."""
        rules: list[str] = []
        for _ in range(n):
            rule = self._env.reset()
            state = self._state(rule)
            for _ in range(max_steps):
                action = max(
                    self._env.action_space,
                    key=lambda a: self._Q[state].get(a, 0.0),
                )
                rule, _, done = self._env.step(action)
                state = self._state(rule)
                if done:
                    break
            if self._env._parser.validate(rule):
                rules.append(rule)
        return rules

    def save(self, path: Path) -> None:
        """Persist Q-table to JSON."""
        data = {
            "epsilon": self._epsilon,
            "episode_rewards": self._episode_rewards,
            "q_table": {
                state: dict(actions)
                for state, actions in self._Q.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Restore Q-table from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._epsilon = float(data.get("epsilon", self._cfg.epsilon_min))
        self._episode_rewards = list(data.get("episode_rewards", []))
        for state, actions in data.get("q_table", {}).items():
            for action, val in actions.items():
                self._Q[state][action] = float(val)

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": len(self._episode_rewards),
            "epsilon": self._epsilon,
            "mean_reward_last_100": (
                sum(self._episode_rewards[-100:]) / min(100, len(self._episode_rewards))
                if self._episode_rewards else 0.0
            ),
        }
