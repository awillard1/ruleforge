"""
ruleforge/mcts.py
-----------------
Monte Carlo Tree Search for Hashcat rule construction.

Treats each rule as a tree where:
- The root is an empty rule.
- Each node represents a rule with n operations.
- Each edge represents appending one operation token.
- Leaf evaluation uses the provided fitness function.

Uses UCT (Upper Confidence bounds applied to Trees) for node selection.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .parser import Parser, Token, _arity
from .generator import RuleGenerator

logger = logging.getLogger(__name__)

# Type alias
FitnessFn = Callable[[str], float]


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


class MCTSNode:
    """A node in the MCTS tree.

    Attributes:
        rule:     The partial rule built so far.
        parent:   Parent node, or ``None`` for root.
        children: Expanded child nodes.
        visits:   Number of times this node has been visited.
        value:    Cumulative simulation reward.
    """

    __slots__ = ("rule", "parent", "children", "visits", "value", "_untried")

    def __init__(self, rule: str = "", parent: "MCTSNode | None" = None) -> None:
        self.rule = rule
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self._untried: list[str] | None = None  # lazy-initialized moves

    def is_fully_expanded(self, moves: list[str]) -> bool:
        if self._untried is None:
            return False
        return len(self._untried) == 0

    def best_child(self, c: float) -> "MCTSNode":
        """UCT child selection."""

        def uct(child: "MCTSNode") -> float:
            if child.visits == 0:
                return float("inf")
            exploit = child.value / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore

        return max(self.children, key=uct)

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward


# ---------------------------------------------------------------------------
# MCTS Config
# ---------------------------------------------------------------------------


@dataclass
class MCTSConfig:
    """Configuration for the MCTS engine."""

    n_simulations: int = 200     # simulations per search call
    max_depth: int = 10          # maximum rule length
    exploration_constant: float = 1.414  # UCT c parameter
    rollout_depth: int = 5       # random rollout length


# ---------------------------------------------------------------------------
# MCTS Engine
# ---------------------------------------------------------------------------


class MCTSOptimizer:
    """Monte Carlo Tree Search optimizer for rule construction.

    Args:
        parser:     A :class:`~ruleforge.parser.Parser` instance.
        generator:  A :class:`~ruleforge.generator.RuleGenerator` for
                    move generation and parameter sampling.
        fitness_fn: Rule → float fitness callable.
        config:     :class:`MCTSConfig`.
        rng:        Random source.
    """

    def __init__(
        self,
        parser: Parser,
        generator: RuleGenerator,
        fitness_fn: FitnessFn,
        config: MCTSConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._parser = parser
        self._gen = generator
        self._fitness_fn = fitness_fn
        self._cfg = config or MCTSConfig()
        self._rng = rng or random.Random()
        self._root = MCTSNode(rule="")
        self._best_rules: list[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def _moves(self, rule: str) -> list[str]:
        """Return all possible one-operation extensions of *rule*."""
        toks = self._parser.try_parse(rule)
        if toks is None:
            toks = []
        if len(toks) >= self._cfg.max_depth:
            return []
        moves: list[str] = []
        for cmd in sorted(self._gen._allowed_cmds):
            p = self._gen._sample_param(cmd)
            if p is None:
                continue
            new_rule = rule + cmd + p
            if self._parser.validate(new_rule, max_ops=self._cfg.max_depth):
                moves.append(new_rule)
        return moves

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend the tree using UCT until a non-fully-expanded node."""
        while node.children:
            moves = self._moves(node.rule)
            if not node.is_fully_expanded(moves):
                break
            node = node.best_child(self._cfg.exploration_constant)
        return node

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add one new child to *node*."""
        moves = self._moves(node.rule)
        if node._untried is None:
            node._untried = list(moves)
            self._rng.shuffle(node._untried)

        if not node._untried:
            return node

        move = node._untried.pop()
        child = MCTSNode(rule=move, parent=node)
        node.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Simulation (rollout)
    # ------------------------------------------------------------------

    def _rollout(self, node: MCTSNode) -> float:
        """Random rollout from *node*; return fitness of final rule."""
        rule = node.rule
        for _ in range(self._cfg.rollout_depth):
            moves = self._moves(rule)
            if not moves:
                break
            rule = self._rng.choice(moves)

        if not rule or not self._parser.validate(rule):
            return 0.0

        score = self._fitness_fn(rule)
        self._best_rules.append((rule, score))
        return score

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, root_rule: str = "") -> list[str]:
        """Run MCTS from *root_rule*.

        Returns:
            Top rule strings discovered (sorted by fitness, descending).
        """
        self._root = MCTSNode(rule=root_rule)
        self._best_rules.clear()

        for _ in range(self._cfg.n_simulations):
            node = self._select(self._root)
            if self._moves(node.rule):
                node = self._expand(node)
            reward = self._rollout(node)
            self._backpropagate(node, reward)

        # Deduplicate and sort
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for rule, score in sorted(self._best_rules, key=lambda rs: rs[1], reverse=True):
            if rule not in seen and self._parser.validate(rule):
                seen.add(rule)
                unique.append((rule, score))

        logger.info(
            "MCTS search complete: simulations=%d unique_rules=%d",
            self._cfg.n_simulations,
            len(unique),
        )
        return [r for r, _ in unique]

    def top_rules(self, n: int = 100) -> list[str]:
        """Return top *n* rules from the last search."""
        seen: set[str] = set()
        out: list[str] = []
        for rule, _ in sorted(self._best_rules, key=lambda rs: rs[1], reverse=True):
            if rule not in seen and self._parser.validate(rule):
                seen.add(rule)
                out.append(rule)
            if len(out) >= n:
                break
        return out

    def stats(self) -> dict[str, Any]:
        if not self._best_rules:
            return {"total_rules": 0}
        scores = [s for _, s in self._best_rules]
        return {
            "total_rules": len(self._best_rules),
            "best_score": max(scores),
            "mean_score": sum(scores) / len(scores),
        }
