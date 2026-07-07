"""
ruleforge/scoring.py
--------------------
Scoring Engine — composite rule quality score.

Combines multiple signals with configurable weights:
- Novelty (inverse frequency of shape)
- Coverage (estimated unique outputs)
- Entropy (operation diversity)
- Uniqueness (not previously generated)
- Runtime efficiency (inversely proportional to hashcat runtime)
- Historical success (stored fitness scores)
- Parameter diversity
- Operation diversity
- Template rarity
- Markov probability
- N-gram probability
- Grammar usefulness
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .parser import Parser, Token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------


@dataclass
class ScoreWeights:
    """Configurable weights for each scoring component."""

    novelty: float = 1.0
    coverage: float = 1.0
    entropy: float = 0.5
    uniqueness: float = 0.5
    runtime: float = 0.3
    historical: float = 0.8
    param_diversity: float = 0.5
    op_diversity: float = 0.7
    template_rarity: float = 0.6
    markov_prob: float = 0.4
    ngram_prob: float = 0.4
    grammar: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return {
            "novelty": self.novelty,
            "coverage": self.coverage,
            "entropy": self.entropy,
            "uniqueness": self.uniqueness,
            "runtime": self.runtime,
            "historical": self.historical,
            "param_diversity": self.param_diversity,
            "op_diversity": self.op_diversity,
            "template_rarity": self.template_rarity,
            "markov_prob": self.markov_prob,
            "ngram_prob": self.ngram_prob,
            "grammar": self.grammar,
        }


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Detailed score for a single rule."""

    rule: str
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "total": self.total, "components": self.components}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """Compute composite quality scores for Hashcat rules.

    Args:
        parser:       A :class:`~ruleforge.parser.Parser` instance.
        cmd_freq:     Counter of operation frequencies from analyzer.
        param_freq:   Parameter frequency mapping from analyzer.
        trans:        Transition table from analyzer.
        start_freq:   Start-operation frequency from analyzer.
        weights:      :class:`ScoreWeights`.
        known_rules:  Set of already-seen rules (for uniqueness penalty).
        history:      Historical fitness scores ``{rule → float}``.
    """

    def __init__(
        self,
        parser: Parser,
        cmd_freq: Counter[str],
        param_freq: dict[str, Counter[str]],
        trans: dict[str, Counter[str]],
        start_freq: Counter[str],
        weights: ScoreWeights | None = None,
        known_rules: set[str] | None = None,
        history: dict[str, float] | None = None,
    ) -> None:
        self._parser = parser
        self._cmd = cmd_freq
        self._params = param_freq
        self._trans = trans
        self._start = start_freq
        self._weights = weights or ScoreWeights()
        self._known = known_rules or set()
        self._history = history or {}

        # External model scores (optional; set via setters)
        self._ngram_scorer: Any | None = None
        self._markov_log_probs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # External model injection
    # ------------------------------------------------------------------

    def set_ngram_scorer(self, scorer: Any) -> None:
        """Inject an :class:`~ruleforge.ngram.NGramEngine` for log-prob scoring."""
        self._ngram_scorer = scorer

    def set_markov_log_probs(self, probs: dict[str, float]) -> None:
        """Inject precomputed Markov log-probabilities."""
        self._markov_log_probs = probs

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _novelty(self, toks: list[Token]) -> float:
        """Inverse commonness of the operation shape."""
        shape_freq = self._start.get(toks[0].cmd, 0) if toks else 0
        for i in range(len(toks) - 1):
            shape_freq += self._trans.get(toks[i].cmd, Counter()).get(toks[i + 1].cmd, 0)
        return -math.log(shape_freq + 1.0) * 0.2

    def _coverage(self, toks: list[Token]) -> float:
        """Heuristic: rules touching more distinct characters score higher."""
        unique_params = {t.param for t in toks if t.param}
        return math.log1p(len(unique_params)) * 0.5

    def _entropy(self, toks: list[Token]) -> float:
        """Shannon entropy of operation frequencies in this rule."""
        counts = Counter(t.cmd for t in toks)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log2(c / total) for c in counts.values() if c > 0
        )

    def _uniqueness(self, rule: str) -> float:
        """Penalty if rule is already known."""
        return -2.0 if rule in self._known else 0.0

    def _param_diversity(self, toks: list[Token]) -> float:
        """Diversity of parameter values."""
        params = [t.param for t in toks if t.param]
        if not params:
            return 0.0
        unique = len(set(params))
        return math.log1p(unique) / math.log1p(len(params))

    def _op_diversity(self, toks: list[Token]) -> float:
        """Ratio of unique operations to total operations."""
        if not toks:
            return 0.0
        return len({t.cmd for t in toks}) / len(toks)

    def _template_rarity(self, toks: list[Token]) -> float:
        """Rarity of parameter values relative to known corpus."""
        bonus = 0.0
        for t in toks:
            if not t.param:
                continue
            pool = self._params.get(t.cmd, Counter())
            denom = sum(pool.values()) + 1
            freq = pool.get(t.param, 0) + 1
            bonus += -math.log(freq / denom)
        return bonus

    def _markov_prob(self, rule: str) -> float:
        return self._markov_log_probs.get(rule, 0.0)

    def _ngram_prob(self, rule: str) -> float:
        if self._ngram_scorer is None:
            return 0.0
        try:
            return self._ngram_scorer.score_rule(rule, self._parser)
        except Exception:  # noqa: BLE001
            return 0.0

    def _historical(self, rule: str) -> float:
        return self._history.get(rule, 0.0)

    def _length_bonus(self, toks: list[Token]) -> float:
        n = len(toks)
        if 2 <= n <= 8:
            return 0.5
        if n > 12:
            return -0.5
        return 0.0

    def _repetitive_penalty(self, toks: list[Token]) -> float:
        counts = Counter(t.cmd for t in toks)
        max_rep = max(counts.values(), default=0)
        return -0.6 if max_rep >= 4 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, rule: str) -> float:
        """Return the composite score for *rule*."""
        return self.score_detailed(rule).total

    def score_detailed(self, rule: str) -> ScoreBreakdown:
        """Return a :class:`ScoreBreakdown` for *rule*."""
        toks = self._parser.try_parse(rule)
        if not toks:
            return ScoreBreakdown(rule=rule, total=float("-inf"))

        w = self._weights
        comps: dict[str, float] = {}

        def _add(name: str, weight: float, value: float) -> None:
            comps[name] = value
            return

        comps["novelty"] = self._novelty(toks)
        comps["coverage"] = self._coverage(toks)
        comps["entropy"] = self._entropy(toks)
        comps["uniqueness"] = self._uniqueness(rule)
        comps["param_diversity"] = self._param_diversity(toks)
        comps["op_diversity"] = self._op_diversity(toks)
        comps["template_rarity"] = self._template_rarity(toks)
        comps["markov_prob"] = self._markov_prob(rule)
        comps["ngram_prob"] = self._ngram_prob(rule)
        comps["historical"] = self._historical(rule)
        comps["length_bonus"] = self._length_bonus(toks)
        comps["repetitive_penalty"] = self._repetitive_penalty(toks)

        total = (
            w.novelty * comps["novelty"]
            + w.coverage * comps["coverage"]
            + w.entropy * comps["entropy"]
            + w.uniqueness * comps["uniqueness"]
            + w.param_diversity * comps["param_diversity"]
            + w.op_diversity * comps["op_diversity"]
            + w.template_rarity * comps["template_rarity"]
            + w.markov_prob * comps["markov_prob"]
            + w.ngram_prob * comps["ngram_prob"]
            + w.historical * comps["historical"]
            + comps["length_bonus"]
            + comps["repetitive_penalty"]
        )

        return ScoreBreakdown(rule=rule, total=total, components=comps)

    def rank(self, rules: list[str]) -> list[tuple[str, float]]:
        """Sort *rules* by score descending and return ``[(rule, score), …]``."""
        scored = [(r, self.score(r)) for r in rules]
        scored.sort(key=lambda rs: rs[1], reverse=True)
        return scored
