"""
ruleforge/templates.py
----------------------
Rule template learning — extract abstract templates from concrete rules
and rank them by usefulness.

A *template* abstracts away parameter values, leaving only operation
sequences annotated with parameter types.

Example::

    Rules:  l$1   l$!   l$@   l$2025
    Template: lowercase → append(char)

Templates are serialised as sequences of ``(op_name, param_type)`` pairs.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parser import Parser, Token, _arity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter type classifier
# ---------------------------------------------------------------------------

def _param_type(cmd: str, param: str) -> str:
    """Return a descriptive type string for *param* of operation *cmd*."""
    if not param:
        return ""
    ar = _arity(cmd)
    if ar == 0:
        return ""
    if ar == 1:
        ch = param[0]
        if ch.isdigit():
            return "digit"
        if ch.isalpha():
            return "alpha"
        return "symbol"
    if ar == 2:
        return f"{'digit' if param[0].isdigit() else 'char'}_{'digit' if param[1].isdigit() else 'char'}"
    if ar == 3:
        return "pos_char_char"
    return "unknown"


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """An abstract template extracted from one or more concrete rules.

    Attributes:
        steps: Tuple of ``(op_char, param_type)`` pairs.
    """

    steps: tuple[tuple[str, str], ...]

    @classmethod
    def from_tokens(cls, tokens: list[Token]) -> "Template":
        """Build a :class:`Template` from a token list."""
        steps = tuple((t.cmd, _param_type(t.cmd, t.param)) for t in tokens)
        return cls(steps=steps)

    def signature(self) -> str:
        """Human-readable signature string."""
        parts = []
        for cmd, ptype in self.steps:
            from .parser import Parser as _P
            name = _P.op_name(cmd)
            parts.append(f"{name}({ptype})" if ptype else name)
        return " → ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": list(self.steps), "signature": self.signature()}

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# TemplateBank
# ---------------------------------------------------------------------------

@dataclass
class TemplateEntry:
    """Stores a template alongside aggregated statistics."""

    template: Template
    count: int = 0
    example_rules: list[str] = field(default_factory=list)
    usefulness_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.template.signature(),
            "steps": list(self.template.steps),
            "count": self.count,
            "usefulness_score": self.usefulness_score,
            "examples": self.example_rules[:5],
        }


class TemplateLearner:
    """Extract and rank reusable rule templates.

    Args:
        parser:         A :class:`~ruleforge.parser.Parser` instance.
        max_examples:   Maximum example rules stored per template.

    Usage::

        learner = TemplateLearner(parser)
        learner.ingest_rules(["l$1", "l$!", "l$2", "u$1"])
        ranked = learner.ranked_templates()
    """

    def __init__(
        self,
        parser: Parser,
        max_examples: int = 10,
    ) -> None:
        self._parser = parser
        self._max_examples = max_examples
        self._entries: dict[Template, TemplateEntry] = {}
        self._total_rules: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_rules(self, rules: list[str]) -> None:
        """Learn templates from an iterable of rule strings."""
        for rule in rules:
            tokens = self._parser.try_parse(rule)
            if not tokens:
                continue
            tmpl = Template.from_tokens(tokens)
            if tmpl not in self._entries:
                self._entries[tmpl] = TemplateEntry(template=tmpl)
            entry = self._entries[tmpl]
            entry.count += 1
            if len(entry.example_rules) < self._max_examples:
                entry.example_rules.append(rule)
            self._total_rules += 1

        self._score_all()

    def ingest_file(self, path: Path) -> None:
        """Learn templates from *path*."""
        logger.info("Learning templates from %s", path)
        rules = self._parser.parse_file(path)
        self.ingest_rules(rules)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_all(self) -> None:
        """Compute usefulness scores for all templates."""
        if not self._entries or self._total_rules == 0:
            return

        # Frequency-weighted score with a diversity bonus
        for entry in self._entries.values():
            freq = entry.count / self._total_rules
            # IDF-like rarity: prefer templates that are frequent enough to
            # be useful but not so common they're trivially obvious.
            # Peak around 1–10% frequency.
            freq_bonus = -abs(math.log10(max(freq, 1e-9)) + 2)
            # Longer templates (more ops) are generally more specific / useful
            length_bonus = math.log1p(len(entry.template))
            entry.usefulness_score = freq_bonus + length_bonus

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def ranked_templates(self, top_n: int | None = None) -> list[TemplateEntry]:
        """Return templates sorted by usefulness (descending)."""
        ranked = sorted(
            self._entries.values(),
            key=lambda e: e.usefulness_score,
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked

    def top_n(self, n: int = 20) -> list[TemplateEntry]:
        """Return the top *n* most useful templates."""
        return self.ranked_templates(top_n=n)

    def generate_from_template(
        self,
        template: Template,
        param_sampler: "ParamSampler | None" = None,
    ) -> str | None:
        """Instantiate *template* by sampling concrete parameter values.

        Args:
            template:      The template to instantiate.
            param_sampler: Optional custom parameter sampler.
                           Uses :class:`DefaultParamSampler` if not provided.

        Returns:
            A concrete rule string, or ``None`` if instantiation fails.
        """
        sampler = param_sampler or DefaultParamSampler()
        parts: list[str] = []
        for cmd, _ptype in template.steps:
            param = sampler.sample(cmd)
            if param is None:
                return None
            parts.append(cmd + param)
        candidate = "".join(parts)
        return candidate if self._parser.validate(candidate) else None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path, top_n: int | None = None) -> None:
        """Write ranked templates to *path* as JSON."""
        data = [e.to_dict() for e in self.ranked_templates(top_n=top_n)]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Templates exported to %s", path)


# ---------------------------------------------------------------------------
# Parameter Sampler
# ---------------------------------------------------------------------------

class ParamSampler:
    """Abstract base for parameter samplers."""

    def sample(self, cmd: str) -> str | None:
        raise NotImplementedError


class DefaultParamSampler(ParamSampler):
    """Simple deterministic parameter sampler (for testing)."""

    _DIGIT = "0"
    _ALPHA = "a"
    _SYMBOL = "!"

    def sample(self, cmd: str) -> str | None:
        ar = _arity(cmd)
        if ar == 0:
            return ""
        if ar == 1:
            return self._DIGIT
        if ar == 2:
            return self._ALPHA + self._ALPHA
        if ar == 3:
            return "0" + self._ALPHA + self._ALPHA
        return None


class FrequencyParamSampler(ParamSampler):
    """Sample parameters proportional to learned frequencies.

    Args:
        param_freq: ``{cmd: Counter(param_str → count)}`` mapping from
                    :class:`~ruleforge.analyzer.Analyzer`.
        rng:        Random source.
    """

    def __init__(
        self,
        param_freq: dict[str, Counter[str]],
        rng: "random.Random | None" = None,
    ) -> None:
        import random as _random
        self._freq = param_freq
        self._rng = rng or _random.Random()

        # Fallbacks
        self._safe1 = list("0123456789!@#$%^&*.-_")
        self._safe_pos = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def sample(self, cmd: str) -> str | None:
        ar = _arity(cmd)
        if ar == 0:
            return ""
        pool = self._freq.get(cmd)
        if pool:
            items = list(pool.items())
            total = sum(v for _, v in items)
            if total > 0:
                r = self._rng.uniform(0, total)
                acc = 0.0
                for param, w in items:
                    acc += w
                    if acc >= r:
                        return param
                return items[-1][0]
        # Fallback
        if ar == 1:
            return self._rng.choice(self._safe1)
        if ar == 2:
            return self._rng.choice(self._safe1) + self._rng.choice(self._safe1)
        if ar == 3:
            return (
                self._rng.choice(self._safe_pos)
                + self._rng.choice(self._safe1)
                + self._rng.choice(self._safe1)
            )
        return None
