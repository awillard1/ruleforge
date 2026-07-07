"""
ruleforge/analyzer.py
---------------------
Rule Analyzer — collect statistics from one or more Hashcat rule files.

Statistics collected
~~~~~~~~~~~~~~~~~~~~
- Operation frequency
- Operation ordering (start / end positions)
- Parameter frequencies per operation
- Operation pairs (bigrams)
- Operation triplets (trigrams)
- Transition probabilities
- Rule lengths distribution
- Unique / duplicate / invalid rule counts
- Shannon entropy
- Rule complexity scores
- Operation rarity scores
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parser import Parser, Token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Aggregated statistics produced by :class:`Analyzer`."""

    # Counts
    total_lines: int = 0
    valid_lines: int = 0
    invalid_lines: int = 0
    comment_or_empty: int = 0
    unique_count: int = 0
    duplicate_count: int = 0

    # Frequency tables (serialisable)
    cmd_freq: dict[str, int] = field(default_factory=dict)
    start_freq: dict[str, int] = field(default_factory=dict)
    end_freq: dict[str, int] = field(default_factory=dict)
    param_freq: dict[str, dict[str, int]] = field(default_factory=dict)
    pair_freq: dict[str, int] = field(default_factory=dict)   # "AB" → count
    triple_freq: dict[str, int] = field(default_factory=dict)  # "ABC" → count
    len_dist: dict[int, int] = field(default_factory=dict)

    # Derived metrics
    transition_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    cmd_rarity: dict[str, float] = field(default_factory=dict)
    entropy: float = 0.0
    mean_complexity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "counts": {
                "total_lines": self.total_lines,
                "valid_lines": self.valid_lines,
                "invalid_lines": self.invalid_lines,
                "comment_or_empty": self.comment_or_empty,
                "unique": self.unique_count,
                "duplicates": self.duplicate_count,
            },
            "cmd_freq": self.cmd_freq,
            "start_freq": self.start_freq,
            "end_freq": self.end_freq,
            "param_freq": self.param_freq,
            "pair_freq": self.pair_freq,
            "triple_freq": self.triple_freq,
            "len_dist": {str(k): v for k, v in self.len_dist.items()},
            "transition_probs": self.transition_probs,
            "cmd_rarity": self.cmd_rarity,
            "entropy": self.entropy,
            "mean_complexity": self.mean_complexity,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class Analyzer:
    """Ingest rule files and build statistical models.

    Args:
        parser: A :class:`~ruleforge.parser.Parser` instance.

    Usage::

        parser = Parser()
        analyzer = Analyzer(parser)
        analyzer.ingest_file(Path("best64.rule"))
        result = analyzer.result()
        print(result.to_json())
    """

    def __init__(self, parser: Parser) -> None:
        self._parser = parser

        self._total_lines: int = 0
        self._valid_lines: int = 0
        self._invalid_lines: int = 0
        self._comment_or_empty: int = 0

        self._unique_rules: set[str] = set()
        self._seen_all: Counter[str] = Counter()

        self._cmd: Counter[str] = Counter()
        self._start: Counter[str] = Counter()
        self._end: Counter[str] = Counter()
        self._params: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._pairs: Counter[str] = Counter()
        self._triples: Counter[str] = Counter()
        self._trans: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._len_dist: Counter[int] = Counter()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, path: Path) -> None:
        """Ingest all rules from *path*.

        Args:
            path: Path to a Hashcat rule file.
        """
        logger.info("Ingesting rule file: %s", path)
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                self._total_lines += 1
                s = raw.strip()

                if not s or s.startswith("#"):
                    self._comment_or_empty += 1
                    continue

                tokens = self._parser.try_parse(s)
                if tokens is None:
                    self._invalid_lines += 1
                    logger.debug("Invalid rule: %r", s)
                    continue

                rule = self._parser.serialize(tokens)
                if not rule or not self._parser.validate(rule):
                    self._invalid_lines += 1
                    continue

                self._valid_lines += 1
                self._seen_all[rule] += 1
                self._unique_rules.add(rule)

        # Rebuild operation statistics from unique valid rules
        self._rebuild_stats()
        logger.info(
            "Ingested %s: valid=%d unique=%d invalid=%d",
            path.name,
            self._valid_lines,
            len(self._unique_rules),
            self._invalid_lines,
        )

    def ingest_rules(self, rules: list[str]) -> None:
        """Ingest rules from an in-memory list."""
        for rule in rules:
            self._total_lines += 1
            s = rule.strip()
            if not s or s.startswith("#"):
                self._comment_or_empty += 1
                continue
            tokens = self._parser.try_parse(s)
            if tokens is None:
                self._invalid_lines += 1
                continue
            canonical = self._parser.serialize(tokens)
            if not canonical:
                self._invalid_lines += 1
                continue
            self._valid_lines += 1
            self._seen_all[canonical] += 1
            self._unique_rules.add(canonical)
        self._rebuild_stats()

    def _rebuild_stats(self) -> None:
        """Recompute all counters from the current set of unique rules."""
        self._cmd.clear()
        self._start.clear()
        self._end.clear()
        self._params.clear()
        self._pairs.clear()
        self._triples.clear()
        self._trans.clear()
        self._len_dist.clear()

        for rule in self._unique_rules:
            toks = self._parser.try_parse(rule)
            if not toks:
                continue

            self._len_dist[len(toks)] += 1
            self._start[toks[0].cmd] += 1
            self._end[toks[-1].cmd] += 1

            for t in toks:
                self._cmd[t.cmd] += 1
                if t.param:
                    self._params[t.cmd][t.param] += 1

            for i in range(len(toks) - 1):
                pair = toks[i].cmd + toks[i + 1].cmd
                self._pairs[pair] += 1
                self._trans[toks[i].cmd][toks[i + 1].cmd] += 1

            for i in range(len(toks) - 2):
                triple = toks[i].cmd + toks[i + 1].cmd + toks[i + 2].cmd
                self._triples[triple] += 1

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def _compute_transition_probs(self) -> dict[str, dict[str, float]]:
        probs: dict[str, dict[str, float]] = {}
        for src, dests in self._trans.items():
            total = sum(dests.values())
            probs[src] = {dst: cnt / total for dst, cnt in dests.items()}
        return probs

    def _compute_rarity(self) -> dict[str, float]:
        """Rarity score: inverse frequency normalized to [0, 1]."""
        if not self._cmd:
            return {}
        max_freq = max(self._cmd.values())
        return {cmd: 1.0 - (freq / max_freq) for cmd, freq in self._cmd.items()}

    def _compute_entropy(self) -> float:
        """Shannon entropy over operation frequencies."""
        total = sum(self._cmd.values())
        if total == 0:
            return 0.0
        return -sum(
            (f / total) * math.log2(f / total) for f in self._cmd.values() if f > 0
        )

    def _complexity(self, tokens: list[Token]) -> float:
        """Heuristic complexity score for a single rule."""
        n = len(tokens)
        unique_ops = len({t.cmd for t in tokens})
        param_count = sum(1 for t in tokens if t.param)
        return n * 0.4 + unique_ops * 0.4 + param_count * 0.2

    def _compute_mean_complexity(self) -> float:
        scores = []
        for rule in self._unique_rules:
            toks = self._parser.try_parse(rule)
            if toks:
                scores.append(self._complexity(toks))
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result(self) -> AnalysisResult:
        """Return a snapshot of the current analysis state."""
        dup_count = sum(
            max(0, cnt - 1) for cnt in self._seen_all.values()
        )
        r = AnalysisResult(
            total_lines=self._total_lines,
            valid_lines=self._valid_lines,
            invalid_lines=self._invalid_lines,
            comment_or_empty=self._comment_or_empty,
            unique_count=len(self._unique_rules),
            duplicate_count=dup_count,
            cmd_freq=dict(self._cmd),
            start_freq=dict(self._start),
            end_freq=dict(self._end),
            param_freq={k: dict(v) for k, v in self._params.items()},
            pair_freq=dict(self._pairs),
            triple_freq=dict(self._triples),
            len_dist=dict(self._len_dist),
            transition_probs=self._compute_transition_probs(),
            cmd_rarity=self._compute_rarity(),
            entropy=self._compute_entropy(),
            mean_complexity=self._compute_mean_complexity(),
        )
        return r

    # ------------------------------------------------------------------
    # Read-only accessors (used by other modules)
    # ------------------------------------------------------------------

    @property
    def unique_rules(self) -> frozenset[str]:
        return frozenset(self._unique_rules)

    @property
    def cmd(self) -> Counter[str]:
        return Counter(self._cmd)

    @property
    def start(self) -> Counter[str]:
        return Counter(self._start)

    @property
    def end(self) -> Counter[str]:
        return Counter(self._end)

    @property
    def params(self) -> dict[str, Counter[str]]:
        return {k: Counter(v) for k, v in self._params.items()}

    @property
    def trans(self) -> dict[str, Counter[str]]:
        return {k: Counter(v) for k, v in self._trans.items()}

    @property
    def len_dist(self) -> Counter[int]:
        return Counter(self._len_dist)

    def export_json(self, path: Path, indent: int = 2) -> None:
        """Write analysis result to *path* as JSON."""
        path.write_text(self.result().to_json(indent=indent), encoding="utf-8")
        logger.info("Analysis exported to %s", path)
