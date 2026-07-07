"""
ruleforge/grammar.py
--------------------
Probabilistic Context-Free Grammar (PCFG) learning from password corpora.

Learns structures like:
  Word+Year, Name+123, Word+!, Month+Year, Company+Number

and uses the learned grammar to generate plausible password candidates
or Hashcat rule sequences.
"""

from __future__ import annotations

import json
import logging
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token types (terminal categories)
# ---------------------------------------------------------------------------

class TokType:
    WORD = "Word"
    NAME = "Name"
    NUMBER = "Number"
    YEAR = "Year"
    MONTH = "Month"
    DAY = "Day"
    SYMBOL = "Symbol"
    LEET = "Leet"
    UPPER = "Upper"
    LOWER = "Lower"
    MIXED = "Mixed"
    OTHER = "Other"


# ---------------------------------------------------------------------------
# Password segmenter
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2}|2[1-9]\d{2})$")
_MONTH_RE = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)$",
    re.IGNORECASE,
)
_DAY_RE = re.compile(r"^(0?[1-9]|[12]\d|3[01])$")
_NUMBER_RE = re.compile(r"^\d+$")
_SYMBOL_RE = re.compile(r"^[!@#$%^&*\-_.]+$")
_WORD_RE = re.compile(r"^[a-zA-Z]{3,}$")
_LEET_MAP = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}


def _classify_segment(seg: str) -> str:
    if _YEAR_RE.match(seg):
        return TokType.YEAR
    if _NUMBER_RE.match(seg):
        return TokType.NUMBER
    if _SYMBOL_RE.match(seg):
        return TokType.SYMBOL
    if _MONTH_RE.match(seg):
        return TokType.MONTH
    if _DAY_RE.match(seg):
        return TokType.DAY
    if _WORD_RE.match(seg):
        if seg[0].isupper() and seg[1:].islower():
            return TokType.NAME
        if seg.isupper():
            return TokType.UPPER
        if seg.islower():
            return TokType.LOWER
        return TokType.MIXED
    # Check leet
    de_leet = "".join(_LEET_MAP.get(c, c) for c in seg.lower())
    if re.fullmatch(r"[a-z]+", de_leet):
        return TokType.LEET
    return TokType.OTHER


def _segment_password(pw: str) -> list[str]:
    """Split a password into typed segments."""
    # Split on boundaries between digit/alpha/symbol runs
    return re.findall(r"[a-zA-Z]+|\d+|[!@#$%^&*\-_.]+|.", pw)


def _classify_password(pw: str) -> list[str]:
    """Return sequence of token type strings for *pw*."""
    segs = _segment_password(pw)
    return [_classify_segment(s) for s in segs]


# ---------------------------------------------------------------------------
# Grammar rule
# ---------------------------------------------------------------------------

@dataclass
class GrammarRule:
    """A single PCFG production: LHS → RHS with probability."""

    lhs: str
    rhs: tuple[str, ...]
    prob: float = 0.0
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lhs": self.lhs,
            "rhs": list(self.rhs),
            "prob": self.prob,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# PCFG Learner
# ---------------------------------------------------------------------------


class PCFGLearner:
    """Learn a PCFG from password structures.

    Args:
        max_length: Ignore passwords longer than this.
    """

    ROOT = "Password"

    def __init__(self, max_length: int = 32) -> None:
        self._max_length = max_length
        # lhs → Counter(rhs_tuple → count)
        self._counts: defaultdict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
        self._total_passwords: int = 0

    def learn(self, passwords: list[str]) -> None:
        """Learn from a list of password strings."""
        for pw in passwords:
            if not pw or len(pw) > self._max_length:
                continue
            structure = tuple(_classify_password(pw))
            if structure:
                self._counts[self.ROOT][structure] += 1
                self._total_passwords += 1

        logger.debug(
            "PCFG: learned from %d passwords, %d unique structures",
            self._total_passwords,
            len(self._counts[self.ROOT]),
        )

    def learn_from_file(self, path: Path) -> None:
        """Learn from a password file (one password per line)."""
        passwords: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                pw = line.rstrip("\n")
                if pw:
                    passwords.append(pw)
        self.learn(passwords)
        logger.info("PCFG learned from file: %s", path)

    def top_structures(self, n: int = 20) -> list[tuple[tuple[str, ...], int]]:
        """Return the most common password structures."""
        return self._counts[self.ROOT].most_common(n)

    def build(self) -> "PCFG":
        """Convert learned counts to a normalized :class:`PCFG`."""
        rules: list[GrammarRule] = []
        for lhs, rhs_counts in self._counts.items():
            total = sum(rhs_counts.values())
            for rhs, count in rhs_counts.items():
                rules.append(GrammarRule(
                    lhs=lhs,
                    rhs=rhs,
                    prob=count / total if total > 0 else 0.0,
                    count=count,
                ))
        return PCFG(rules=rules, total_passwords=self._total_passwords)


# ---------------------------------------------------------------------------
# PCFG
# ---------------------------------------------------------------------------


class PCFG:
    """Probabilistic Context-Free Grammar for password structures.

    Args:
        rules:             List of :class:`GrammarRule` objects.
        total_passwords:   Total number of training passwords.
    """

    def __init__(
        self,
        rules: list[GrammarRule],
        total_passwords: int = 0,
    ) -> None:
        self._rules = rules
        self._total = total_passwords
        # Index: lhs → list of (rhs, prob)
        self._index: defaultdict[str, list[tuple[tuple[str, ...], float]]] = defaultdict(list)
        for rule in rules:
            self._index[rule.lhs].append((rule.rhs, rule.prob))
        # Sort by probability descending
        for lhs in self._index:
            self._index[lhs].sort(key=lambda rp: rp[1], reverse=True)

    def top_n(self, lhs: str = "Password", n: int = 20) -> list[GrammarRule]:
        """Return top-n rules for *lhs* sorted by probability."""
        return sorted(
            [r for r in self._rules if r.lhs == lhs],
            key=lambda r: r.prob,
            reverse=True,
        )[:n]

    def sample_structure(self, lhs: str = "Password", rng: Any = None) -> tuple[str, ...] | None:
        """Sample a structure from the grammar."""
        import random as _random
        _rng = rng or _random.Random()
        productions = self._index.get(lhs, [])
        if not productions:
            return None
        rhs_list, probs = zip(*productions)
        total = sum(probs)
        if total <= 0:
            return None
        r = _rng.uniform(0, total)
        acc = 0.0
        for rhs, prob in zip(rhs_list, probs):
            acc += prob
            if acc >= r:
                return rhs
        return rhs_list[-1]

    def log_prob_structure(self, structure: tuple[str, ...]) -> float:
        """Return log-probability of a password structure."""
        probs = {rhs: prob for rhs, prob in self._index.get("Password", [])}
        p = probs.get(structure, 0.0)
        return math.log(p) if p > 0 else float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_passwords": self._total,
            "rules": [r.to_dict() for r in self._rules],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PCFG":
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = [
            GrammarRule(
                lhs=r["lhs"],
                rhs=tuple(r["rhs"]),
                prob=float(r["prob"]),
                count=int(r["count"]),
            )
            for r in data["rules"]
        ]
        return cls(rules=rules, total_passwords=int(data.get("total_passwords", 0)))
