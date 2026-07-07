"""
ruleforge/passwords.py
----------------------
Password Corpus Analyzer.

Extracts structural features and statistics from password lists:
- Lengths
- Character classes
- Year / month / day patterns
- Keyboard walks
- Leet substitutions
- CamelCase / mixed case
- Common suffixes and prefixes
- Repeated characters
- Numbers and symbols
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"19\d{2}|20\d{2}")
_MONTH_RE = re.compile(
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december",
    re.IGNORECASE,
)
_DAY_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])\b")
_LEET_MAP: dict[str, str] = {
    "4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a",
}
_KEYBOARD_ROWS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "1234567890",
]
_KEYBOARD_WALKS_RE = re.compile(
    r"(qwerty|asdf|zxcv|qwer|wert|erty|rtyu|tyui|yuio|uiop|"
    r"asdfg|sdfgh|dfghj|fghjk|ghjkl|12345|23456|34567|45678|56789|67890)",
    re.IGNORECASE,
)
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _has_keyboard_walk(pw: str) -> bool:
    return bool(_KEYBOARD_WALKS_RE.search(pw))


def _has_leet(pw: str) -> bool:
    return any(c in _LEET_MAP for c in pw)


def _is_camel_case(pw: str) -> bool:
    return bool(_CAMEL_RE.search(pw))


def _char_classes(pw: str) -> dict[str, bool]:
    return {
        "has_upper": any(c.isupper() for c in pw),
        "has_lower": any(c.islower() for c in pw),
        "has_digit": any(c.isdigit() for c in pw),
        "has_symbol": any(not c.isalnum() for c in pw),
    }


def _common_prefix(pw: str, length: int = 3) -> str:
    return pw[:length] if len(pw) >= length else pw


def _common_suffix(pw: str, length: int = 3) -> str:
    return pw[-length:] if len(pw) >= length else pw


# ---------------------------------------------------------------------------
# Password Statistics
# ---------------------------------------------------------------------------


@dataclass
class PasswordStats:
    """Aggregated statistics over a password corpus."""

    total: int = 0
    length_dist: dict[int, int] = field(default_factory=dict)
    char_class_dist: dict[str, int] = field(default_factory=dict)

    has_year: int = 0
    has_month: int = 0
    has_day: int = 0
    has_keyboard_walk: int = 0
    has_leet: int = 0
    has_camel_case: int = 0
    has_repeat: int = 0

    year_values: dict[str, int] = field(default_factory=dict)
    top_prefixes: dict[str, int] = field(default_factory=dict)
    top_suffixes: dict[str, int] = field(default_factory=dict)
    top_numbers: dict[str, int] = field(default_factory=dict)
    top_symbols: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "length_dist": {str(k): v for k, v in self.length_dist.items()},
            "char_class_dist": self.char_class_dist,
            "has_year": self.has_year,
            "has_month": self.has_month,
            "has_day": self.has_day,
            "has_keyboard_walk": self.has_keyboard_walk,
            "has_leet": self.has_leet,
            "has_camel_case": self.has_camel_case,
            "has_repeat": self.has_repeat,
            "year_values": dict(sorted(self.year_values.items(),
                                       key=lambda kv: kv[1], reverse=True)[:20]),
            "top_prefixes": dict(sorted(self.top_prefixes.items(),
                                        key=lambda kv: kv[1], reverse=True)[:50]),
            "top_suffixes": dict(sorted(self.top_suffixes.items(),
                                        key=lambda kv: kv[1], reverse=True)[:50]),
            "top_numbers": dict(sorted(self.top_numbers.items(),
                                       key=lambda kv: kv[1], reverse=True)[:30]),
            "top_symbols": dict(sorted(self.top_symbols.items(),
                                       key=lambda kv: kv[1], reverse=True)[:20]),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class PasswordAnalyzer:
    """Analyze password corpora and extract statistical features.

    Args:
        prefix_len:  Length for prefix extraction.
        suffix_len:  Length for suffix extraction.
        min_length:  Skip passwords shorter than this.
        max_length:  Skip passwords longer than this.
    """

    def __init__(
        self,
        prefix_len: int = 3,
        suffix_len: int = 3,
        min_length: int = 1,
        max_length: int = 64,
    ) -> None:
        self._prefix_len = prefix_len
        self._suffix_len = suffix_len
        self._min_len = min_length
        self._max_len = max_length

        self._total = 0
        self._len_dist: Counter[int] = Counter()
        self._char_class_combos: Counter[str] = Counter()

        self._has_year = 0
        self._has_month = 0
        self._has_day = 0
        self._has_walk = 0
        self._has_leet = 0
        self._has_camel = 0
        self._has_repeat = 0

        self._year_vals: Counter[str] = Counter()
        self._prefixes: Counter[str] = Counter()
        self._suffixes: Counter[str] = Counter()
        self._numbers: Counter[str] = Counter()
        self._symbols: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def analyze(self, passwords: list[str]) -> None:
        """Analyze a list of passwords."""
        for pw in passwords:
            if not pw or not (self._min_len <= len(pw) <= self._max_len):
                continue
            self._process(pw)
        logger.debug("Analyzed %d passwords", self._total)

    def analyze_file(self, path: Path) -> None:
        """Analyze passwords from *path*."""
        logger.info("Analyzing password file: %s", path)
        passwords: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                pw = line.rstrip("\n")
                if pw:
                    passwords.append(pw)
        self.analyze(passwords)

    def _process(self, pw: str) -> None:
        self._total += 1
        self._len_dist[len(pw)] += 1

        # Char classes
        cc = _char_classes(pw)
        combo = (
            ("U" if cc["has_upper"] else "")
            + ("L" if cc["has_lower"] else "")
            + ("D" if cc["has_digit"] else "")
            + ("S" if cc["has_symbol"] else "")
        )
        self._char_class_combos[combo] += 1

        # Feature flags
        if _YEAR_RE.search(pw):
            self._has_year += 1
            for y in _YEAR_RE.findall(pw):
                self._year_vals[y] += 1
        if _MONTH_RE.search(pw):
            self._has_month += 1
        if _DAY_RE.search(pw):
            self._has_day += 1
        if _has_keyboard_walk(pw):
            self._has_walk += 1
        if _has_leet(pw):
            self._has_leet += 1
        if _is_camel_case(pw):
            self._has_camel += 1
        if _REPEAT_RE.search(pw):
            self._has_repeat += 1

        # Prefix / suffix
        self._prefixes[_common_prefix(pw, self._prefix_len)] += 1
        self._suffixes[_common_suffix(pw, self._suffix_len)] += 1

        # Embedded numbers
        for num in re.findall(r"\d+", pw):
            self._numbers[num] += 1

        # Symbols
        for sym in re.findall(r"[!@#$%^&*\-_.]+", pw):
            self._symbols[sym] += 1

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result(self) -> PasswordStats:
        """Return a snapshot of all collected statistics."""
        return PasswordStats(
            total=self._total,
            length_dist=dict(self._len_dist),
            char_class_dist=dict(self._char_class_combos),
            has_year=self._has_year,
            has_month=self._has_month,
            has_day=self._has_day,
            has_keyboard_walk=self._has_walk,
            has_leet=self._has_leet,
            has_camel_case=self._has_camel,
            has_repeat=self._has_repeat,
            year_values=dict(self._year_vals),
            top_prefixes=dict(self._prefixes),
            top_suffixes=dict(self._suffixes),
            top_numbers=dict(self._numbers),
            top_symbols=dict(self._symbols),
        )

    def export_json(self, path: Path) -> None:
        """Write analysis result to *path* as JSON."""
        path.write_text(self.result().to_json(), encoding="utf-8")
        logger.info("Password analysis exported to %s", path)
