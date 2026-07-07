"""
ruleforge/masks.py
------------------
Mask Learning — generate, cluster, and export Hashcat mask files.

A Hashcat mask represents the character structure of a password:
  ?u = uppercase, ?l = lowercase, ?d = digit, ?s = special, ?a = any

Example:
  Password: Football2025!
  Mask:     ?u?l?l?l?l?l?l?l?d?d?d?d?s
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Character-class mapping
# ---------------------------------------------------------------------------

_UPPER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_LOWER_SET = set("abcdefghijklmnopqrstuvwxyz")
_DIGIT_SET = set("0123456789")
_SPECIAL_SET = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _char_class(ch: str) -> str:
    """Return the Hashcat mask token for a single character."""
    if ch in _UPPER_SET:
        return "?u"
    if ch in _LOWER_SET:
        return "?l"
    if ch in _DIGIT_SET:
        return "?d"
    if ch in _SPECIAL_SET:
        return "?s"
    return "?b"  # binary / other


def password_to_mask(password: str) -> str:
    """Convert *password* to a Hashcat mask string."""
    return "".join(_char_class(ch) for ch in password)


# ---------------------------------------------------------------------------
# Mask statistics
# ---------------------------------------------------------------------------


@dataclass
class MaskStats:
    """Statistics for a specific mask pattern."""

    mask: str
    count: int = 0
    total_length: int = 0
    examples: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def length(self) -> int:
        """Password length implied by this mask (number of ?x tokens)."""
        return self.mask.count("?")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask,
            "length": self.length,
            "count": self.count,
            "score": self.score,
            "examples": self.examples[:3],
        }


# ---------------------------------------------------------------------------
# Mask Learner
# ---------------------------------------------------------------------------


class MaskLearner:
    """Learn mask patterns from a password corpus.

    Args:
        max_examples:  Maximum example passwords stored per mask.
        min_length:    Ignore passwords shorter than this.
        max_length:    Ignore passwords longer than this.
    """

    def __init__(
        self,
        max_examples: int = 5,
        min_length: int = 4,
        max_length: int = 32,
    ) -> None:
        self._max_examples = max_examples
        self._min_len = min_length
        self._max_len = max_length
        self._mask_stats: dict[str, MaskStats] = {}
        self._total_passwords: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def learn(self, passwords: list[str]) -> None:
        """Learn masks from *passwords*."""
        for pw in passwords:
            if not pw or not (self._min_len <= len(pw) <= self._max_len):
                continue
            mask = password_to_mask(pw)
            if mask not in self._mask_stats:
                self._mask_stats[mask] = MaskStats(mask=mask)
            entry = self._mask_stats[mask]
            entry.count += 1
            entry.total_length += len(pw)
            if len(entry.examples) < self._max_examples:
                entry.examples.append(pw)
            self._total_passwords += 1

        self._score_all()
        logger.debug(
            "Mask learner: %d passwords → %d unique masks",
            self._total_passwords,
            len(self._mask_stats),
        )

    def learn_from_file(self, path: Path) -> None:
        """Learn masks from a password file."""
        passwords: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                pw = line.rstrip("\n")
                if pw:
                    passwords.append(pw)
        self.learn(passwords)

    # ------------------------------------------------------------------
    # Scoring / ranking
    # ------------------------------------------------------------------

    def _score_all(self) -> None:
        if self._total_passwords == 0:
            return
        for stats in self._mask_stats.values():
            freq = stats.count / self._total_passwords
            # Prefer frequent masks of medium length
            length_bonus = -abs(stats.length - 10) * 0.05
            stats.score = math.log(freq + 1e-9) + length_bonus

    def ranked(self, top_n: int | None = None) -> list[MaskStats]:
        """Return masks sorted by score descending."""
        ranked = sorted(
            self._mask_stats.values(),
            key=lambda m: m.score,
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked

    # ------------------------------------------------------------------
    # Clustering
    # -----------
    # Groups masks by length for simple length-bucket clustering.

    def clusters_by_length(self) -> dict[int, list[MaskStats]]:
        """Group masks by password length."""
        clusters: dict[int, list[MaskStats]] = {}
        for stats in self._mask_stats.values():
            length = stats.length
            if length not in clusters:
                clusters[length] = []
            clusters[length].append(stats)
        # Sort each bucket by score
        for length in clusters:
            clusters[length].sort(key=lambda m: m.score, reverse=True)
        return clusters

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_hcmask(self, path: Path, top_n: int | None = None) -> None:
        """Write masks to *path* in ``.hcmask`` format.

        Each line is: ``mask`` (optionally with count comment).
        """
        masks = self.ranked(top_n=top_n)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for m in masks:
                fh.write(m.mask + "\n")
        logger.info("Exported %d masks to %s", len(masks), path)

    def export_json(self, path: Path, top_n: int | None = None) -> None:
        """Write mask statistics to *path* as JSON."""
        data = [m.to_dict() for m in self.ranked(top_n=top_n)]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Mask JSON exported to %s", path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "total_passwords": self._total_passwords,
            "unique_masks": len(self._mask_stats),
            "top_5": [m.to_dict() for m in self.ranked(5)],
        }
