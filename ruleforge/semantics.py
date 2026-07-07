"""
ruleforge/semantics.py
----------------------
Semantic Analyzer — automatically categorize words from password corpora.

Uses:
- WordNet (via NLTK, optional)
- Public dictionaries (bundled word lists)
- GeoNames-style word lists (optional)
- IMDB / MusicBrainz lists (optional)
- Plugin-supplied categorizers

Falls back to heuristic pattern matching when external resources are unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

class Category:
    WORD = "word"
    NAME = "name"
    CITY = "city"
    COUNTRY = "country"
    SPORTS_TEAM = "sports_team"
    MOVIE = "movie"
    GAME = "game"
    ANIME = "anime"
    MUSIC = "music"
    BRAND = "brand"
    NUMBER = "number"
    YEAR = "year"
    KEYBOARD = "keyboard"
    LEET = "leet"
    PATTERN = "pattern"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Heuristic classifiers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")
_ALL_DIGITS_RE = re.compile(r"^\d+$")
_KEYBOARD_RE = re.compile(
    r"qwerty|asdf|zxcv|qwer|1234|4321|abcdef", re.IGNORECASE
)
_LEET_MAP = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}


def _classify_heuristic(word: str) -> str:
    if _YEAR_RE.match(word):
        return Category.YEAR
    if _ALL_DIGITS_RE.match(word):
        return Category.NUMBER
    if _KEYBOARD_RE.search(word):
        return Category.KEYBOARD
    leet_decoded = "".join(_LEET_MAP.get(c, c) for c in word.lower())
    if leet_decoded != word.lower() and leet_decoded.isalpha():
        return Category.LEET
    if word.istitle() and len(word) >= 4:
        return Category.NAME
    if word.isalpha():
        return Category.WORD
    return Category.OTHER


# ---------------------------------------------------------------------------
# WordNet integration (optional)
# ---------------------------------------------------------------------------

def _classify_wordnet(word: str) -> str | None:
    """Try to classify *word* using NLTK WordNet.

    Returns ``None`` if NLTK is not installed or lookup fails.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import]
        synsets = wn.synsets(word.lower())
        if not synsets:
            return None
        # Use the lexname of the most common synset
        lex = synsets[0].lexname()
        if "person" in lex or "name" in lex:
            return Category.NAME
        if "location" in lex or "place" in lex:
            return Category.CITY
        return Category.WORD
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# SemanticAnalyzer
# ---------------------------------------------------------------------------

CategorizerFn = Callable[[str], str | None]


@dataclass
class CategoryResult:
    """Category assignment for a single word."""

    word: str
    category: str
    source: str  # e.g. "wordnet", "heuristic", "plugin"

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "category": self.category, "source": self.source}


class SemanticAnalyzer:
    """Categorize words from a password corpus.

    Uses a pipeline of categorizers:
    1. Plugin-supplied categorizers (if registered).
    2. WordNet (if NLTK is available).
    3. Heuristic fallback.

    Args:
        use_wordnet:   Enable WordNet lookups (requires NLTK).
        word_lists:    Optional ``{category: set_of_words}`` mapping for
                       fast exact-match lookup.
    """

    def __init__(
        self,
        use_wordnet: bool = True,
        word_lists: dict[str, set[str]] | None = None,
    ) -> None:
        self._use_wordnet = use_wordnet
        self._word_lists: dict[str, set[str]] = word_lists or {}
        self._plugins: list[tuple[str, CategorizerFn]] = []

        # Results cache
        self._cache: dict[str, CategoryResult] = {}

    # ------------------------------------------------------------------
    # Word list loading
    # ------------------------------------------------------------------

    def load_word_list(self, category: str, path: Path) -> None:
        """Load a plain-text word list (one word per line) for *category*."""
        words: set[str] = set()
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                w = line.strip()
                if w:
                    words.add(w.lower())
        if category not in self._word_lists:
            self._word_lists[category] = set()
        self._word_lists[category] |= words
        logger.info("Loaded %d words for category %r from %s", len(words), category, path)

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def register_plugin(self, name: str, fn: CategorizerFn) -> None:
        """Register a custom categorizer plugin.

        *fn* should accept a word string and return a category string or
        ``None`` to fall through to the next categorizer.
        """
        self._plugins.append((name, fn))
        logger.debug("Registered semantic plugin: %s", name)

    # ------------------------------------------------------------------
    # Classification pipeline
    # ------------------------------------------------------------------

    def classify(self, word: str) -> CategoryResult:
        """Classify a single *word*."""
        if word in self._cache:
            return self._cache[word]

        lower = word.lower()

        # 1. Exact word-list match
        for category, words in self._word_lists.items():
            if lower in words:
                result = CategoryResult(word=word, category=category, source="word_list")
                self._cache[word] = result
                return result

        # 2. Plugin pipeline
        for plugin_name, fn in self._plugins:
            cat = fn(word)
            if cat is not None:
                result = CategoryResult(word=word, category=cat, source=plugin_name)
                self._cache[word] = result
                return result

        # 3. WordNet
        if self._use_wordnet:
            cat = _classify_wordnet(word)
            if cat is not None:
                result = CategoryResult(word=word, category=cat, source="wordnet")
                self._cache[word] = result
                return result

        # 4. Heuristic fallback
        cat = _classify_heuristic(word)
        result = CategoryResult(word=word, category=cat, source="heuristic")
        self._cache[word] = result
        return result

    def classify_many(self, words: list[str]) -> list[CategoryResult]:
        """Classify a list of words."""
        return [self.classify(w) for w in words]

    # ------------------------------------------------------------------
    # Corpus analysis
    # ------------------------------------------------------------------

    def analyze_passwords(self, passwords: list[str]) -> dict[str, list[str]]:
        """Classify all words extracted from *passwords*.

        Returns ``{category → [words]}``.
        """
        categories: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        for pw in passwords:
            # Split password into alphabetic tokens
            for tok in re.findall(r"[a-zA-Z]+", pw):
                if tok.lower() not in seen and len(tok) >= 3:
                    seen.add(tok.lower())
                    r = self.classify(tok)
                    categories[r.category].append(tok)
        return dict(categories)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Write all cached classifications to *path*."""
        data = [r.to_dict() for r in self._cache.values()]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        cat_counts: Counter[str] = Counter()
        for r in self._cache.values():
            cat_counts[r.category] += 1
        return {
            "total_classified": len(self._cache),
            "by_category": dict(cat_counts),
            "plugins": [name for name, _ in self._plugins],
        }
