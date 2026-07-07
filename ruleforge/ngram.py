"""
ruleforge/ngram.py
------------------
N-Gram Engine for operation sequences.

Implements:
- Bigram (2-gram)
- Trigram (3-gram)
- 4-gram
- Back-off smoothing (Katz-style)
- Add-one / Laplace smoothing
- Kneser-Ney smoothing (full, not modified)
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

BOS = "<s>"
EOS = "</s>"


# ---------------------------------------------------------------------------
# Core NGram counter
# ---------------------------------------------------------------------------


class NGramCounter:
    """Count n-grams up to order *n* over tokenized sequences.

    Args:
        n: Maximum n-gram order (2 for bigram, 3 for trigram, etc.).
    """

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        # counts[order] = Counter(tuple_of_tokens → int)
        self._counts: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(n + 1)]
        self._vocab: set[str] = set()

    @property
    def n(self) -> int:
        return self._n

    @property
    def vocab(self) -> frozenset[str]:
        return frozenset(self._vocab)

    def add_sequence(self, tokens: list[str]) -> None:
        """Add a token sequence to the counts."""
        padded = [BOS] * (self._n - 1) + tokens + [EOS]
        for token in tokens + [EOS]:
            self._vocab.add(token)
        for order in range(1, self._n + 1):
            for i in range(order - 1, len(padded)):
                gram = tuple(padded[i - order + 1 : i + 1])
                self._counts[order][gram] += 1

    def count(self, gram: tuple[str, ...]) -> int:
        """Return the count for a specific n-gram."""
        order = len(gram)
        if order > self._n:
            return 0
        return self._counts[order].get(gram, 0)

    def total_unigrams(self) -> int:
        return sum(self._counts[1].values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self._n,
            "vocab": sorted(self._vocab),
            "counts": [
                {"|".join(k): v for k, v in cntr.items()}
                for cntr in self._counts
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NGramCounter":
        obj = cls(n=int(d["n"]))
        obj._vocab = set(d.get("vocab", []))
        for order, raw in enumerate(d.get("counts", [])):
            cntr: Counter[tuple[str, ...]] = Counter()
            for key, val in raw.items():
                cntr[tuple(key.split("|"))] = int(val)
            obj._counts[order] = cntr
        return obj


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------


class LaplaceNGram:
    """N-gram language model with Laplace (add-one) smoothing.

    Args:
        counter: A trained :class:`NGramCounter`.
        alpha:   Smoothing constant (default 1.0 = Laplace).
    """

    def __init__(self, counter: NGramCounter, alpha: float = 1.0) -> None:
        self._counter = counter
        self._alpha = alpha

    def prob(self, context: tuple[str, ...], token: str) -> float:
        """Smoothed probability P(token | context)."""
        n = len(context) + 1
        gram = context + (token,)
        numerator = self._counter.count(gram) + self._alpha
        # Denominator: sum over vocab of count(context, v) + alpha * |V|
        V = len(self._counter.vocab) + 1  # +1 for EOS
        ctx_total = sum(
            self._counter.count(context + (v,)) for v in self._counter.vocab | {EOS}
        )
        denominator = ctx_total + self._alpha * V
        return numerator / denominator if denominator > 0 else 1.0 / V

    def log_prob(self, sequence: list[str]) -> float:
        """Log-probability of a full sequence."""
        n = self._counter.n
        padded = [BOS] * (n - 1) + sequence + [EOS]
        lp = 0.0
        for i in range(n - 1, len(padded)):
            ctx = tuple(padded[i - n + 1 : i])
            tok = padded[i]
            p = self.prob(ctx, tok)
            lp += math.log(p) if p > 0 else -1e9
        return lp


class BackoffNGram:
    """Katz back-off n-gram model.

    Falls back to lower-order models when higher-order counts are zero,
    with a constant back-off weight *beta*.

    Args:
        counter: Trained :class:`NGramCounter`.
        beta:    Back-off penalty multiplier (default 0.75).
    """

    def __init__(self, counter: NGramCounter, beta: float = 0.75) -> None:
        self._counter = counter
        self._beta = beta

    def prob(self, context: tuple[str, ...], token: str) -> float:
        """Back-off probability P(token | context)."""
        if not context:
            # Unigram fallback
            total = self._counter.total_unigrams()
            count = self._counter.count((token,))
            return max(count, 1) / max(total, 1)

        gram = context + (token,)
        cnt = self._counter.count(gram)
        if cnt > 0:
            ctx_total = sum(
                self._counter.count(context + (v,))
                for v in self._counter.vocab | {EOS}
            )
            return cnt / ctx_total if ctx_total > 0 else 0.0

        # Back off
        return self._beta * self.prob(context[1:], token)

    def log_prob(self, sequence: list[str]) -> float:
        n = self._counter.n
        padded = [BOS] * (n - 1) + sequence + [EOS]
        lp = 0.0
        for i in range(n - 1, len(padded)):
            ctx = tuple(padded[i - n + 1 : i])
            tok = padded[i]
            p = self.prob(ctx, tok)
            lp += math.log(p) if p > 0 else -1e9
        return lp


class KneserNeyNGram:
    """Full (non-modified) Kneser-Ney smoothed n-gram model.

    Args:
        counter:  Trained :class:`NGramCounter`.
        discount: Absolute discount parameter D (default 0.75).
    """

    def __init__(self, counter: NGramCounter, discount: float = 0.75) -> None:
        self._counter = counter
        self._D = discount
        # Precompute continuation counts for unigrams (KN lower-order)
        self._continuation: Counter[str] = Counter()
        self._precompute_continuation()

    def _precompute_continuation(self) -> None:
        """Count how many unique bigram left-contexts each token follows."""
        n = self._counter.n
        if n < 2:
            return
        seen: dict[str, set[str]] = defaultdict(set)
        for gram, cnt in self._counter._counts[2].items():
            if cnt > 0:
                left, right = gram[0], gram[1]
                seen[right].add(left)
        for tok, contexts in seen.items():
            self._continuation[tok] = len(contexts)

    def _kn_unigram(self, token: str) -> float:
        total = sum(self._continuation.values())
        return self._continuation.get(token, 0) / max(total, 1)

    def prob(self, context: tuple[str, ...], token: str) -> float:
        """KN probability P(token | context)."""
        if not context:
            return self._kn_unigram(token)

        gram = context + (token,)
        cnt = self._counter.count(gram)
        disc = max(cnt - self._D, 0)

        ctx_total = sum(
            self._counter.count(context + (v,))
            for v in self._counter.vocab | {EOS}
        )
        if ctx_total == 0:
            return self.prob(context[1:], token)

        # Number of unique tokens following this context
        lambda_w = (
            self._D
            * sum(
                1
                for v in self._counter.vocab | {EOS}
                if self._counter.count(context + (v,)) > 0
            )
            / ctx_total
        )
        return disc / ctx_total + lambda_w * self.prob(context[1:], token)

    def log_prob(self, sequence: list[str]) -> float:
        n = self._counter.n
        padded = [BOS] * (n - 1) + sequence + [EOS]
        lp = 0.0
        for i in range(n - 1, len(padded)):
            ctx = tuple(padded[i - n + 1 : i])
            tok = padded[i]
            p = self.prob(ctx, tok)
            lp += math.log(p) if p > 0 else -1e9
        return lp


# ---------------------------------------------------------------------------
# High-level NGramEngine
# ---------------------------------------------------------------------------


class NGramEngine:
    """Unified N-gram engine supporting bigram, trigram and 4-gram models.

    Args:
        n:           Maximum n-gram order (default 3).
        smoothing:   ``"laplace"``, ``"backoff"``, or ``"kneser_ney"``.
        alpha:       Laplace smoothing constant (used when smoothing="laplace").
        discount:    KN/back-off discount (used for other smoothing types).
    """

    SMOOTHING_OPTIONS = ("laplace", "backoff", "kneser_ney")

    def __init__(
        self,
        n: int = 3,
        smoothing: str = "backoff",
        alpha: float = 1.0,
        discount: float = 0.75,
    ) -> None:
        if smoothing not in self.SMOOTHING_OPTIONS:
            raise ValueError(
                f"smoothing must be one of {self.SMOOTHING_OPTIONS}, got {smoothing!r}"
            )
        self._n = n
        self._smoothing = smoothing
        self._alpha = alpha
        self._discount = discount
        self._counter: NGramCounter | None = None
        self._model: LaplaceNGram | BackoffNGram | KneserNeyNGram | None = None

    def train(self, rules: list[str], parser: Any) -> None:
        """Train the engine from *rules*."""
        counter = NGramCounter(n=self._n)
        for rule in rules:
            toks = parser.try_parse(rule)
            if toks:
                counter.add_sequence([t.cmd for t in toks])
        self._counter = counter

        if self._smoothing == "laplace":
            self._model = LaplaceNGram(counter, alpha=self._alpha)
        elif self._smoothing == "backoff":
            self._model = BackoffNGram(counter, beta=self._discount)
        else:
            self._model = KneserNeyNGram(counter, discount=self._discount)

        logger.info(
            "Trained %d-gram engine (%s) on %d rules; vocab size=%d",
            self._n,
            self._smoothing,
            len(rules),
            len(counter.vocab),
        )

    def log_prob(self, ops: list[str]) -> float:
        """Return log-probability of an operation sequence."""
        if self._model is None:
            raise RuntimeError("Call train() first")
        return self._model.log_prob(ops)

    def score_rule(self, rule: str, parser: Any) -> float:
        """Score a rule string; returns log-prob or -inf on invalid input."""
        toks = parser.try_parse(rule)
        if not toks:
            return float("-inf")
        return self.log_prob([t.cmd for t in toks])

    def vocab(self) -> frozenset[str]:
        if self._counter is None:
            return frozenset()
        return self._counter.vocab

    def save(self, path: Path) -> None:
        """Persist counter to JSON at *path*."""
        if self._counter is None:
            raise RuntimeError("No trained model to save")
        data = {"smoothing": self._smoothing, "alpha": self._alpha,
                "discount": self._discount, "counter": self._counter.to_dict()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Restore counter from JSON at *path* and rebuild model."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._smoothing = data["smoothing"]
        self._alpha = float(data.get("alpha", 1.0))
        self._discount = float(data.get("discount", 0.75))
        counter = NGramCounter.from_dict(data["counter"])
        self._counter = counter
        self._n = counter.n
        if self._smoothing == "laplace":
            self._model = LaplaceNGram(counter, alpha=self._alpha)
        elif self._smoothing == "backoff":
            self._model = BackoffNGram(counter, beta=self._discount)
        else:
            self._model = KneserNeyNGram(counter, discount=self._discount)
