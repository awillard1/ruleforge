"""
ruleforge/markov.py
-------------------
Higher-Order Markov Model for Hashcat rule operation sequences.

Supports first, second, third, and variable-order models with optional
interpolation (Jelinek-Mercer style).

The model treats each Hashcat rule as a sequence of operation characters
and learns transition probabilities from a corpus of rules.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Sentinel tokens for start/end of sequence
BOS = "<s>"   # beginning of sequence
EOS = "</s>"  # end of sequence


# ---------------------------------------------------------------------------
# N-gram counter helpers
# ---------------------------------------------------------------------------

def _ngram_key(context: Sequence[str], token: str) -> str:
    """Combine context and token into a compact dict key."""
    return "\x00".join(list(context) + [token])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class MarkovModel:
    """Trained higher-order Markov model.

    Attributes:
        order:   Maximum context length (1 = bigram, 2 = trigram, …).
        counts:  Raw n-gram counts: ``{context_tuple → Counter(next_token)}``.
        vocab:   Set of all observed tokens (operation characters).
    """

    order: int
    counts: dict[tuple[str, ...], Counter[str]] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "vocab": sorted(self.vocab),
            "counts": {
                "|".join(ctx): dict(cntr)
                for ctx, cntr in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MarkovModel":
        counts: dict[tuple[str, ...], Counter[str]] = {}
        for key, cntr in d.get("counts", {}).items():
            ctx = tuple(key.split("|")) if key else ()
            counts[ctx] = Counter(cntr)
        return cls(
            order=int(d["order"]),
            counts=counts,
            vocab=set(d.get("vocab", [])),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MarkovTrainer:
    """Build a :class:`MarkovModel` from rule sequences.

    Args:
        order:       Context window size (1–N).  Defaults to 1.
    """

    def __init__(self, order: int = 1) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self._order = order
        self._counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._vocab: set[str] = set()

    def train(self, sequences: list[list[str]]) -> None:
        """Train on a list of operation sequences."""
        for seq in sequences:
            self._add_sequence(seq)

    def train_rules(self, rules: list[str], parser: Any) -> None:
        """Train directly from a list of rule strings."""
        sequences = []
        for rule in rules:
            toks = parser.try_parse(rule)
            if toks:
                sequences.append([t.cmd for t in toks])
        self.train(sequences)

    def _add_sequence(self, ops: list[str]) -> None:
        padded = [BOS] * self._order + ops + [EOS]
        for token in ops + [EOS]:
            self._vocab.add(token)
        for i in range(self._order, len(padded)):
            ctx = tuple(padded[i - self._order : i])
            token = padded[i]
            self._counts[ctx][token] += 1

    def build(self) -> MarkovModel:
        """Return the trained :class:`MarkovModel`."""
        return MarkovModel(
            order=self._order,
            counts=dict(self._counts),
            vocab=set(self._vocab),
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class MarkovSampler:
    """Generate operation sequences from a trained :class:`MarkovModel`.

    Supports:
    - First-, second-, third-, variable-order generation.
    - Jelinek-Mercer interpolation with lower-order models for back-off.

    Args:
        model:           The primary (highest-order) model.
        lower_models:    Lower-order models for interpolation (lowest first).
        lambdas:         Interpolation weights (must sum to ≤ 1).
                         If not supplied, equal weights are used.
        rng:             Random source.
    """

    def __init__(
        self,
        model: MarkovModel,
        lower_models: list[MarkovModel] | None = None,
        lambdas: list[float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._model = model
        self._lower = lower_models or []
        self._rng = rng or random.Random()

        all_models = self._lower + [model]
        if lambdas is not None:
            if len(lambdas) != len(all_models):
                raise ValueError(
                    f"len(lambdas)={len(lambdas)} must equal number of models "
                    f"{len(all_models)}"
                )
            self._lambdas = lambdas
        else:
            w = 1.0 / len(all_models)
            self._lambdas = [w] * len(all_models)

    def _prob(
        self,
        context: tuple[str, ...],
        token: str,
        model: MarkovModel,
    ) -> float:
        cntr = model.counts.get(context, Counter())
        total = sum(cntr.values())
        if total == 0:
            return 0.0
        return cntr.get(token, 0) / total

    def _interpolated_prob(self, context: tuple[str, ...], token: str) -> float:
        """Jelinek-Mercer interpolated probability."""
        all_models = self._lower + [self._model]
        p = 0.0
        for lam, mdl in zip(self._lambdas, all_models):
            # Shorten context to match model order
            ctx = context[-mdl.order :] if len(context) >= mdl.order else context
            p += lam * self._prob(ctx, token, mdl)
        return p

    def _next_token(self, context: tuple[str, ...]) -> str | None:
        vocab = self._model.vocab | {EOS}
        candidates = list(vocab)
        weights = [self._interpolated_prob(context, t) for t in candidates]
        total = sum(weights)
        if total <= 0:
            # Uniform fallback
            return self._rng.choice(candidates)
        r = self._rng.uniform(0, total)
        acc = 0.0
        for tok, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                return tok
        return candidates[-1]

    def sample(
        self,
        max_len: int = 12,
        min_len: int = 1,
    ) -> list[str]:
        """Sample an operation sequence from the model.

        Args:
            max_len: Maximum number of operations (not counting EOS).
            min_len: Minimum sequence length (retries if shorter).

        Returns:
            List of operation character strings.
        """
        order = self._model.order
        context: list[str] = [BOS] * order
        ops: list[str] = []

        for _ in range(max_len):
            ctx = tuple(context[-order:])
            token = self._next_token(ctx)
            if token is None or token == EOS:
                break
            ops.append(token)
            context.append(token)

        return ops if len(ops) >= min_len else []

    def sample_many(self, n: int, max_len: int = 12) -> list[list[str]]:
        """Sample *n* sequences."""
        return [self.sample(max_len=max_len) for _ in range(n)]


# ---------------------------------------------------------------------------
# Variable-order model
# ---------------------------------------------------------------------------


class VariableOrderMarkov:
    """Variable-order Markov model combining models of different orders.

    Automatically trains models of orders 1 through *max_order* and uses
    Jelinek-Mercer interpolation during sampling.

    Args:
        max_order: Highest order to train (e.g. 3 for 1st+2nd+3rd).
        lambdas:   Interpolation weights (list of *max_order* floats).
                   Defaults to equal weighting.
    """

    def __init__(
        self,
        max_order: int = 3,
        lambdas: list[float] | None = None,
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self._max_order = max_order
        self._lambdas = lambdas
        self._models: list[MarkovModel] = []
        self._sampler: MarkovSampler | None = None

    def train(self, rules: list[str], parser: Any) -> None:
        """Train all sub-models from *rules*."""
        trainers = [MarkovTrainer(order=o) for o in range(1, self._max_order + 1)]
        for trainer in trainers:
            trainer.train_rules(rules, parser)
        self._models = [t.build() for t in trainers]
        self._sampler = MarkovSampler(
            model=self._models[-1],
            lower_models=self._models[:-1],
            lambdas=self._lambdas,
        )
        logger.info(
            "Trained variable-order Markov (orders 1–%d) on %d rules",
            self._max_order,
            len(rules),
        )

    def sample(self, max_len: int = 12) -> list[str]:
        """Sample a single operation sequence."""
        if self._sampler is None:
            raise RuntimeError("Call train() before sample()")
        return self._sampler.sample(max_len=max_len)

    def sample_many(self, n: int, max_len: int = 12) -> list[list[str]]:
        if self._sampler is None:
            raise RuntimeError("Call train() before sample_many()")
        return self._sampler.sample_many(n=n, max_len=max_len)

    def save(self, path: Path) -> None:
        """Save all sub-models to *path* (JSON)."""
        data = [m.to_dict() for m in self._models]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load sub-models from *path* (JSON)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._models = [MarkovModel.from_dict(d) for d in data]
        self._sampler = MarkovSampler(
            model=self._models[-1],
            lower_models=self._models[:-1],
            lambdas=self._lambdas,
        )
