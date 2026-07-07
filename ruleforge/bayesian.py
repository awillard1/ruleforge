"""
ruleforge/bayesian.py
---------------------
Bayesian Optimization for unexplored Hashcat rule-space regions.

Uses a Gaussian Process surrogate model to balance exploration vs.
exploitation via the Upper Confidence Bound (UCB) acquisition function.

This implementation encodes rules as fixed-length feature vectors
(operation frequency histograms) to keep dependencies light.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parser import Parser, ALL_OPS
from .generator import RuleGenerator

logger = logging.getLogger(__name__)

# Sorted list of known operations for stable feature encoding
_OP_INDEX: dict[str, int] = {op: i for i, op in enumerate(sorted(ALL_OPS))}
_OP_DIM: int = len(_OP_INDEX)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _rule_to_features(rule: str, parser: Parser) -> list[float] | None:
    """Encode a rule as a normalized operation-frequency histogram.

    Returns a list of *_OP_DIM* floats, or ``None`` if the rule is invalid.
    """
    toks = parser.try_parse(rule)
    if not toks:
        return None
    vec = [0.0] * _OP_DIM
    for t in toks:
        idx = _OP_INDEX.get(t.cmd)
        if idx is not None:
            vec[idx] += 1.0
    n = len(toks)
    return [v / n for v in vec]


# ---------------------------------------------------------------------------
# Lightweight RBF kernel GP
# ---------------------------------------------------------------------------

def _rbf_kernel(x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
    """Squared-exponential (RBF) kernel k(x1, x2)."""
    sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return math.exp(-sq_dist / (2.0 * length_scale ** 2))


class _GaussianProcess:
    """Minimal GP regression using a Cholesky-free inverse."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6) -> None:
        self._ls = length_scale
        self._noise = noise
        self._X: list[list[float]] = []
        self._y: list[float] = []
        # Precomputed K_inv for observed data
        self._K_inv: list[list[float]] | None = None

    def _kernel_matrix(
        self, X: list[list[float]], noise: bool = True
    ) -> list[list[float]]:
        n = len(X)
        K: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                K[i][j] = _rbf_kernel(X[i], X[j], self._ls)
            if noise:
                K[i][i] += self._noise
        return K

    @staticmethod
    def _mat_inv(M: list[list[float]]) -> list[list[float]] | None:
        """Gauss-Jordan inversion of square matrix M (in-place copy)."""
        n = len(M)
        A = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(M)]
        for col in range(n):
            pivot = None
            for row in range(col, n):
                if abs(A[row][col]) > 1e-12:
                    pivot = row
                    break
            if pivot is None:
                return None
            A[col], A[pivot] = A[pivot], A[col]
            scale = A[col][col]
            A[col] = [v / scale for v in A[col]]
            for row in range(n):
                if row == col:
                    continue
                factor = A[row][col]
                A[row] = [A[row][k] - factor * A[col][k] for k in range(2 * n)]
        return [[A[i][n + j] for j in range(n)] for i in range(n)]

    def _mat_vec(self, M: list[list[float]], v: list[float]) -> list[float]:
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        self._X = X
        self._y = y
        if X:
            K = self._kernel_matrix(X)
            self._K_inv = self._mat_inv(K)

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Return (mean, std) prediction for query point *x*."""
        if not self._X or self._K_inv is None:
            return 0.0, 1.0  # Prior

        k_star = [_rbf_kernel(x, xi, self._ls) for xi in self._X]
        mean = sum(k_star[i] * (self._mat_vec(self._K_inv, self._y)[i])
                   for i in range(len(self._X)))

        k_ss = _rbf_kernel(x, x, self._ls)
        Kinv_k = self._mat_vec(self._K_inv, k_star)
        variance = k_ss - sum(k_star[i] * Kinv_k[i] for i in range(len(k_star)))
        variance = max(variance, 0.0)
        return mean, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Bayesian Optimizer
# ---------------------------------------------------------------------------


@dataclass
class BayesianConfig:
    """Configuration for the Bayesian optimizer."""

    n_initial: int = 20          # random seed points before BO starts
    n_iterations: int = 100      # total BO iterations (random + GP-guided)
    kappa: float = 2.0           # UCB exploration-exploitation trade-off
    gp_length_scale: float = 1.0
    gp_noise: float = 1e-6
    batch_size_per_iter: int = 50  # candidates to evaluate per iteration
    random_ratio: float = 0.30    # fraction of iterations using random exploration


class BayesianOptimizer:
    """Bayesian optimizer over the Hashcat rule space.

    Uses UCB acquisition to direct the search toward
    high-fitness unexplored regions.

    Args:
        parser:     A :class:`~ruleforge.parser.Parser` instance.
        generator:  A :class:`~ruleforge.generator.RuleGenerator` for
                    candidate generation.
        fitness_fn: Callable mapping a rule string to a float score.
        config:     :class:`BayesianConfig`.
        rng:        Random source.
    """

    def __init__(
        self,
        parser: Parser,
        generator: RuleGenerator,
        fitness_fn: Any,
        config: BayesianConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._parser = parser
        self._gen = generator
        self._fitness_fn = fitness_fn
        self._cfg = config or BayesianConfig()
        self._rng = rng or random.Random()

        self._gp = _GaussianProcess(
            length_scale=self._cfg.gp_length_scale,
            noise=self._cfg.gp_noise,
        )
        self._observed_X: list[list[float]] = []
        self._observed_y: list[float] = []
        self._observed_rules: list[str] = []

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _ucb(self, x: list[float]) -> float:
        mean, std = self._gp.predict(x)
        return mean + self._cfg.kappa * std

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(self, n: int) -> list[tuple[str, list[float]]]:
        seen = set(self._observed_rules)
        out: list[tuple[str, list[float]]] = []
        tries = 0
        while len(out) < n and tries < n * 10:
            tries += 1
            rule = self._gen.generate_one()
            if rule in seen:
                continue
            feat = _rule_to_features(rule, self._parser)
            if feat is None:
                continue
            out.append((rule, feat))
        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[tuple[str, float]]:
        """Run Bayesian optimization.

        Returns:
            List of ``(rule, fitness)`` tuples sorted by fitness descending.
        """
        cfg = self._cfg

        # Phase 1: random initialization
        init_candidates = self._generate_candidates(cfg.n_initial)
        for rule, feat in init_candidates:
            score = self._fitness_fn(rule)
            self._observed_X.append(feat)
            self._observed_y.append(score)
            self._observed_rules.append(rule)

        logger.info(
            "Bayesian init: %d points, best=%.4f",
            len(self._observed_y),
            max(self._observed_y, default=0.0),
        )

        # Phase 2: GP-guided iterations
        for iteration in range(cfg.n_iterations):
            self._gp.fit(self._observed_X, self._observed_y)

            if self._rng.random() < cfg.random_ratio:
                # Pure random exploration
                candidates = self._generate_candidates(cfg.batch_size_per_iter)
            else:
                # UCB-guided: pick best from random pool
                pool = self._generate_candidates(cfg.batch_size_per_iter * 3)
                pool.sort(key=lambda rc: self._ucb(rc[1]), reverse=True)
                candidates = pool[: cfg.batch_size_per_iter]

            for rule, feat in candidates:
                score = self._fitness_fn(rule)
                self._observed_X.append(feat)
                self._observed_y.append(score)
                self._observed_rules.append(rule)

            if iteration % 10 == 0:
                best_so_far = max(self._observed_y, default=0.0)
                logger.debug("BO iteration %d: best=%.4f", iteration, best_so_far)

        # Return sorted results
        paired = list(zip(self._observed_rules, self._observed_y))
        paired.sort(key=lambda rv: rv[1], reverse=True)
        return paired

    def best_rules(self, n: int = 100) -> list[str]:
        """Return the top *n* rules found so far."""
        paired = sorted(
            zip(self._observed_rules, self._observed_y),
            key=lambda rv: rv[1],
            reverse=True,
        )
        return [r for r, _ in paired[:n]]

    def save(self, path: Path) -> None:
        data = {
            "observed_rules": self._observed_rules,
            "observed_y": self._observed_y,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self._observed_rules = list(data["observed_rules"])
        self._observed_y = list(data["observed_y"])
        self._observed_X = [
            _rule_to_features(r, self._parser) or []
            for r in self._observed_rules
        ]
