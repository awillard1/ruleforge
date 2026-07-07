"""
ruleforge/generator.py
----------------------
Probabilistic Rule Generator.

Combines Markov chains, N-gram models, templates, and random exploration
in a configurable weighted mixture to produce candidate Hashcat rules.

Compatible with the legacy parallel worker payload format from hashcat_rules.py.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parser import Parser, Token, _arity, _ok_char, MAX_OPS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation strategy weights
# ---------------------------------------------------------------------------

@dataclass
class MixtureWeights:
    """Configurable weights for the generator mixture."""

    markov: float = 0.50
    ngram: float = 0.00      # reserved; requires NGramEngine
    template: float = 0.10
    random_explore: float = 0.05
    mutate: float = 0.35

    def normalize(self) -> "MixtureWeights":
        total = self.markov + self.ngram + self.template + self.random_explore + self.mutate
        if total <= 0:
            raise ValueError("All weights are zero")
        return MixtureWeights(
            markov=self.markov / total,
            ngram=self.ngram / total,
            template=self.template / total,
            random_explore=self.random_explore / total,
            mutate=self.mutate / total,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wchoice(counter: Counter[str], rng: random.Random, fallback: str | None = None) -> str | None:
    """Weighted random choice from a Counter."""
    if not counter:
        return fallback
    items = list(counter.items())
    total = sum(v for _, v in items)
    if total <= 0:
        return fallback
    r = rng.uniform(0, total)
    acc = 0.0
    for k, w in items:
        acc += w
        if acc >= r:
            return k
    return items[-1][0]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


class RuleGenerator:
    """Generate candidate Hashcat rules using a configurable mixture.

    Args:
        parser:              A :class:`~ruleforge.parser.Parser` instance.
        source_rules:        Known-valid rules to learn from and mutate.
        cmd_freq:            Counter of operation frequencies.
        start_freq:          Counter of first-operation frequencies.
        end_freq:            Counter of last-operation frequencies.
        trans:               Transition table ``{cmd → Counter(next_cmd)}``.
        param_freq:          Parameter frequencies ``{cmd → Counter(param)}``.
        len_dist:            Rule length distribution ``Counter(int → count)``.
        weights:             :class:`MixtureWeights` for strategy selection.
        allow_param_fallback: Fall back to generic params when learned pool is empty.
        rng:                 Random source.
    """

    _SAFE1: list[str] = list("0123456789!@#$%^&*.-_")
    _SAFE_POS: list[str] = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(
        self,
        parser: Parser,
        source_rules: list[str],
        cmd_freq: Counter[str],
        start_freq: Counter[str],
        end_freq: Counter[str],
        trans: dict[str, Counter[str]],
        param_freq: dict[str, Counter[str]],
        len_dist: Counter[int],
        weights: MixtureWeights | None = None,
        allow_param_fallback: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        self._parser = parser
        self._source = list(source_rules)
        self._source_set: set[str] = set(source_rules)

        self._cmd = cmd_freq
        self._start = start_freq
        self._end = end_freq
        self._trans: dict[str, Counter[str]] = trans
        self._params = param_freq
        self._len_dist = len_dist

        self._weights = (weights or MixtureWeights()).normalize()
        self._allow_fallback = allow_param_fallback
        self._rng = rng or random.Random()

        self._allowed_cmds: set[str] = set(cmd_freq.keys())

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_param(self, cmd: str) -> str | None:
        ar = _arity(cmd)
        if ar == 0:
            return ""
        pool = self._params.get(cmd)
        if pool:
            p = _wchoice(pool, self._rng)
            if p is not None and len(p) == ar and all(_ok_char(c) for c in p):
                return p
        if not self._allow_fallback:
            return None
        if ar == 1:
            return self._rng.choice(self._SAFE1)
        if ar == 2:
            return self._rng.choice(self._SAFE1) + self._rng.choice(self._SAFE1)
        if ar == 3:
            return (
                self._rng.choice(self._SAFE_POS)
                + self._rng.choice(self._SAFE1)
                + self._rng.choice(self._SAFE1)
            )
        return None

    # ------------------------------------------------------------------
    # Length sampling
    # ------------------------------------------------------------------

    def _sample_len(self, max_ops: int) -> int:
        n = _wchoice(Counter({str(k): v for k, v in self._len_dist.items()}), self._rng)
        try:
            return max(1, min(max_ops, int(n or 4)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 4

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def _valid(self, rule: str, max_ops: int = MAX_OPS) -> bool:
        if not self._parser.validate(rule, max_ops=max_ops):
            return False
        toks = self._parser.try_parse(rule)
        if not toks:
            return False
        for t in toks:
            if t.cmd not in self._allowed_cmds:
                return False
            if _arity(t.cmd) < 0 or len(t.param) != _arity(t.cmd):
                return False
            if any(not _ok_char(c) for c in t.param):
                return False
        return True

    # ------------------------------------------------------------------
    # Markov generation
    # ------------------------------------------------------------------

    def _build_markov(self, max_ops: int = 12) -> str:
        n = self._sample_len(max_ops)
        start = _wchoice(self._start, self._rng, fallback=":")
        if start not in self._allowed_cmds:
            return ""
        cmds: list[str] = [start]
        while len(cmds) < n:
            c = _wchoice(
                self._trans.get(cmds[-1], Counter()),
                self._rng,
                fallback=_wchoice(self._cmd, self._rng, fallback=":"),
            )
            if c and c in self._allowed_cmds:
                cmds.append(c)
            else:
                break

        toks: list[Token] = []
        for c in cmds:
            p = self._sample_param(c)
            if p is None:
                return ""
            toks.append(Token(c, p))
        rule = self._parser.serialize(toks)
        return rule if self._valid(rule) else ""

    # ------------------------------------------------------------------
    # Template-based generation
    # ------------------------------------------------------------------

    def _build_from_template(self, max_ops: int = 12) -> str:
        """Pick a random source rule as a template skeleton and re-parameterize."""
        if not self._source:
            return self._build_markov(max_ops)
        base = self._rng.choice(self._source)
        toks = self._parser.try_parse(base)
        if not toks:
            return ""
        new_toks: list[Token] = []
        for t in toks:
            p = self._sample_param(t.cmd)
            if p is None:
                return ""
            new_toks.append(Token(t.cmd, p))
        rule = self._parser.serialize(new_toks)
        return rule if self._valid(rule, max_ops=max_ops) else ""

    # ------------------------------------------------------------------
    # Random exploration
    # ------------------------------------------------------------------

    def _build_random(self, max_ops: int = 12) -> str:
        n = self._rng.randint(1, min(6, max_ops))
        toks: list[Token] = []
        cmds = list(self._allowed_cmds)
        for _ in range(n):
            c = self._rng.choice(cmds)
            p = self._sample_param(c)
            if p is None:
                return ""
            toks.append(Token(c, p))
        rule = self._parser.serialize(toks)
        return rule if self._valid(rule, max_ops=max_ops) else ""

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, base: str, max_ops: int = MAX_OPS) -> str:
        """Mutate *base* rule to produce a novel candidate."""
        toks = self._parser.try_parse(base)
        if not toks:
            return self._build_markov(min(12, max_ops))

        op = self._rng.choices(
            ["replace_param", "replace_cmd", "insert", "delete", "swap"],
            weights=[0.36, 0.24, 0.18, 0.12, 0.10],
            k=1,
        )[0]

        t = toks[:]

        if op == "replace_param":
            idx = [i for i, x in enumerate(t) if _arity(x.cmd) > 0]
            if idx:
                i = self._rng.choice(idx)
                p = self._sample_param(t[i].cmd)
                if p is not None:
                    t[i] = Token(t[i].cmd, p)

        elif op == "replace_cmd":
            i = self._rng.randrange(len(t))
            prev = t[i - 1].cmd if i > 0 else None
            c = (
                _wchoice(self._trans.get(prev, Counter()), self._rng)
                if prev
                else _wchoice(self._start, self._rng)
            )
            if c and c in self._allowed_cmds:
                p = self._sample_param(c)
                if p is not None:
                    t[i] = Token(c, p)

        elif op == "insert":
            if len(t) < max_ops:
                i = self._rng.randint(0, len(t))
                prev = t[i - 1].cmd if i > 0 else None
                c = (
                    _wchoice(self._trans.get(prev, Counter()), self._rng)
                    if prev
                    else _wchoice(self._start, self._rng)
                )
                if c and c in self._allowed_cmds:
                    p = self._sample_param(c)
                    if p is not None:
                        t.insert(i, Token(c, p))

        elif op == "delete":
            if len(t) > 1:
                del t[self._rng.randrange(len(t))]

        elif op == "swap":
            if len(t) > 1:
                i = self._rng.randint(0, len(t) - 2)
                t[i], t[i + 1] = t[i + 1], t[i]

        rule = self._parser.serialize(t)
        return rule if self._valid(rule, max_ops=max_ops) else ""

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def generate_one(self, max_ops: int = MAX_OPS) -> str:
        """Generate a single candidate rule using the configured mixture."""
        w = self._weights
        strategy = self._rng.choices(
            ["markov", "ngram", "template", "random", "mutate"],
            weights=[w.markov, w.ngram, w.template, w.random_explore, w.mutate],
            k=1,
        )[0]

        if strategy == "mutate" and self._source:
            base = self._rng.choice(self._source)
            return self.mutate(base, max_ops=max_ops)
        if strategy == "template":
            return self._build_from_template(max_ops=max_ops)
        if strategy == "random":
            return self._build_random(max_ops=max_ops)
        return self._build_markov(max_ops=max_ops)

    def generate_batch(
        self,
        batch_size: int,
        max_ops: int = MAX_OPS,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Generate up to *batch_size* unique novel rule candidates."""
        exc = exclude or self._source_set
        out: list[str] = []
        tries = 0
        max_tries = batch_size * 15

        while len(out) < batch_size and tries < max_tries:
            tries += 1
            cand = self.generate_one(max_ops=max_ops)
            if cand and cand not in exc and cand not in out:
                out.append(cand)

        return out

    # ------------------------------------------------------------------
    # Offline scoring
    # ------------------------------------------------------------------

    def offline_score(self, rule: str) -> float:
        """Compute a heuristic novelty score without calling hashcat."""
        toks = self._parser.try_parse(rule)
        if not toks:
            return float("-inf")

        rarity_bonus = 0.0
        for t in toks:
            if t.param:
                pool = self._params.get(t.cmd, Counter())
                denom = sum(pool.values()) + 1
                freq = pool.get(t.param, 0) + 1
                rarity_bonus += -math.log(freq / denom)

        n = len(toks)
        length_bonus = 0.5 if 2 <= n <= 8 else (-0.5 if n > 12 else 0.0)

        cmd_counts = Counter(t.cmd for t in toks)
        repetitive_penalty = -0.6 if max(cmd_counts.values(), default=0) >= 4 else 0.0

        shape_freq = self._start.get(toks[0].cmd, 0)
        for i in range(len(toks) - 1):
            shape_freq += self._trans.get(toks[i].cmd, Counter()).get(toks[i + 1].cmd, 0)
        commonness_penalty = math.log(shape_freq + 1.0) * 0.2

        return rarity_bonus + length_bonus + repetitive_penalty - commonness_penalty


# ---------------------------------------------------------------------------
# Multiprocessing worker (spawn-safe, top-level)
# ---------------------------------------------------------------------------


def _worker_payload_to_generator(payload: dict[str, Any]) -> RuleGenerator:
    """Reconstruct a :class:`RuleGenerator` from a serializable payload."""
    parser = Parser()
    rng = random.Random(payload["seed"])
    return RuleGenerator(
        parser=parser,
        source_rules=payload["source_rules"],
        cmd_freq=Counter(payload["cmd"]),
        start_freq=Counter(payload["start"]),
        end_freq=Counter(payload["end"]),
        trans={k: Counter(v) for k, v in payload["trans"].items()},
        param_freq={k: Counter(v) for k, v in payload["params"].items()},
        len_dist=Counter(payload["len_dist"]),
        allow_param_fallback=payload.get("allow_param_fallback", False),
        weights=MixtureWeights(
            markov=payload.get("w_markov", 0.50),
            template=payload.get("w_template", 0.10),
            random_explore=payload.get("w_random", 0.05),
            mutate=payload.get("w_mutate", 0.35),
        ),
        rng=rng,
    )


def _build_rule_from_vom(
    gen: "RuleGenerator",
    vom: Any,
    max_ops: int,
) -> str:
    """Generate a rule using a pre-trained VariableOrderMarkov model.

    Samples an operation sequence from *vom*, then parameterises each
    operation using the generator's learned parameter frequencies.
    """
    ops: list[str] = vom.sample(max_len=max_ops)
    if not ops:
        return ""
    toks: list[Token] = []
    for c in ops:
        if c not in gen._allowed_cmds:
            return ""
        p = gen._sample_param(c)
        if p is None:
            return ""
        toks.append(Token(c, p))
    rule = gen._parser.serialize(toks)
    return rule if gen._valid(rule, max_ops=max_ops) else ""


def _build_rule_from_templates(
    gen: "RuleGenerator",
    templates_data: list[dict[str, Any]],
    max_ops: int,
) -> str:
    """Instantiate a randomly selected pre-learned template.

    Each entry in *templates_data* must contain a ``"steps"`` key with a
    list of ``[cmd, param_type]`` pairs (as written by
    :meth:`~ruleforge.templates.TemplateLearner.export_json`).
    """
    if not templates_data:
        return gen._build_from_template(max_ops)
    entry = gen._rng.choice(templates_data)
    steps = entry.get("steps", [])
    if not steps:
        return gen._build_from_template(max_ops)
    toks: list[Token] = []
    for item in steps:
        cmd = item[0] if isinstance(item, (list, tuple)) else item.get("cmd", "")
        if cmd not in gen._allowed_cmds:
            return ""
        p = gen._sample_param(cmd)
        if p is None:
            return ""
        toks.append(Token(cmd, p))
    if not toks:
        return ""
    rule = gen._parser.serialize(toks)
    return rule if gen._valid(rule, max_ops=max_ops) else ""


def worker_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker function (spawn-safe).

    Generates a batch and scores each candidate with the offline scorer.
    Optionally uses a pre-trained VariableOrderMarkov model (``vom_data``)
    and/or pre-learned templates (``templates_data``) from the payload.
    """
    import traceback as _tb

    try:
        gen = _worker_payload_to_generator(payload)
        batch_size = int(payload.get("batch_size", 2500))
        max_ops = int(payload.get("max_ops", 10))
        source_set = set(payload["source_rules"])

        # Reconstruct optional pre-trained VOM from payload
        vom: Any = None
        if "vom_data" in payload:
            try:
                from .markov import VariableOrderMarkov, MarkovModel, MarkovSampler
                vom = VariableOrderMarkov()
                vom._models = [MarkovModel.from_dict(d) for d in payload["vom_data"]]
                if vom._models:
                    vom._sampler = MarkovSampler(
                        model=vom._models[-1],
                        lower_models=vom._models[:-1],
                        rng=gen._rng,
                    )
                else:
                    vom = None
            except Exception:  # noqa: BLE001
                vom = None

        templates_data: list[dict[str, Any]] | None = payload.get("templates_data")

        out: list[tuple[str, float, str]] = []
        tries = 0
        max_tries = batch_size * 12

        rng = random.Random(payload["seed"])
        while len(out) < batch_size and tries < max_tries:
            tries += 1
            mutate_ratio = payload.get("mutate_ratio", 0.90)
            roll = rng.random()
            if roll < mutate_ratio and gen._source:
                cand = gen.mutate(rng.choice(gen._source), max_ops=max_ops)
                origin = "mutate"
            else:
                # Use pre-trained models when available; fall back to basic markov
                # Distribute the non-mutate budget: 70% markov/vom, 30% templates
                use_template = (templates_data is not None) and (rng.random() < 0.30)
                if use_template:
                    cand = _build_rule_from_templates(gen, templates_data, max_ops)  # type: ignore[arg-type]
                    origin = "template"
                elif vom is not None:
                    cand = _build_rule_from_vom(gen, vom, max_ops)
                    origin = "vom"
                else:
                    cand = gen._build_markov(max_ops=max_ops)
                    origin = "markov"
            if not cand or cand in source_set:
                continue
            score = gen.offline_score(cand)
            out.append((cand, score, origin))

        return {"ok": True, "items": out, "tries": tries}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "trace": _tb.format_exc()}
