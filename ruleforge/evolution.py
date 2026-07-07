"""
ruleforge/evolution.py
----------------------
Evolutionary Optimizer — Genetic Algorithm for Hashcat rule evolution.

Features:
- Tournament selection
- Crossover (single-point, uniform)
- Mutation (delegated to RuleGenerator)
- Elitism
- Adaptive mutation rate
- Checkpointing / resume
- Parallel fitness evaluation
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .parser import Parser, Token, _arity
from .generator import RuleGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A single rule in the population."""

    rule: str
    fitness: float = 0.0
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "fitness": self.fitness, "generation": self.generation}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Individual":
        return cls(rule=d["rule"], fitness=float(d["fitness"]), generation=int(d["generation"]))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 500
    elite_fraction: float = 0.05          # fraction of population kept as elite
    tournament_size: int = 5
    crossover_prob: float = 0.70
    mutation_prob: float = 0.30
    mutation_rate_min: float = 0.10
    mutation_rate_max: float = 0.50
    adaptive_mutation: bool = True
    max_generations: int = 100
    target_rules: int = 1000
    stagnation_limit: int = 15            # generations without improvement → adapt
    max_ops: int = 10
    num_workers: int = 1
    checkpoint_interval: int = 10         # save every N generations


# ---------------------------------------------------------------------------
# Fitness function type
# ---------------------------------------------------------------------------

FitnessFn = Callable[[str], float]


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------


class GeneticOptimizer:
    """Genetic algorithm for evolving high-fitness Hashcat rules.

    Args:
        parser:     A :class:`~ruleforge.parser.Parser` instance.
        generator:  A :class:`~ruleforge.generator.RuleGenerator` for mutation.
        fitness_fn: Callable mapping a rule string to a float fitness score.
        config:     :class:`EvolutionConfig` instance.
        rng:        Random source.
    """

    def __init__(
        self,
        parser: Parser,
        generator: RuleGenerator,
        fitness_fn: FitnessFn,
        config: EvolutionConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._parser = parser
        self._gen = generator
        self._fitness_fn = fitness_fn
        self._cfg = config or EvolutionConfig()
        self._rng = rng or random.Random()

        self._population: list[Individual] = []
        self._generation: int = 0
        self._best_fitness: float = float("-inf")
        self._stagnation: int = 0
        self._current_mutation_prob: float = self._cfg.mutation_prob

        # History
        self._best_per_gen: list[float] = []
        self._mean_per_gen: list[float] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, seed_rules: list[str] | None = None) -> None:
        """Create the initial population.

        Args:
            seed_rules: Optional list of rules to seed the population.
        """
        pop: list[Individual] = []
        if seed_rules:
            for r in seed_rules[: self._cfg.population_size]:
                if self._parser.validate(r):
                    pop.append(Individual(rule=r, generation=0))

        # Fill remaining slots
        while len(pop) < self._cfg.population_size:
            cand = self._gen.generate_one(max_ops=self._cfg.max_ops)
            if cand and self._parser.validate(cand):
                pop.append(Individual(rule=cand, generation=0))

        self._evaluate_population(pop)
        self._population = sorted(pop, key=lambda x: x.fitness, reverse=True)
        logger.info("Population initialized: size=%d", len(self._population))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(self, individuals: list[Individual]) -> None:
        """Compute fitness for all individuals that have fitness == 0.0."""
        for ind in individuals:
            ind.fitness = self._fitness_fn(ind.rule)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self) -> Individual:
        """Tournament selection."""
        contestants = self._rng.choices(self._population, k=self._cfg.tournament_size)
        return max(contestants, key=lambda x: x.fitness)

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(self, parent_a: Individual, parent_b: Individual) -> tuple[str, str]:
        """Single-point crossover on token lists."""
        toks_a = self._parser.try_parse(parent_a.rule) or []
        toks_b = self._parser.try_parse(parent_b.rule) or []

        if not toks_a or not toks_b:
            return parent_a.rule, parent_b.rule

        if self._rng.random() > self._cfg.crossover_prob:
            return parent_a.rule, parent_b.rule

        pt_a = self._rng.randint(0, len(toks_a))
        pt_b = self._rng.randint(0, len(toks_b))

        child_toks_a = toks_a[:pt_a] + toks_b[pt_b:]
        child_toks_b = toks_b[:pt_b] + toks_a[pt_a:]

        max_ops = self._cfg.max_ops

        def _make(toks: list[Token]) -> str:
            if not toks or len(toks) > max_ops:
                return ""
            r = self._parser.serialize(toks)
            return r if self._parser.validate(r, max_ops=max_ops) else ""

        return _make(child_toks_a), _make(child_toks_b)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, individual: Individual) -> str:
        """Apply mutation with probability *current_mutation_prob*."""
        if self._rng.random() < self._current_mutation_prob:
            return self._gen.mutate(individual.rule, max_ops=self._cfg.max_ops)
        return individual.rule

    # ------------------------------------------------------------------
    # Adaptive mutation
    # ------------------------------------------------------------------

    def _adapt_mutation(self) -> None:
        """Increase mutation rate on stagnation, decrease on improvement."""
        if not self._cfg.adaptive_mutation:
            return
        if self._stagnation >= self._cfg.stagnation_limit:
            self._current_mutation_prob = min(
                self._cfg.mutation_rate_max,
                self._current_mutation_prob * 1.2,
            )
            logger.debug("Mutation rate increased to %.3f", self._current_mutation_prob)
        else:
            self._current_mutation_prob = max(
                self._cfg.mutation_rate_min,
                self._current_mutation_prob * 0.98,
            )

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------

    def _step(self) -> None:
        """Execute one generation."""
        cfg = self._cfg
        pop_size = cfg.population_size
        elite_n = max(1, int(pop_size * cfg.elite_fraction))

        # Elitism: keep top individuals
        new_pop: list[Individual] = self._population[:elite_n]

        while len(new_pop) < pop_size:
            pa = self._tournament_select()
            pb = self._tournament_select()

            child_a_rule, child_b_rule = self._crossover(pa, pb)

            for rule in (child_a_rule, child_b_rule):
                if len(new_pop) >= pop_size:
                    break
                # Wrap in a temp Individual for mutation
                tmp = Individual(rule=rule or pa.rule, generation=self._generation)
                mutated = self._mutate(tmp)
                if mutated and self._parser.validate(mutated, max_ops=cfg.max_ops):
                    new_pop.append(Individual(rule=mutated, generation=self._generation))
                elif rule and self._parser.validate(rule, max_ops=cfg.max_ops):
                    new_pop.append(Individual(rule=rule, generation=self._generation))

        # Evaluate new individuals
        self._evaluate_population(new_pop[elite_n:])

        self._population = sorted(new_pop, key=lambda x: x.fitness, reverse=True)
        self._generation += 1

        best = self._population[0].fitness
        mean = sum(x.fitness for x in self._population) / len(self._population)
        self._best_per_gen.append(best)
        self._mean_per_gen.append(mean)

        if best > self._best_fitness:
            self._best_fitness = best
            self._stagnation = 0
        else:
            self._stagnation += 1

        self._adapt_mutation()
        logger.debug(
            "Gen %d: best=%.4f mean=%.4f stagnation=%d mut=%.3f",
            self._generation,
            best,
            mean,
            self._stagnation,
            self._current_mutation_prob,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        checkpoint_path: Path | None = None,
    ) -> list[Individual]:
        """Run the genetic algorithm.

        Args:
            checkpoint_path: Directory to save checkpoints into.

        Returns:
            The final (sorted) population.
        """
        cfg = self._cfg
        logger.info(
            "Starting evolution: pop=%d gens=%d target=%d",
            cfg.population_size,
            cfg.max_generations,
            cfg.target_rules,
        )
        if not self._population:
            self.initialize()

        for _ in range(cfg.max_generations):
            self._step()
            if checkpoint_path and self._generation % cfg.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path)

        return self._population

    # ------------------------------------------------------------------
    # Top results
    # ------------------------------------------------------------------

    def top_rules(self, n: int | None = None) -> list[str]:
        """Return top rule strings by fitness."""
        pop = self._population
        if n:
            pop = pop[:n]
        return [ind.rule for ind in pop]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cp = {
            "generation": self._generation,
            "best_fitness": self._best_fitness,
            "stagnation": self._stagnation,
            "mutation_prob": self._current_mutation_prob,
            "best_per_gen": self._best_per_gen,
            "mean_per_gen": self._mean_per_gen,
            "population": [ind.to_dict() for ind in self._population],
        }
        cp_file = directory / f"evo_gen_{self._generation:05d}.json"
        cp_file.write_text(json.dumps(cp, indent=2), encoding="utf-8")
        logger.info("Checkpoint saved: %s", cp_file)

    def load_checkpoint(self, cp_file: Path) -> None:
        """Resume from a checkpoint JSON file."""
        data = json.loads(cp_file.read_text(encoding="utf-8"))
        self._generation = int(data["generation"])
        self._best_fitness = float(data["best_fitness"])
        self._stagnation = int(data["stagnation"])
        self._current_mutation_prob = float(data["mutation_prob"])
        self._best_per_gen = list(data.get("best_per_gen", []))
        self._mean_per_gen = list(data.get("mean_per_gen", []))
        self._population = [Individual.from_dict(d) for d in data["population"]]
        logger.info("Resumed from checkpoint: gen=%d", self._generation)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a summary of the optimization run."""
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": self._best_fitness,
            "current_mutation_prob": self._current_mutation_prob,
            "stagnation": self._stagnation,
            "best_per_generation": self._best_per_gen,
            "mean_per_generation": self._mean_per_gen,
        }
