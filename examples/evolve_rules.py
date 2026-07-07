"""
Example: run the genetic algorithm optimizer.

Usage::

    python examples/evolve_rules.py rockyou_sample.rule
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator
from ruleforge.scoring import Scorer
from ruleforge.evolution import GeneticOptimizer, EvolutionConfig
import random, argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rules_file", type=Path)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--population", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parser = Parser()
    analyzer = Analyzer(parser)
    analyzer.ingest_file(args.rules_file)

    rng = random.Random(args.seed)
    gen = RuleGenerator(
        parser=parser,
        source_rules=list(analyzer.unique_rules),
        cmd_freq=analyzer.cmd,
        start_freq=analyzer.start,
        end_freq=analyzer.end,
        trans=analyzer.trans,
        param_freq=analyzer.params,
        len_dist=analyzer.len_dist,
        allow_param_fallback=True,
        rng=rng,
    )
    scorer = Scorer(
        parser=parser,
        cmd_freq=analyzer.cmd,
        param_freq=analyzer.params,
        trans=analyzer.trans,
        start_freq=analyzer.start,
    )

    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        elite_fraction=0.05,
        tournament_size=5,
        crossover_prob=0.70,
        mutation_prob=0.30,
        adaptive_mutation=True,
        stagnation_limit=5,
        checkpoint_dir=None,
    )

    optimizer = GeneticOptimizer(
        config=config,
        generator=gen,
        scorer=scorer,
        rng=rng,
    )

    print(f"[*] Running {args.generations} generations …")
    best = optimizer.run()
    print(f"\n[*] Best {min(20, len(best))} rules:")
    for rule, score in sorted(best, key=lambda x: x[1], reverse=True)[:20]:
        print(f"{score:.4f}\t{rule}")


if __name__ == "__main__":
    main()
