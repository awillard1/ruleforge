"""
Example: Basic analysis + generation pipeline.

Usage::

    python examples/basic_generate.py rockyou_sample.rule --count 5000

The script reads an existing Hashcat rule file, learns its statistical
structure, generates new candidate rules, scores them, and writes the
top results to stdout.
"""

import argparse
import random
import sys
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator
from ruleforge.scoring import Scorer


def main() -> None:
    ap = argparse.ArgumentParser(description="RuleForge basic generate example")
    ap.add_argument("rules_file", type=Path, help="Input .rule file")
    ap.add_argument("--count", type=int, default=5_000, help="Rules to generate")
    ap.add_argument("--max-ops", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=200, help="Print top N rules")
    args = ap.parse_args()

    parser = Parser()

    # ── analyse ─────────────────────────────────────────────────────────
    print(f"[*] Analysing {args.rules_file} …", flush=True)
    analyzer = Analyzer(parser)
    analyzer.ingest_file(args.rules_file)
    result = analyzer.result()
    print(f"    Unique rules  : {result.unique_count}")
    print(f"    Avg length    : {result.avg_ops:.2f} ops")
    print(f"    Top commands  : {', '.join(k for k, _ in result.top_cmds[:5])}")

    # ── generate ────────────────────────────────────────────────────────
    print(f"\n[*] Generating {args.count} candidate rules …", flush=True)
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

    candidates = gen.generate_batch(args.count, max_ops=args.max_ops)
    print(f"    Generated     : {len(candidates)} valid rules")

    # ── score & rank ────────────────────────────────────────────────────
    print("\n[*] Scoring …", flush=True)
    scorer = Scorer(
        parser=parser,
        cmd_freq=analyzer.cmd,
        param_freq=analyzer.params,
        trans=analyzer.trans,
        start_freq=analyzer.start,
    )
    ranked = scorer.rank(candidates)

    # ── print top N ─────────────────────────────────────────────────────
    print(f"\n[*] Top {args.top} rules:")
    for rule, score in ranked[: args.top]:
        print(f"{score:.4f}\t{rule}")


if __name__ == "__main__":
    main()
