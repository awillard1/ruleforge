# RuleForge

**RuleForge** is a modular, production-quality Hashcat rule generation framework built on
probabilistic modelling, evolutionary optimisation, and optional runtime feedback.

---

## Features

| Category | What's included |
|---|---|
| **Analysis** | Rule-file statistics, command/param frequency, transition matrices |
| **Probabilistic generation** | Markov chains (variable-order + Jelinek-Mercer), N-grams (Laplace / back-off / Kneser-Ney), PCFG |
| **Evolutionary optimisation** | Genetic algorithm with elitism, adaptive mutation, tournament selection |
| **Reinforcement learning** | Q-table agent with ε-greedy exploration and decay |
| **Bayesian optimisation** | UCB acquisition on a lightweight pure-Python GP |
| **Monte Carlo Tree Search** | UCT-based rule assembly |
| **Scoring** | 12-component composite scorer (novelty, coverage, entropy, Markov probability, …) |
| **Runtime feedback** | Optional Hashcat subprocess evaluation loop with caching |
| **Persistence** | 10-table SQLite database; WAL mode; upsert everywhere |
| **Checkpointing** | JSON snapshots with configurable keep-last pruning |
| **Reporting** | JSON / CSV / TSV / HTML / Markdown / PDF |
| **Visualisation** | Histograms, transition heat-maps, fitness history (matplotlib) |
| **Plugin system** | Dynamic file/directory loading; scorer, generator, semantics plug-points |
| **CLI** | Rich Typer interface: `analyze`, `generate`, `learn`, `train`, `evaluate`, `optimize`, `report`, … |

---

## Architecture

```
ruleforge/
├── parser.py        – Hashcat rule lexer/parser (Token, Parser, ParseError)
├── analyzer.py      – Statistical analysis (AnalysisResult, Analyzer)
├── templates.py     – Template learning and ranked sampling
├── markov.py        – Variable-order Markov with interpolation
├── ngram.py         – N-gram engine (Laplace / back-off / Kneser-Ney)
├── generator.py     – Probabilistic rule generator + spawn-safe worker
├── evolution.py     – Genetic algorithm optimiser
├── reinforcement.py – Q-learning agent
├── bayesian.py      – Bayesian optimisation with GP-UCB
├── mcts.py          – Monte Carlo Tree Search optimiser
├── runtime.py       – Hashcat subprocess evaluator + WordSampler
├── scoring.py       – 12-component composite scorer
├── grammar.py       – PCFG learner (password segmentation)
├── masks.py         – Mask learning (.hcmask export)
├── passwords.py     – Password corpus analysis
├── semantics.py     – Semantic word categorisation + plugin support
├── database.py      – SQLite persistence layer
├── checkpoint.py    – JSON checkpoint manager
├── reporting.py     – Multi-format report writer
├── visualization.py – Chart generator
├── plugins.py       – Plugin registry and dynamic loader
└── cli.py           – Typer CLI (entry-point: `ruleforge`)
```

---

## Installation

### From source (editable)

```bash
git clone https://github.com/awillard1/ruleforge.git
cd ruleforge
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install ruleforge
```

**Python ≥ 3.11 required.**

---

## Quick start

### Analyse an existing rule file

```bash
ruleforge analyze rockyou.rule
```

### Generate new rules

```bash
ruleforge generate rockyou.rule \
    --count 200000 \
    --max-ops 10 \
    --out candidate.rule
```

### Run the genetic optimiser

```bash
ruleforge optimize rockyou.rule \
    --generations 50 \
    --population 500 \
    --out evolved.rule
```

### Learn from a password corpus

```bash
ruleforge learn passwords.txt \
    --markov-order 3 \
    --ngram 3 \
    --grammar
```

### Evaluate with Hashcat

```bash
ruleforge evaluate candidate.rule \
    --wordlist rockyou.txt \
    --hash-file hashes.txt
```

### Generate a report

```bash
ruleforge report --db ruleforge.db --format html --out report.html
```

---

## Python API

```python
from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator
from ruleforge.scoring import Scorer
import random

parser = Parser()

# Analyse source rules
analyzer = Analyzer(parser)
analyzer.ingest_file("rockyou.rule")
result = analyzer.result()
print(f"Unique rules: {result.unique_count}, avg ops: {result.avg_ops:.1f}")

# Build generator from learned statistics
rng = random.Random(42)
gen = RuleGenerator(
    parser=parser,
    source_rules=list(analyzer.unique_rules),
    cmd_freq=analyzer.cmd,
    start_freq=analyzer.start,
    end_freq=analyzer.end,
    trans=analyzer.trans,
    param_freq=analyzer.params,
    len_dist=analyzer.len_dist,
    rng=rng,
)

# Generate + score
candidates = gen.generate_batch(10_000, max_ops=10)
scorer = Scorer(
    parser=parser,
    cmd_freq=analyzer.cmd,
    param_freq=analyzer.params,
    trans=analyzer.trans,
    start_freq=analyzer.start,
)
top = scorer.rank(candidates)[:200]
for rule, score in top:
    print(f"{score:.4f}  {rule}")
```

---

## Configuration

RuleForge accepts YAML, TOML, or JSON configuration files.
A full-annotated example lives at [`config/default.yaml`](config/default.yaml).

Pass a config file with:

```bash
ruleforge generate rockyou.rule --config config/default.yaml
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Linting and type-checking

```bash
ruff check ruleforge/
mypy ruleforge/
black --check ruleforge/ tests/
```

---

## Project layout

```
ruleforge/          – Python package
tests/              – pytest test suite
examples/           – Standalone usage scripts
config/             – Default configuration templates
benchmarks/         – Performance benchmarks
docs/               – Extended documentation
plugins/            – Drop-in plugin directory
hashcat_rules.py    – Original monolithic script (preserved for reference)
pyproject.toml      – Build / dependency / tool config
```

---

## Backward compatibility

The original `hashcat_rules.py` is preserved unmodified. The modular framework
is a strict superset: the same `worker_generate` payload format is used in
`ruleforge/generator.py`, and the `WordSampler` is a direct port.

---

## License

MIT
