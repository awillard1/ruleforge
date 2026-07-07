# Project Specification
## Project Name
RuleForge

## Goal

Develop the most advanced offline Hashcat rule learning and optimization framework possible.

The system must preserve 100% compatibility with existing Hashcat rule syntax while learning from existing rule files, password lists, and runtime cracking results to generate increasingly effective rules.

This project replaces the current monolithic script with a modular architecture while preserving every existing feature.

---

# Primary Design Goals

1. Never remove existing functionality.
2. Every release must be backwards compatible.
3. Every component should be independently testable.
4. Support Windows, Linux and macOS.
5. Python 3.11+
6. Fully type hinted.
7. Extensive logging.
8. Modular architecture.
9. Plugin system.
10. Resume interrupted jobs.

---

# Core Requirements

The following functionality MUST exist.

## Rule Parser

Support full Hashcat rule syntax.

Requirements:

- Parse every operation
- Validate rules
- Serialize rules
- Deserialize rules
- Preserve comments
- Preserve formatting when requested
- Detect malformed rules
- Maximum compatibility with latest Hashcat

---

## Rule Analyzer

Analyze one or more rule files.

Collect statistics including:

- operation frequency
- operation ordering
- parameter frequencies
- operation pairs
- operation triplets
- transition probabilities
- rule lengths
- unique rules
- duplicate rules
- invalid rules
- entropy
- rule complexity
- operation rarity

Export analysis as JSON.

---

## Rule Template Learning

Extract templates from rules.

Example

Original:

```
l$1
l$!
l$@
l$2025
```

Template

```
lowercase
append
parameter
```

Learn reusable templates.

Rank templates by usefulness.

---

## Higher Order Markov Model

Current script uses first order Markov.

Replace with configurable models.

Support

- first order
- second order
- third order
- variable order

Allow interpolation.

---

## N-Gram Engine

Implement

- bigram
- trigram
- 4-gram

Backoff smoothing.

Laplace smoothing.

Kneser-Ney if possible.

---

## Probabilistic Rule Generator

Generate rules using

- Markov
- N-Gram
- Templates
- Random exploration

Weighted configurable mixture.

---

## Evolutionary Optimizer

Implement genetic algorithm.

Population

Fitness

Selection

Mutation

Crossover

Elitism

Tournament selection

Adaptive mutation

Checkpointing

Resume

Parallel evaluation

---

## Reinforcement Learning

Implement optional RL.

State

Current rule.

Actions

Insert operation

Delete operation

Replace operation

Swap operation

Modify parameter

Reward

New passwords generated

Coverage

Unique outputs

Runtime efficiency

---

## Bayesian Optimization

Use Bayesian optimization to explore unexplored rule space.

Prioritize promising regions.

Avoid redundant exploration.

---

## Monte Carlo Tree Search

Treat each rule as a tree.

Operations are nodes.

Use UCT.

Support configurable exploration constant.

---

# Runtime Evaluation

Support two modes.

Offline

Runtime

Runtime mode uses

hashcat --stdout

Measure

unique outputs

coverage

duplicate rate

new outputs

speed

cache results.

---

# Password Analysis

Analyze password corpora.

Extract

lengths

character classes

years

months

days

keyboard walks

leet patterns

camel case

mixed case

company names

sports

movies

games

anime

cities

countries

brands

common suffixes

common prefixes

numbers

symbols

repeated characters

---

# Semantic Analyzer

Automatically categorize words.

Use

WordNet

Wikipedia dumps (optional)

IMDB

MusicBrainz

GeoNames

Public dictionaries

Allow plugins.

---

# Grammar Learning

Implement PCFG.

Learn structures such as

Word+Year

Name+123

Word+!

Month+Year

Company+Number

Generate probable grammars.

---

# Mask Learning

Generate masks from passwords.

Example

Password

```
Football2025!
```

Mask

```
?u?l?l?l?l?l?l?l?l?d?d?d?d?s
```

Cluster masks.

Rank masks.

Export .hcmask.

---

# Rule Effectiveness Database

Store every generated rule.

Track

times generated

times evaluated

runtime

coverage

duplicates

fitness

last used

best score

Store in SQLite.

---

# Persistent Learning

Every execution improves future generations.

Never lose learned statistics.

Support exporting and importing models.

---

# Scoring Engine

Score should combine

novelty

coverage

entropy

uniqueness

runtime

historical success

parameter diversity

operation diversity

Template rarity

Markov probability

N-Gram probability

Grammar usefulness

Weights configurable.

---

# Plugin System

Allow plugins.

Plugin types

Analyzer

Generator

Scorer

Exporter

Importer

Runtime

Password Analyzer

Grammar

Mask

Visualization

---

# Parallelization

Use multiprocessing.

Avoid GIL bottlenecks.

Support

ProcessPoolExecutor

Shared memory

Batch scheduling

Dynamic worker allocation.

---

# Database

SQLite.

Tables

rules

templates

statistics

runtime

coverage

fitness

models

jobs

history

passwords

---

# Checkpointing

Resume any interrupted generation.

Save

population

scores

models

random seeds

statistics

database state

---

# Reporting

Generate

JSON

CSV

TSV

HTML

Markdown

PDF

Include charts.

---

# Visualization

Generate

histograms

operation graphs

transition graphs

fitness graphs

coverage graphs

Markov graphs

---

# CLI

Use argparse or Typer.

Commands

analyze

generate

learn

train

evaluate

benchmark

optimize

resume

report

visualize

database

plugins

---

# Configuration

Support

JSON

YAML

TOML

Command line overrides.

---

# Logging

Structured logging.

Multiple verbosity levels.

Progress bars.

Timing.

Memory usage.

---

# Testing

Minimum

95% coverage.

Tests

unit

integration

performance

stress

fuzz

property based

---

# Documentation

Auto generate API documentation.

Include

architecture

examples

CLI reference

developer guide

plugin guide

---

# Performance Goals

Analyze

1 million rules in under 60 seconds.

Generate

10 million candidate rules per hour on 16 cores.

SQLite cache under 2GB.

Memory under 4GB for normal workloads.

---

# Coding Standards

PEP8

Black

Ruff

MyPy

Pytest

Type hints everywhere.

No global mutable state.

Dependency injection where practical.

---

# Project Structure

```
RuleForge/
│
├── ruleforge/
│   ├── parser.py
│   ├── analyzer.py
│   ├── templates.py
│   ├── markov.py
│   ├── ngram.py
│   ├── generator.py
│   ├── evolution.py
│   ├── reinforcement.py
│   ├── bayesian.py
│   ├── mcts.py
│   ├── runtime.py
│   ├── scoring.py
│   ├── grammar.py
│   ├── masks.py
│   ├── passwords.py
│   ├── semantics.py
│   ├── database.py
│   ├── checkpoint.py
│   ├── reporting.py
│   ├── visualization.py
│   ├── plugins.py
│   └── cli.py
│
├── tests/
├── benchmarks/
├── examples/
├── docs/
├── plugins/
├── config/
└── README.md
```

---

# Final Objective

The finished project should significantly outperform traditional rule generators by combining:

- Statistical learning
- Higher-order Markov models
- N-Grams
- Template mining
- Evolutionary optimization
- Reinforcement learning
- Bayesian optimization
- Monte Carlo Tree Search
- Runtime feedback
- Persistent learning
- Grammar extraction
- Semantic analysis
- Mask generation

while remaining fully compatible with Hashcat rule syntax and preserving all existing capabilities from the original script.
