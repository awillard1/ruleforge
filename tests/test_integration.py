"""Integration test — full pipeline from analysis to generation."""

import random
import pytest
from pathlib import Path
from collections import Counter

from ruleforge.parser import Parser
from ruleforge.analyzer import Analyzer
from ruleforge.generator import RuleGenerator
from ruleforge.templates import TemplateLearner
from ruleforge.markov import VariableOrderMarkov
from ruleforge.scoring import Scorer
from ruleforge.reporting import Report, Reporter
from ruleforge.database import Database


SOURCE_RULES = [
    "l$1", "l$!", "l$@", "lu", "uc", "cl", "l", "u", "c", "r", "d",
    "l$2", "u^!", "c$0", "lu$1", "l$1u", "u$2", "c^a", "r$!", "d$1",
    "lc", "ul", "cu", "rc", "dl", "l$3", "u$!", "c$2", "l^a",
]


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def analyzer(parser):
    a = Analyzer(parser)
    a.ingest_rules(SOURCE_RULES)
    return a


@pytest.fixture
def generator(parser, analyzer):
    rng = random.Random(42)
    return RuleGenerator(
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


class TestFullPipeline:
    def test_analyze_then_generate(self, parser, analyzer, generator):
        result = analyzer.result()
        assert result.unique_count > 0

        batch = generator.generate_batch(50, max_ops=8)
        assert len(batch) > 0
        for rule in batch:
            assert parser.validate(rule, max_ops=8)

    def test_template_learning(self, parser, analyzer):
        learner = TemplateLearner(parser)
        learner.ingest_rules(list(analyzer.unique_rules))
        top = learner.top_n(5)
        assert len(top) > 0

    def test_markov_then_generate(self, parser):
        vom = VariableOrderMarkov(max_order=2)
        vom.train(SOURCE_RULES, parser)
        seqs = vom.sample_many(10, max_len=6)
        # Check all sequences are lists
        for seq in seqs:
            assert isinstance(seq, list)

    def test_scoring(self, parser, analyzer):
        scorer = Scorer(
            parser=parser,
            cmd_freq=analyzer.cmd,
            param_freq=analyzer.params,
            trans=analyzer.trans,
            start_freq=analyzer.start,
        )
        ranked = scorer.rank(SOURCE_RULES)
        assert len(ranked) == len(SOURCE_RULES)

    def test_database_roundtrip(self, generator, tmp_path):
        db = Database(":memory:")
        rules = generator.generate_batch(20)
        db.bulk_insert_rules([(r, generator.offline_score(r), "test") for r in rules])
        assert db.rule_count() == len(rules)
        top = db.top_rules_by_fitness(10)
        assert len(top) <= 10
        db.close()

    def test_reporting(self, analyzer, tmp_path):
        result = analyzer.result()
        report = Report(
            title="Integration Test",
            analysis=result.to_dict(),
            generation={"kept_rules": 100, "rounds": 5},
            top_rules=[{"rule": "l$1", "score": 1.0, "origin": "markov", "round": 1}],
        )
        reporter = Reporter(report)
        reporter.write_all(tmp_path, stem="integration")
        assert (tmp_path / "integration.json").exists()
        assert (tmp_path / "integration.html").exists()
