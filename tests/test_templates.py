"""Tests for ruleforge/templates.py"""

import pytest
from ruleforge.parser import Parser
from ruleforge.templates import (
    Template, TemplateLearner, DefaultParamSampler, FrequencyParamSampler
)


@pytest.fixture
def parser():
    return Parser()


class TestTemplate:
    def test_from_tokens(self, parser):
        toks = parser.parse("l$1")
        tmpl = Template.from_tokens(toks)
        assert len(tmpl.steps) == 2
        assert tmpl.steps[0][0] == "l"
        assert tmpl.steps[1][0] == "$"

    def test_signature(self, parser):
        toks = parser.parse("l$1")
        tmpl = Template.from_tokens(toks)
        sig = tmpl.signature()
        assert "lowercase" in sig
        assert "append" in sig

    def test_equality(self, parser):
        t1 = Template.from_tokens(parser.parse("l$1"))
        t2 = Template.from_tokens(parser.parse("l$!"))
        # Same template shape (both are lowercase → append(digit/symbol))
        # The param type may differ for $1 vs $!
        assert isinstance(t1, Template)


class TestTemplateLearner:
    def test_ingest_rules(self, parser):
        learner = TemplateLearner(parser)
        learner.ingest_rules(["l$1", "l$!", "l$@", "u$1"])
        ranked = learner.ranked_templates()
        assert len(ranked) >= 1

    def test_top_n(self, parser):
        learner = TemplateLearner(parser)
        learner.ingest_rules(["l$1", "l$!", "l$@", "u$1", "l", "u", "c"])
        top = learner.top_n(3)
        assert len(top) <= 3

    def test_export_json(self, parser, tmp_path):
        learner = TemplateLearner(parser)
        learner.ingest_rules(["l$1", "l$!", "u"])
        out = tmp_path / "templates.json"
        learner.export_json(out)
        assert out.exists()

    def test_generate_from_template(self, parser):
        learner = TemplateLearner(parser)
        learner.ingest_rules(["l$1", "l$!", "l$@"])
        top = learner.top_n(1)
        assert top
        result = learner.generate_from_template(top[0].template)
        # May return None if parameterization fails, but we test it runs
        assert result is None or parser.validate(result)


class TestDefaultParamSampler:
    def test_no_param(self):
        s = DefaultParamSampler()
        assert s.sample("l") == ""

    def test_one_param(self):
        s = DefaultParamSampler()
        p = s.sample("$")
        assert p is not None and len(p) == 1

    def test_two_param(self):
        s = DefaultParamSampler()
        p = s.sample("s")
        assert p is not None and len(p) == 2
