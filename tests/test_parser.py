"""Tests for ruleforge/parser.py"""

import pytest
from ruleforge.parser import Parser, Token, ParseError, parse, serialize, validate, MAX_OPS


class TestToken:
    def test_valid_no_param(self):
        t = Token("l", "")
        assert t.cmd == "l"
        assert t.param == ""
        assert t.arity == 0

    def test_valid_one_param(self):
        t = Token("$", "1")
        assert t.cmd == "$"
        assert t.param == "1"
        assert t.arity == 1

    def test_valid_two_param(self):
        t = Token("s", "aA")
        assert t.param == "aA"

    def test_invalid_cmd_length(self):
        with pytest.raises(ValueError):
            Token("ll", "")

    def test_invalid_arity(self):
        with pytest.raises(ValueError):
            Token("$", "")   # needs 1 char

    def test_invalid_param_char(self):
        with pytest.raises(ValueError):
            Token("$", "\n")

    def test_serialize(self):
        assert Token("l", "").serialize() == "l"
        assert Token("$", "1").serialize() == "$1"
        assert Token("s", "aA").serialize() == "saA"


class TestParser:
    def setup_method(self):
        self.p = Parser()

    def test_empty_line(self):
        assert self.p.parse("") == []

    def test_comment_line(self):
        assert self.p.parse("# a comment") == []
        assert self.p.parse("  # another") == []

    def test_single_noop(self):
        toks = self.p.parse(":")
        assert len(toks) == 1
        assert toks[0].cmd == ":"

    def test_lowercase(self):
        toks = self.p.parse("l")
        assert toks[0].cmd == "l"

    def test_append_char(self):
        toks = self.p.parse("$1")
        assert len(toks) == 1
        assert toks[0].cmd == "$"
        assert toks[0].param == "1"

    def test_multi_op(self):
        toks = self.p.parse("l$1")
        assert len(toks) == 2
        assert toks[0].cmd == "l"
        assert toks[1].cmd == "$"
        assert toks[1].param == "1"

    def test_substitute(self):
        toks = self.p.parse("saA")
        assert toks[0].cmd == "s"
        assert toks[0].param == "aA"

    def test_unknown_op_raises(self):
        with pytest.raises(ParseError):
            self.p.parse("X")

    def test_truncated_param_raises(self):
        with pytest.raises(ParseError):
            self.p.parse("$")  # needs 1 param

    def test_try_parse_returns_none_on_invalid(self):
        assert self.p.try_parse("X") is None

    def test_try_parse_valid(self):
        toks = self.p.try_parse("l$1")
        assert toks is not None
        assert len(toks) == 2

    def test_validate_valid(self):
        assert self.p.validate("l$1") is True

    def test_validate_empty(self):
        assert self.p.validate("") is False

    def test_validate_max_ops(self):
        # 32 l's should be invalid (> 31)
        rule = "l" * 32
        assert self.p.validate(rule, max_ops=31) is False
        assert self.p.validate("l" * 31, max_ops=31) is True

    def test_serialize(self):
        toks = self.p.parse("l$1")
        assert self.p.serialize(toks) == "l$1"

    def test_normalize(self):
        assert self.p.normalize("l$1") == "l$1"
        assert self.p.normalize("XXXXX") is None

    def test_op_name(self):
        assert Parser.op_name("l") == "lowercase"
        assert Parser.op_name("$") == "append"

    def test_module_level_functions(self):
        toks = parse("l$1")
        assert serialize(toks) == "l$1"
        assert validate("l$1") is True
