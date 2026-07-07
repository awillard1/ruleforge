"""
ruleforge/parser.py
-------------------
Full Hashcat rule syntax parser, validator, serializer and deserializer.

Hashcat rule operations reference (v6+):
  https://hashcat.net/wiki/doku.php?id=rule_based_attack
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hashcat operation arity table
# ---------------------------------------------------------------------------
#
# 0-param ops  — no arguments follow the command character
# 1-param ops  — one character follows
# 2-param ops  — two characters follow
# 3-param ops  — three characters follow  (position + 2 chars for 'i')
#                                          (position + length for 'O')
# ---------------------------------------------------------------------------

#: Operations that take no parameters.
NO_PARAM: frozenset[str] = frozenset(":lucCtrdpf{}[]qkKEPIRMV")

#: Operations that take exactly one parameter character.
ONE_PARAM: frozenset[str] = frozenset("$^TDi@zZyY")

#: Operations that take exactly two parameter characters.
TWO_PARAM: frozenset[str] = frozenset("so")

#: Operations that take exactly three parameter characters (pos + 2 chars).
THREE_PARAM: frozenset[str] = frozenset("iO")

#: All known operation characters.
ALL_OPS: frozenset[str] = NO_PARAM | ONE_PARAM | TWO_PARAM | THREE_PARAM

#: Maximum number of operations in a single rule (Hashcat hard limit).
MAX_OPS: int = 31


def _arity(cmd: str) -> int:
    """Return the number of parameter characters for *cmd*, or -1 if unknown."""
    if cmd in NO_PARAM:
        return 0
    if cmd in ONE_PARAM:
        return 1
    if cmd in TWO_PARAM:
        return 2
    if cmd in THREE_PARAM:
        return 3
    return -1


def _ok_char(ch: str) -> bool:
    """Return True if *ch* is a printable ASCII character valid in a rule."""
    if not isinstance(ch, str) or len(ch) != 1:
        return False
    o = ord(ch)
    return 32 <= o <= 126 and ch not in ("\r", "\n", "\t")


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Token:
    """A single Hashcat rule operation with its parameters.

    Attributes:
        cmd:   The single-character operation code.
        param: The raw parameter string (empty if no parameters required).
    """

    cmd: str
    param: str = ""

    def __post_init__(self) -> None:
        if len(self.cmd) != 1:
            raise ValueError(f"Token.cmd must be exactly 1 character, got {self.cmd!r}")
        if _arity(self.cmd) < 0:
            raise ValueError(f"Unknown operation {self.cmd!r}")
        expected = _arity(self.cmd)
        if len(self.param) != expected:
            raise ValueError(
                f"Operation {self.cmd!r} expects {expected} param chars, "
                f"got {len(self.param)}: {self.param!r}"
            )
        for ch in self.param:
            if not _ok_char(ch):
                raise ValueError(
                    f"Invalid parameter character {ch!r} in operation {self.cmd!r}"
                )

    def serialize(self) -> str:
        """Return the canonical string representation of this token."""
        return self.cmd + self.param

    @property
    def arity(self) -> int:
        """Number of parameter characters expected by this operation."""
        return _arity(self.cmd)


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------


class ParseError(ValueError):
    """Raised when a rule string cannot be parsed."""

    def __init__(self, message: str, position: int | None = None, rule: str = "") -> None:
        self.position = position
        self.rule = rule
        detail = f" (at position {position})" if position is not None else ""
        super().__init__(f"{message}{detail}: {rule!r}" if rule else f"{message}{detail}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Parse, validate, serialize and deserialize Hashcat rule strings.

    The parser is intentionally stateless and thread-safe.

    Examples:
        >>> p = Parser()
        >>> tokens = p.parse("l$1")
        >>> [t.serialize() for t in tokens]
        ['l', '$1']
        >>> p.serialize(tokens)
        'l$1'
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, line: str) -> list[Token]:
        """Parse *line* into a list of :class:`Token` objects.

        Lines that are blank or start with ``#`` return an empty list
        (they are valid but carry no operations).

        Args:
            line: A single rule string (may include trailing whitespace).

        Returns:
            Ordered list of :class:`Token` objects.

        Raises:
            ParseError: If *line* contains an invalid or malformed rule.
        """
        if not isinstance(line, str):
            raise ParseError("Input must be a string")
        s = line.strip()
        if not s or s.startswith("#"):
            return []

        tokens: list[Token] = []
        i = 0
        n = len(s)

        while i < n:
            ch = s[i]
            if ch.isspace():
                i += 1
                continue

            ar = _arity(ch)
            if ar < 0:
                raise ParseError(f"Unknown operation {ch!r}", position=i, rule=s)

            param_start = i + 1
            param_end = param_start + ar

            if param_end > n:
                raise ParseError(
                    f"Operation {ch!r} requires {ar} parameter chars but input ends",
                    position=i,
                    rule=s,
                )

            param = s[param_start:param_end]
            for j, pc in enumerate(param):
                if not _ok_char(pc):
                    raise ParseError(
                        f"Invalid parameter character {pc!r} for operation {ch!r}",
                        position=param_start + j,
                        rule=s,
                    )

            try:
                tokens.append(Token(ch, param))
            except ValueError as exc:
                raise ParseError(str(exc), position=i, rule=s) from exc

            i = param_end

        return tokens

    def try_parse(self, line: str) -> list[Token] | None:
        """Like :meth:`parse` but returns ``None`` instead of raising."""
        try:
            return self.parse(line)
        except ParseError:
            return None

    def serialize(self, tokens: list[Token]) -> str:
        """Serialize *tokens* back to a canonical Hashcat rule string."""
        return "".join(t.serialize() for t in tokens)

    def validate(self, rule: str, *, max_ops: int = MAX_OPS) -> bool:
        """Return ``True`` if *rule* is a syntactically valid Hashcat rule.

        Args:
            rule:    The rule string to validate.
            max_ops: Maximum number of operations allowed (default: 31).
        """
        tokens = self.try_parse(rule)
        if tokens is None:
            return False
        return 1 <= len(tokens) <= max_ops

    def normalize(self, rule: str) -> str | None:
        """Parse *rule* and re-serialize to canonical form.

        Returns ``None`` if the rule is invalid.
        """
        tokens = self.try_parse(rule)
        if tokens is None:
            return None
        return self.serialize(tokens)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def iter_file(
        self,
        path: Path,
        *,
        skip_comments: bool = True,
        skip_invalid: bool = True,
    ) -> Iterator[tuple[int, str, list[Token] | None]]:
        """Iterate over rules in *path*.

        Yields:
            ``(line_number, raw_line, tokens_or_none)`` tuples.
            *tokens_or_none* is ``None`` for invalid lines when
            *skip_invalid* is ``False``.
        """
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for lineno, raw in enumerate(fh, start=1):
                stripped = raw.rstrip("\n")
                s = stripped.strip()

                if not s or (skip_comments and s.startswith("#")):
                    continue

                tokens = self.try_parse(s)
                if tokens is None:
                    if skip_invalid:
                        logger.debug("Skipping invalid rule at line %d: %r", lineno, s)
                        continue
                    yield lineno, stripped, None
                else:
                    yield lineno, stripped, tokens

    def parse_file(self, path: Path, *, preserve_formatting: bool = False) -> list[str]:
        """Return all valid rule strings from *path*.

        Args:
            path:                The rule file to read.
            preserve_formatting: If ``True`` the original whitespace
                                 is preserved; otherwise rules are
                                 re-serialized to canonical form.
        """
        results: list[str] = []
        for _lineno, raw, tokens in self.iter_file(path):
            if tokens is None:
                continue
            results.append(raw if preserve_formatting else self.serialize(tokens))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_comment_or_empty(line: str) -> bool:
        """Return ``True`` if *line* is blank or a comment."""
        s = line.strip()
        return not s or s.startswith("#")

    @staticmethod
    def op_name(cmd: str) -> str:
        """Return a human-readable name for operation *cmd*."""
        _NAMES: dict[str, str] = {
            ":": "noop",
            "l": "lowercase",
            "u": "uppercase",
            "c": "capitalize",
            "C": "lowercase_first_uppercase_rest",
            "t": "toggle_case_all",
            "T": "toggle_case_at",
            "r": "reverse",
            "d": "duplicate",
            "p": "duplicate_n",
            "f": "reflect",
            "{": "rotate_left",
            "}": "rotate_right",
            "$": "append",
            "^": "prepend",
            "[": "delete_first",
            "]": "delete_last",
            "D": "delete_at",
            "x": "extract",
            "O": "omit",
            "i": "insert",
            "o": "overwrite",
            "s": "substitute",
            "@": "purge",
            "z": "duplicate_first",
            "Z": "duplicate_last",
            "q": "duplicate_all",
            "k": "swap_first",
            "K": "swap_last",
            "E": "title_case",
            "P": "pluralize",
            "I": "invert",
            "R": "bitwise_right",
            "M": "memorize",
            "V": "extract_memory",
            "y": "duplicate_first_n",
            "Y": "duplicate_last_n",
        }
        return _NAMES.get(cmd, f"op_{cmd}")


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_default_parser = Parser()


def parse(rule: str) -> list[Token]:
    """Parse *rule* using the module-level default :class:`Parser`."""
    return _default_parser.parse(rule)


def serialize(tokens: list[Token]) -> str:
    """Serialize *tokens* using the module-level default :class:`Parser`."""
    return _default_parser.serialize(tokens)


def validate(rule: str, *, max_ops: int = MAX_OPS) -> bool:
    """Validate *rule* using the module-level default :class:`Parser`."""
    return _default_parser.validate(rule, max_ops=max_ops)
