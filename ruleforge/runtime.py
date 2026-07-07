"""
ruleforge/runtime.py
--------------------
Runtime Evaluation — optional hashcat --stdout integration.

Supports two modes:
- **Offline**: Pure heuristic scoring (no hashcat required).
- **Runtime**: Live hashcat --stdout execution with unique-output counting.

Also provides the word-list sampler from the original hashcat_rules.py,
extended with configurable stratified sampling.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Word Sampler
# ---------------------------------------------------------------------------


class WordSampler:
    """Load and sample words from a file with optional stratification.

    Preserves the original behaviour from ``hashcat_rules.py`` while
    exposing it as a proper class.
    """

    @staticmethod
    def shape_signature(word: str) -> str:
        """Return a compact shape string for *word*."""
        mapping = {True: "U", False: "L"}
        out: list[str] = []
        for ch in word:
            if ch.isupper():
                out.append("U")
            elif ch.islower():
                out.append("L")
            elif ch.isdigit():
                out.append("D")
            else:
                out.append("S")
        return "".join(out)

    @staticmethod
    def len_bucket(n: int) -> str:
        if n <= 4:
            return "01-04"
        if n <= 8:
            return "05-08"
        if n <= 12:
            return "09-12"
        if n <= 16:
            return "13-16"
        return "17+"

    @staticmethod
    def alpha_stem(word: str) -> str:
        s = re.sub(r"[^a-z]", "", word.lower())
        return re.sub(r"(.)\1+", r"\1", s)

    @classmethod
    def signature(cls, word: str) -> str:
        return f"{cls.len_bucket(len(word))}|{cls.shape_signature(word)}"

    @classmethod
    def load_words(
        cls,
        path: Path,
        sample_size: int,
        rng: Any,
        *,
        dedupe_exact: bool = True,
        stratified: bool = True,
        max_per_signature: int = 400,
        max_per_stem: int = 150,
    ) -> tuple[list[str], dict[str, Any]]:
        """Load and sample words from *path*.

        Returns:
            (words, stats_dict)
        """
        words: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    words.append(s)

        raw_count = len(words)
        if raw_count == 0:
            return [], {"raw_count": 0, "selected_count": 0}

        if dedupe_exact:
            seen: set[str] = set()
            deduped: list[str] = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    deduped.append(w)
            words = deduped

        exact_deduped = len(words)

        from collections import defaultdict

        sig_buckets: dict[str, list[str]] = defaultdict(list)
        stem_counter: Counter[tuple[str, str]] = Counter()

        for w in words:
            sig = cls.signature(w)
            stem = cls.alpha_stem(w)
            key = (sig, stem)
            if stem and stem_counter[key] >= max_per_stem:
                continue
            stem_counter[key] += 1
            if len(sig_buckets[sig]) < max_per_signature:
                sig_buckets[sig].append(w)

        if not stratified:
            flat = [w for arr in sig_buckets.values() for w in arr]
            if sample_size > 0 and len(flat) > sample_size:
                flat = rng.sample(flat, sample_size)
            return flat, {
                "raw_count": raw_count,
                "exact_deduped_count": exact_deduped,
                "signature_groups": len(sig_buckets),
                "selected_count": len(flat),
                "stratified": False,
            }

        keys = list(sig_buckets.keys())
        rng.shuffle(keys)
        idx: dict[str, int] = {k: 0 for k in keys}
        selected: list[str] = []
        target = sample_size if sample_size > 0 else sum(len(sig_buckets[k]) for k in keys)

        while len(selected) < target:
            progressed = False
            for k in keys:
                i = idx[k]
                if i < len(sig_buckets[k]):
                    selected.append(sig_buckets[k][i])
                    idx[k] += 1
                    progressed = True
                    if len(selected) >= target:
                        break
            if not progressed:
                break

        return selected, {
            "raw_count": raw_count,
            "exact_deduped_count": exact_deduped,
            "signature_groups": len(sig_buckets),
            "selected_count": len(selected),
            "stratified": True,
            "max_per_signature": max_per_signature,
            "max_per_stem": max_per_stem,
        }


# ---------------------------------------------------------------------------
# RuntimeEvaluator
# ---------------------------------------------------------------------------


class RuntimeEvaluator:
    """Optional hashcat --stdout novelty scoring.

    Spawns hashcat as a subprocess to measure unique outputs for each rule.
    Results are cached to avoid re-running the same rule twice.

    Args:
        hashcat_bin:  Path to the hashcat binary (default: ``"hashcat"``).
        timeout_sec:  Per-rule subprocess timeout in seconds.
    """

    def __init__(
        self,
        hashcat_bin: str = "hashcat",
        timeout_sec: int = 30,
    ) -> None:
        self._bin = hashcat_bin
        self._timeout = timeout_sec
        self._cache: dict[str, set[str]] = {}
        self._call_count: int = 0
        self._error_count: int = 0

    def outputs_for_rule(
        self,
        rule: str,
        words: list[str],
    ) -> tuple[bool, set[str], str]:
        """Run ``hashcat --stdout`` for *rule* against *words*.

        Args:
            rule:  A single Hashcat rule string.
            words: Word list to apply the rule to.

        Returns:
            ``(success, outputs, error_message)``
        """
        if rule in self._cache:
            return True, self._cache[rule], ""

        rule_file: Path | None = None
        word_file: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".rule", delete=False
            ) as rf:
                rf.write(rule + "\n")
                rule_file = Path(rf.name)

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".txt", delete=False
            ) as wf:
                for w in words:
                    wf.write(w + "\n")
                word_file = Path(wf.name)

            cmd = [self._bin, "--stdout", "-r", str(rule_file), str(word_file)]
            t0 = time.monotonic()
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            elapsed = time.monotonic() - t0
            self._call_count += 1

            if cp.returncode != 0:
                self._error_count += 1
                err = (cp.stderr or "").strip()
                logger.debug("hashcat error for rule %r: %s", rule, err)
                return False, set(), err

            outputs: set[str] = {
                line.strip() for line in cp.stdout.splitlines() if line.strip()
            }
            self._cache[rule] = outputs
            logger.debug(
                "hashcat: rule=%r outputs=%d elapsed=%.2fs",
                rule, len(outputs), elapsed,
            )
            return True, outputs, ""

        except subprocess.TimeoutExpired:
            self._error_count += 1
            return False, set(), "timeout"
        except FileNotFoundError:
            return False, set(), f"hashcat binary not found: {self._bin!r}"
        except Exception as exc:  # noqa: BLE001
            self._error_count += 1
            return False, set(), str(exc)
        finally:
            if rule_file:
                rule_file.unlink(missing_ok=True)
            if word_file:
                word_file.unlink(missing_ok=True)

    def novelty(
        self,
        rule: str,
        words: list[str],
        baseline: set[str],
    ) -> int:
        """Return number of outputs for *rule* not already in *baseline*."""
        ok, outputs, _ = self.outputs_for_rule(rule, words)
        if not ok:
            return 0
        return len(outputs - baseline)

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
