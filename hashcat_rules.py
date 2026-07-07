#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Hashcat Rule Learner/Generator (offline + optional runtime scoring)

Key goals:
- Fast generation with multiprocessing
- No mandatory hashcat dependency (offline mode by default)
- Optional hashcat --stdout coverage scoring
- Progress bars and robust logging
- Wordlist dedupe + stratified sampling (for better representative scoring)

Works on Windows (uses spawn-safe multiprocessing).
Python 3.9+ recommended.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================
# Models / Specs
# =========================

@dataclass(frozen=True)
class Tok:
    cmd: str
    param: str = ""

    def __post_init__(self):
        if not isinstance(self.cmd, str) or not isinstance(self.param, str):
            raise TypeError("Tok fields must be strings.")
        if len(self.cmd) != 1:
            raise ValueError("Tok.cmd must be exactly 1 char.")


class Spec:
    # Conservative subset
    NO_PARAM = set(":lucCtrdpf{}[]qkKEPIRMV")
    ONE_PARAM = set("$^TDi@zZyY")
    TWO_PARAM = set("so")
    THREE_PARAM = set("iO")

    @classmethod
    def arity(cls, c: str) -> int:
        if c in cls.NO_PARAM:
            return 0
        if c in cls.ONE_PARAM:
            return 1
        if c in cls.TWO_PARAM:
            return 2
        if c in cls.THREE_PARAM:
            return 3
        return -1

    @staticmethod
    def ok_char(ch: str) -> bool:
        if not isinstance(ch, str) or len(ch) != 1:
            return False
        o = ord(ch)
        return 32 <= o <= 126 and ch not in ("\r", "\n", "\t")


class Parser:
    def parse(self, line: str) -> List[Tok]:
        if not isinstance(line, str):
            return []
        s = line.strip()
        if not s or s.startswith("#"):
            return []

        out: List[Tok] = []
        i = 0
        n = len(s)

        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
                continue

            ar = Spec.arity(c)
            if ar < 0:
                return []

            i += 1
            if i + ar > n:
                return []

            p = s[i:i + ar] if ar else ""
            i += ar

            if len(p) != ar:
                return []
            if any(not Spec.ok_char(ch) for ch in p):
                return []

            try:
                out.append(Tok(c, p))
            except Exception:
                return []

        return out

    def to_rule(self, toks: List[Tok]) -> str:
        try:
            return "".join(t.cmd + t.param for t in toks)
        except Exception:
            return ""

    def valid(self, rule: str, max_ops: int = 31) -> bool:
        toks = self.parse(rule)
        return bool(toks) and len(toks) <= max_ops


# =========================
# Analysis
# =========================

class Analyzer:
    def __init__(self, parser: Parser):
        self.p = parser

        self.total_lines = 0
        self.valid_lines = 0
        self.invalid_lines = 0
        self.comment_or_empty = 0

        self.unique_rules: Set[str] = set()

        self.cmd = Counter()
        self.start = Counter()
        self.end = Counter()
        self.trans = defaultdict(Counter)   # cmdA -> Counter(cmdB)
        self.params = defaultdict(Counter)  # cmd -> Counter(param)
        self.len_dist = Counter()

    def ingest_file(self, path: Path):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                self.total_lines += 1
                s = raw.strip()

                if not s or s.startswith("#"):
                    self.comment_or_empty += 1
                    continue

                toks = self.p.parse(s)
                if not toks:
                    self.invalid_lines += 1
                    continue

                r = self.p.to_rule(toks)
                if not r or not self.p.valid(r):
                    self.invalid_lines += 1
                    continue

                self.valid_lines += 1
                self.unique_rules.add(r)

        # Build statistics from unique valid rules
        for r in self.unique_rules:
            toks = self.p.parse(r)
            if not toks:
                continue

            self.len_dist[len(toks)] += 1
            self.start[toks[0].cmd] += 1
            self.end[toks[-1].cmd] += 1

            for t in toks:
                self.cmd[t.cmd] += 1
                if t.param:
                    self.params[t.cmd][t.param] += 1

            for i in range(len(toks) - 1):
                self.trans[toks[i].cmd][toks[i + 1].cmd] += 1


def wchoice(counter: Counter, rng: random.Random, fallback=None):
    if not counter:
        return fallback
    items = list(counter.items())
    total = sum(v for _, v in items)
    if total <= 0:
        return fallback
    r = rng.uniform(0, total)
    c = 0.0
    for k, w in items:
        c += w
        if c >= r:
            return k
    return items[-1][0]


# =========================
# Word Sampling
# =========================

class WordSampler:
    @staticmethod
    def shape_signature(w: str) -> str:
        out = []
        for ch in w:
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
    def alpha_stem(w: str) -> str:
        s = re.sub(r"[^a-z]", "", w.lower())
        return re.sub(r"(.)\1+", r"\1", s)

    @classmethod
    def signature(cls, w: str) -> str:
        return f"{cls.len_bucket(len(w))}|{cls.shape_signature(w)}"

    @classmethod
    def load_words(
        cls,
        path: Path,
        sample_size: int,
        rng: random.Random,
        dedupe_exact: bool = True,
        stratified: bool = True,
        max_per_signature: int = 400,
        max_per_stem: int = 150
    ) -> Tuple[List[str], Dict]:
        words = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    words.append(s)

        raw_count = len(words)
        if raw_count == 0:
            return [], {"raw_count": 0, "selected_count": 0}

        if dedupe_exact:
            seen = set()
            ded = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    ded.append(w)
            words = ded

        exact_deduped = len(words)

        sig_buckets = defaultdict(list)
        stem_counter = Counter()

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
                "stratified": False
            }

        keys = list(sig_buckets.keys())
        rng.shuffle(keys)
        idx = {k: 0 for k in keys}
        selected = []
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
            "max_per_stem": max_per_stem
        }


# =========================
# Generator + Offline scoring
# =========================

class StrictGenerator:
    def __init__(self, a: Analyzer, p: Parser, rng: random.Random, allow_param_fallback: bool = False):
        self.a = a
        self.p = p
        self.rng = rng
        self.src = list(a.unique_rules)

        self.allowed_cmds = set(a.cmd.keys())
        self.allow_param_fallback = allow_param_fallback

        self.safe1 = list("0123456789!@#$%^&*.-_")
        self.safe_pos = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def sample_len(self, max_ops: int) -> int:
        n = int(wchoice(self.a.len_dist, self.rng, fallback=4))
        return max(1, min(max_ops, n))

    def sample_start(self) -> str:
        return wchoice(self.a.start, self.rng, fallback=":")

    def sample_next(self, prev: str) -> str:
        return wchoice(self.a.trans.get(prev, Counter()), self.rng,
                       fallback=wchoice(self.a.cmd, self.rng, fallback=":"))

    def sample_param(self, cmd: str) -> Optional[str]:
        ar = Spec.arity(cmd)
        if ar == 0:
            return ""

        pool = self.a.params.get(cmd)
        if pool:
            p = wchoice(pool, self.rng, fallback=None)
            if isinstance(p, str) and len(p) == ar and all(Spec.ok_char(ch) for ch in p):
                return p

        if not self.allow_param_fallback:
            return None

        if ar == 1:
            return self.rng.choice(self.safe1)
        if ar == 2:
            return self.rng.choice(self.safe1) + self.rng.choice(self.safe1)
        if ar == 3:
            return self.rng.choice(self.safe_pos) + self.rng.choice(self.safe1) + self.rng.choice(self.safe1)
        return None

    def is_valid_candidate(self, rule: str, max_ops: int = 31) -> bool:
        if not self.p.valid(rule, max_ops=max_ops):
            return False
        toks = self.p.parse(rule)
        if not toks:
            return False
        for t in toks:
            if t.cmd not in self.allowed_cmds:
                return False
            ar = Spec.arity(t.cmd)
            if ar < 0 or len(t.param) != ar:
                return False
            if any(not Spec.ok_char(ch) for ch in t.param):
                return False
        return True

    def build_markov(self, max_ops: int = 12) -> str:
        n = self.sample_len(max_ops)
        s = self.sample_start()
        if s not in self.allowed_cmds:
            return ""
        cmds = [s]
        while len(cmds) < n:
            c = self.sample_next(cmds[-1])
            if c in self.allowed_cmds:
                cmds.append(c)
            else:
                break

        toks = []
        for c in cmds:
            p = self.sample_param(c)
            if p is None:
                return ""
            toks.append(Tok(c, p))
        r = self.p.to_rule(toks)
        return r if self.is_valid_candidate(r) else ""

    def mutate(self, base: str, max_ops: int = 31) -> str:
        toks = self.p.parse(base)
        if not toks:
            return self.build_markov(min(12, max_ops))

        op = self.rng.choices(
            ["replace_param", "replace_cmd", "insert", "delete", "swap"],
            weights=[0.36, 0.24, 0.18, 0.12, 0.10],
            k=1
        )[0]

        t = toks[:]

        if op == "replace_param":
            idx = [i for i, x in enumerate(t) if Spec.arity(x.cmd) > 0]
            if idx:
                i = self.rng.choice(idx)
                p = self.sample_param(t[i].cmd)
                if p is not None:
                    t[i] = Tok(t[i].cmd, p)

        elif op == "replace_cmd":
            i = self.rng.randrange(len(t))
            prev = t[i - 1].cmd if i > 0 else None
            c = self.sample_next(prev) if prev else self.sample_start()
            if c in self.allowed_cmds:
                p = self.sample_param(c)
                if p is not None:
                    t[i] = Tok(c, p)

        elif op == "insert":
            if len(t) < max_ops:
                i = self.rng.randint(0, len(t))
                prev = t[i - 1].cmd if i > 0 else None
                c = self.sample_next(prev) if prev else self.sample_start()
                if c in self.allowed_cmds:
                    p = self.sample_param(c)
                    if p is not None:
                        t.insert(i, Tok(c, p))

        elif op == "delete":
            if len(t) > 1:
                del t[self.rng.randrange(len(t))]

        elif op == "swap":
            if len(t) > 1:
                i = self.rng.randint(0, len(t) - 2)
                t[i], t[i + 1] = t[i + 1], t[i]

        r = self.p.to_rule(t)
        return r if self.is_valid_candidate(r, max_ops=max_ops) else ""

    # offline heuristic novelty score (no hashcat needed)
    def offline_score(self, rule: str) -> float:
        toks = self.p.parse(rule)
        if not toks:
            return -1e9

        # shape novelty against known rules
        shape = "".join(t.cmd for t in toks)
        shape_freq = 0
        # cheap estimate from transitions + starts
        if toks:
            shape_freq += self.a.start[toks[0].cmd]
            for i in range(len(toks) - 1):
                shape_freq += self.a.trans[toks[i].cmd][toks[i + 1].cmd]

        rarity_bonus = 0.0
        for t in toks:
            if t.param:
                c = self.a.params[t.cmd]
                denom = sum(c.values()) + 1
                freq = c[t.param] + 1
                rarity_bonus += -math.log(freq / denom)

        length_bonus = 0.0
        n = len(toks)
        if 2 <= n <= 8:
            length_bonus += 0.5
        elif n > 12:
            length_bonus -= 0.5

        repetitive_penalty = 0.0
        cmd_counts = Counter([t.cmd for t in toks])
        max_rep = max(cmd_counts.values()) if cmd_counts else 0
        if max_rep >= 4:
            repetitive_penalty -= 0.6

        commonness_penalty = math.log(shape_freq + 1.0) * 0.2
        score = rarity_bonus + length_bonus + repetitive_penalty - commonness_penalty
        return score


# =========================
# Runtime evaluator (optional)
# =========================

class RuntimeEvaluator:
    """
    Optional hashcat --stdout novelty scoring.
    Use only when hashcat is free.
    """

    def __init__(self, hashcat_bin: str = "hashcat", timeout_sec: int = 30):
        self.hashcat_bin = hashcat_bin
        self.timeout_sec = timeout_sec
        self.cache: Dict[str, Set[str]] = {}

    def outputs_for_rule(self, rule: str, words: List[str]) -> Tuple[bool, Set[str], str]:
        if rule in self.cache:
            return True, self.cache[rule], ""

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as rf:
            rf.write(rule + "\n")
            rpath = rf.name
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as wf:
            for w in words:
                wf.write(w + "\n")
            wpath = wf.name

        try:
            cmd = [self.hashcat_bin, "--stdout", "-r", rpath, wpath]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
            if cp.returncode != 0:
                return False, set(), (cp.stderr or "").strip()
            outs = set(x.strip() for x in cp.stdout.splitlines() if x.strip())
            self.cache[rule] = outs
            return True, outs, ""
        except Exception as e:
            return False, set(), str(e)
        finally:
            Path(rpath).unlink(missing_ok=True)
            Path(wpath).unlink(missing_ok=True)


# =========================
# Worker payload
# =========================

def worker_generate_and_score(payload: Dict) -> Dict:
    """
    Worker-safe function (must be top-level for spawn).
    Generates a local batch and scores candidates offline.
    """
    try:
        seed = payload["seed"]
        source_rules = payload["source_rules"]
        cmd = payload["cmd"]
        start = payload["start"]
        end = payload["end"]
        trans = payload["trans"]
        params = payload["params"]
        len_dist = payload["len_dist"]
        max_ops = payload["max_ops"]
        mutate_ratio = payload["mutate_ratio"]
        batch_size = payload["batch_size"]
        allow_param_fallback = payload["allow_param_fallback"]

        # rebuild light analyzer-like structure
        class ALike:
            pass

        a = ALike()
        a.unique_rules = set(source_rules)
        a.cmd = Counter(cmd)
        a.start = Counter(start)
        a.end = Counter(end)
        a.trans = defaultdict(Counter, {k: Counter(v) for k, v in trans.items()})
        a.params = defaultdict(Counter, {k: Counter(v) for k, v in params.items()})
        a.len_dist = Counter(len_dist)

        p = Parser()
        rng = random.Random(seed)
        g = StrictGenerator(a, p, rng, allow_param_fallback=allow_param_fallback)

        out = []
        tries = 0
        max_tries = batch_size * 12

        while len(out) < batch_size and tries < max_tries:
            tries += 1
            if rng.random() < mutate_ratio and g.src:
                cand = g.mutate(rng.choice(g.src), max_ops=max_ops)
                origin = "mutate"
            else:
                cand = g.build_markov(max_ops=max_ops)
                origin = "markov"

            if not cand:
                continue
            if cand in a.unique_rules:
                continue

            score = g.offline_score(cand)
            out.append((cand, score, origin))

        return {"ok": True, "items": out, "tries": tries}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


# =========================
# Utilities
# =========================

def progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="it", dynamic_ncols=True)
    return None


def parse_input_files(inputs: List[str]) -> List[Path]:
    files = []
    for x in inputs:
        p = Path(x)
        if p.is_file():
            files.append(p)
            continue
        for gp in Path(".").glob(x):
            if gp.is_file():
                files.append(gp)

    seen = set()
    out = []
    for f in files:
        rf = str(f.resolve())
        if rf not in seen:
            seen.add(rf)
            out.append(f)
    return out


# =========================
# Main
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel hashcat rule learner/generator (offline-first).")
    ap.add_argument("-i", "--input", nargs="+", required=True, help="Input rule files or globs")
    ap.add_argument("-o", "--output", default="learned_parallel.rule", help="Output rules file")
    ap.add_argument("-g", "--generate", type=int, default=200000, help="Target number of generated novel rules")

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-ops", type=int, default=10)
    ap.add_argument("--mutate-ratio", type=float, default=0.90)
    ap.add_argument("--allow-param-fallback", action="store_true")

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--worker-batch", type=int, default=2500, help="Candidates per worker task")
    ap.add_argument("--max-rounds", type=int, default=2000, help="Safety cap")

    # runtime scoring mode (optional; off by default)
    ap.add_argument("--runtime-score", action="store_true",
                    help="Use hashcat --stdout novelty scoring (slow). Off by default.")
    ap.add_argument("--hashcat-bin", type=str, default="hashcat")
    ap.add_argument("--hashcat-timeout", type=int, default=30)
    ap.add_argument("--words", type=str, default=None)

    # word sample options
    ap.add_argument("--eval-sample", type=int, default=5000)
    ap.add_argument("--dedupe-words", action="store_true")
    ap.add_argument("--stratified-sample", action="store_true")
    ap.add_argument("--max-per-signature", type=int, default=400)
    ap.add_argument("--max-per-stem", type=int, default=150)
    ap.add_argument("--min-novel-outputs", type=int, default=5)

    ap.add_argument("--report", default="parallel_rulegen_report.json")
    ap.add_argument("--score-tsv", default="parallel_rulegen_scores.tsv")
    ap.add_argument("--errors-log", default="parallel_rulegen_errors.log")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    err_log = Path(args.errors_log)
    err_log.write_text("", encoding="utf-8")

    def log_error(msg: str):
        with err_log.open("a", encoding="utf-8") as ef:
            ef.write(msg.rstrip() + "\n")

    parser = Parser()
    analyzer = Analyzer(parser)

    files = parse_input_files(args.input)
    if not files:
        print("No input files found.")
        return 1

    print(f"Analyzing {len(files)} input file(s)...")
    p_an = progress(len(files), "Analyze files")
    for f in files:
        analyzer.ingest_file(f)
        if p_an:
            p_an.update(1)
    if p_an:
        p_an.close()

    if not analyzer.unique_rules:
        print("No valid source rules found.")
        return 2

    # Optional runtime scorer setup
    runtime_mode = bool(args.runtime_score and args.words)
    words = []
    word_stats = {}
    runtime_eval = None
    baseline = set()

    if runtime_mode:
        wp = Path(args.words)
        if not wp.is_file():
            print("Runtime scoring requested but words file not found. Falling back to offline mode.")
            runtime_mode = False
        else:
            words, word_stats = WordSampler.load_words(
                path=wp,
                sample_size=args.eval_sample,
                rng=rng,
                dedupe_exact=args.dedupe_words,
                stratified=args.stratified_sample,
                max_per_signature=args.max_per_signature,
                max_per_stem=args.max_per_stem
            )
            if not words:
                print("Runtime scoring requested but sampled words is empty. Falling back to offline mode.")
                runtime_mode = False
            else:
                runtime_eval = RuntimeEvaluator(args.hashcat_bin, args.hashcat_timeout)
                print(f"Runtime scoring ON | sample words={len(words)}")
                # baseline one-shot per rule (cached across source rules)
                p_base = progress(len(analyzer.unique_rules), "Baseline scoring")
                for r in analyzer.unique_rules:
                    ok, outs, err = runtime_eval.outputs_for_rule(r, words)
                    if ok:
                        baseline |= outs
                    elif err:
                        log_error(f"[baseline] rule={r!r} err={err}")
                    if p_base:
                        p_base.update(1)
                if p_base:
                    p_base.close()
                print(f"Baseline outputs: {len(baseline)}")
    else:
        print("Offline mode ON (no hashcat runtime scoring).")

    # Shared immutable payload for workers
    payload_template = {
        "source_rules": list(analyzer.unique_rules),
        "cmd": dict(analyzer.cmd),
        "start": dict(analyzer.start),
        "end": dict(analyzer.end),
        "trans": {k: dict(v) for k, v in analyzer.trans.items()},
        "params": {k: dict(v) for k, v in analyzer.params.items()},
        "len_dist": dict(analyzer.len_dist),
        "max_ops": max(1, min(31, args.max_ops)),
        "mutate_ratio": args.mutate_ratio,
        "batch_size": args.worker_batch,
        "allow_param_fallback": args.allow_param_fallback
    }

    generated: Dict[str, Dict] = {}
    attempts_est = 0
    rounds = 0
    worker_errors = 0
    runtime_rejects = 0

    target = args.generate
    max_rounds = args.max_rounds

    print(f"Generating target={target:,} with workers={args.workers}, worker_batch={args.worker_batch}")
    p_gen = progress(target, "Keep novel rules")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        while len(generated) < target and rounds < max_rounds:
            rounds += 1

            futures = []
            for i in range(args.workers):
                pl = dict(payload_template)
                pl["seed"] = seed + rounds * 100000 + i * 9973
                futures.append(ex.submit(worker_generate_and_score, pl))

            for fut in as_completed(futures):
                res = fut.result()
                if not res.get("ok"):
                    worker_errors += 1
                    log_error(f"[worker-error] {res.get('error')}\n{res.get('trace')}")
                    continue

                items = res.get("items", [])
                attempts_est += int(res.get("tries", 0))

                for rule, off_score, origin in items:
                    if rule in analyzer.unique_rules or rule in generated:
                        continue

                    if runtime_mode and runtime_eval is not None:
                        ok, outs, err = runtime_eval.outputs_for_rule(rule, words)
                        if not ok or not outs:
                            runtime_rejects += 1
                            if err:
                                log_error(f"[runtime-reject] rule={rule!r} err={err}")
                            continue
                        novset = outs - baseline
                        nov = len(novset)
                        if nov < args.min_novel_outputs:
                            runtime_rejects += 1
                            continue
                        baseline |= novset
                        score = float(nov)
                    else:
                        # offline mode score from worker
                        score = float(off_score)

                    generated[rule] = {
                        "score": score,
                        "origin": origin,
                        "round": rounds
                    }

                    if p_gen:
                        p_gen.update(1)

                    if len(generated) >= target:
                        break

                if len(generated) >= target:
                    break

            if len(generated) >= target:
                break

    if p_gen:
        p_gen.close()

    # Final sorting
    ranked = sorted(generated.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top = ranked[:target]

    out_rules = [r for r, _ in top]

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rules:
            f.write(r + "\n")

    with open(args.score_tsv, "w", encoding="utf-8", newline="\n") as f:
        f.write("rule\tscore\torigin\tround\n")
        for r, meta in top:
            f.write(f"{r}\t{meta['score']:.6f}\t{meta['origin']}\t{meta['round']}\n")

    report = {
        "inputs": [str(x) for x in files],
        "seed": seed,
        "settings": {
            "generate": args.generate,
            "max_ops": args.max_ops,
            "mutate_ratio": args.mutate_ratio,
            "allow_param_fallback": args.allow_param_fallback,
            "workers": args.workers,
            "worker_batch": args.worker_batch,
            "max_rounds": args.max_rounds,
            "runtime_score": runtime_mode
        },
        "analysis": {
            "total_lines": analyzer.total_lines,
            "valid_lines": analyzer.valid_lines,
            "invalid_lines": analyzer.invalid_lines,
            "comment_or_empty": analyzer.comment_or_empty,
            "unique_source_rules": len(analyzer.unique_rules),
            "top_commands": analyzer.cmd.most_common(25),
            "top_lengths": analyzer.len_dist.most_common(15)
        },
        "runtime_word_sampling": word_stats if runtime_mode else None,
        "generation": {
            "kept_rules": len(out_rules),
            "attempts_estimate": attempts_est,
            "rounds": rounds,
            "worker_errors": worker_errors,
            "runtime_rejects": runtime_rejects,
            "baseline_outputs_final": len(baseline) if runtime_mode else None
        },
        "outputs": {
            "rules_file": args.output,
            "score_tsv": args.score_tsv,
            "errors_log": args.errors_log,
            "report_file": args.report
        }
    }

    with open(args.report, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Wrote rules: {args.output} ({len(out_rules):,})")
    print(f"Wrote scores: {args.score_tsv}")
    print(f"Wrote report: {args.report}")
    print(f"Wrote errors: {args.errors_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())