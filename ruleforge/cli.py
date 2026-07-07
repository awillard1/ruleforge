"""
ruleforge/cli.py
----------------
Typer-based CLI for RuleForge.

Commands:
  analyze     — Analyze rule file(s) and print/export statistics
  generate    — Generate novel Hashcat rules
  learn       — Learn templates and models from rule files
  train       — Train Markov / N-gram / evolutionary models
  evaluate    — Evaluate rules with hashcat --stdout
  benchmark   — Benchmark generation speed
  optimize    — Run evolutionary / Bayesian / MCTS optimization
  resume      — Resume an interrupted generation job
  report      — Generate reports from stored data
  visualize   — Generate charts from analysis data
  database    — Database management utilities
  plugins     — Plugin management utilities
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .parser import Parser
from .analyzer import Analyzer
from .templates import TemplateLearner, FrequencyParamSampler
from .markov import VariableOrderMarkov
from .ngram import NGramEngine
from .generator import RuleGenerator, MixtureWeights, worker_generate
from .runtime import RuntimeEvaluator, WordSampler
from .scoring import Scorer, ScoreWeights
from .database import Database
from .checkpoint import CheckpointManager, Checkpoint
from .reporting import Report, Reporter
from .plugins import get_registry, PluginType

app = typer.Typer(
    name="ruleforge",
    help="Advanced Hashcat rule learning and optimization framework.",
    add_completion=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: int = 0) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# Common options (reused across commands)
# ---------------------------------------------------------------------------

_OPT_VERBOSE = Annotated[int, typer.Option("--verbose", "-v", count=True, help="Verbosity level")]
_OPT_DB = Annotated[Optional[Path], typer.Option("--db", help="SQLite database path")]


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    input_files: Annotated[list[Path], typer.Argument(help="Rule file(s) to analyze")],
    output: Annotated[Path, typer.Option("--output", "-o", help="JSON output path")] = Path("analysis.json"),
    top_cmds: Annotated[int, typer.Option("--top-cmds", help="Top commands to show")] = 20,
    verbose: _OPT_VERBOSE = 0,
    db: _OPT_DB = None,
) -> None:
    """Analyze one or more Hashcat rule files and export statistics."""
    _setup_logging(verbose)
    parser = Parser()
    analyzer = Analyzer(parser)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Analyzing…", total=len(input_files))
        for f in input_files:
            if not f.is_file():
                console.print(f"[red]File not found: {f}[/red]")
                raise typer.Exit(1)
            analyzer.ingest_file(f)
            progress.advance(task)

    result = analyzer.result()
    analyzer.export_json(output)
    console.print(f"[green]Analysis written to {output}[/green]")

    # Print summary table
    tbl = Table(title="Analysis Summary")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Total lines", str(result.total_lines))
    tbl.add_row("Valid rules", str(result.valid_lines))
    tbl.add_row("Unique rules", str(result.unique_count))
    tbl.add_row("Duplicates", str(result.duplicate_count))
    tbl.add_row("Invalid lines", str(result.invalid_lines))
    tbl.add_row("Entropy", f"{result.entropy:.4f}")
    tbl.add_row("Mean complexity", f"{result.mean_complexity:.4f}")
    console.print(tbl)

    # Top commands
    top = sorted(result.cmd_freq.items(), key=lambda kv: kv[1], reverse=True)[:top_cmds]
    cmd_tbl = Table(title=f"Top {top_cmds} Operations")
    cmd_tbl.add_column("Op", style="yellow")
    cmd_tbl.add_column("Name")
    cmd_tbl.add_column("Count", justify="right")
    for op, cnt in top:
        cmd_tbl.add_row(op, parser.op_name(op), str(cnt))
    console.print(cmd_tbl)

    if db:
        database = Database(db)
        database.save_statistics(result.to_dict())
        database.close()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@app.command()
def generate(
    input_files: Annotated[list[Path], typer.Argument(help="Source rule file(s)")],
    output: Annotated[Path, typer.Option("--output", "--out", "-o")] = Path("learned.rule"),
    count: Annotated[int, typer.Option("--count", "-g", help="Target rule count")] = 200_000,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    max_ops: Annotated[int, typer.Option("--max-ops")] = 10,
    mutate_ratio: Annotated[float, typer.Option("--mutate-ratio")] = 0.90,
    allow_fallback: Annotated[bool, typer.Option("--allow-param-fallback")] = False,
    workers: Annotated[int, typer.Option("--workers")] = max(1, (os.cpu_count() or 4) - 1),
    worker_batch: Annotated[int, typer.Option("--worker-batch")] = 2500,
    max_rounds: Annotated[int, typer.Option("--max-rounds")] = 2000,
    runtime_score: Annotated[bool, typer.Option("--runtime-score")] = False,
    hashcat_bin: Annotated[str, typer.Option("--hashcat-bin")] = "hashcat",
    hashcat_timeout: Annotated[int, typer.Option("--hashcat-timeout")] = 30,
    words_file: Annotated[Optional[Path], typer.Option("--words")] = None,
    eval_sample: Annotated[int, typer.Option("--eval-sample")] = 5000,
    report_file: Annotated[Path, typer.Option("--report")] = Path("report.json"),
    score_tsv: Annotated[Path, typer.Option("--score-tsv")] = Path("scores.tsv"),
    errors_log: Annotated[Path, typer.Option("--errors-log")] = Path("errors.log"),
    verbose: _OPT_VERBOSE = 0,
    db: _OPT_DB = None,
) -> None:
    """Generate novel Hashcat rules using learned statistics."""
    _setup_logging(verbose)

    _seed = seed if seed is not None else int(time.time())
    rng = random.Random(_seed)

    errors_log.write_text("", encoding="utf-8")

    def _log_error(msg: str) -> None:
        with errors_log.open("a", encoding="utf-8") as ef:
            ef.write(msg.rstrip() + "\n")

    # --- Analysis ---
    parser = Parser()
    analyzer = Analyzer(parser)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as prog:
        task = prog.add_task("Analyzing source rules…", total=len(input_files))
        for f in input_files:
            if not f.is_file():
                console.print(f"[red]File not found: {f}[/red]")
                raise typer.Exit(1)
            analyzer.ingest_file(f)
            prog.advance(task)

    if not analyzer.unique_rules:
        console.print("[red]No valid source rules found.[/red]")
        raise typer.Exit(2)

    console.print(
        f"Loaded [cyan]{len(analyzer.unique_rules):,}[/cyan] unique source rules."
    )

    # --- Optional runtime scorer ---
    runtime_mode = bool(runtime_score and words_file)
    eval_words: list[str] = []
    word_stats: dict = {}
    baseline: set[str] = set()
    runtime_eval: RuntimeEvaluator | None = None

    if runtime_mode:
        assert words_file is not None
        if not words_file.is_file():
            console.print("[yellow]Words file not found; falling back to offline mode.[/yellow]")
            runtime_mode = False
        else:
            eval_words, word_stats = WordSampler.load_words(
                path=words_file,
                sample_size=eval_sample,
                rng=rng,
                stratified=True,
            )
            if not eval_words:
                runtime_mode = False
            else:
                runtime_eval = RuntimeEvaluator(hashcat_bin, hashcat_timeout)
                console.print(f"Runtime scoring ON | words={len(eval_words):,}")
                with Progress(SpinnerColumn(), TextColumn("Building baseline…"),
                              console=console) as prog:
                    task = prog.add_task("Baseline…", total=len(analyzer.unique_rules))
                    for rule in analyzer.unique_rules:
                        ok, outs, err = runtime_eval.outputs_for_rule(rule, eval_words)
                        if ok:
                            baseline |= outs
                        elif err:
                            _log_error(f"[baseline] {rule!r} → {err}")
                        prog.advance(task)
                console.print(f"Baseline outputs: [cyan]{len(baseline):,}[/cyan]")
    else:
        console.print("Offline mode ON.")

    # --- Worker payload ---
    payload_template: dict = {
        "source_rules": list(analyzer.unique_rules),
        "cmd": dict(analyzer.cmd),
        "start": dict(analyzer.start),
        "end": dict(analyzer.end),
        "trans": {k: dict(v) for k, v in analyzer.trans.items()},
        "params": {k: dict(v) for k, v in analyzer.params.items()},
        "len_dist": dict(analyzer.len_dist),
        "max_ops": max(1, min(31, max_ops)),
        "mutate_ratio": mutate_ratio,
        "batch_size": worker_batch,
        "allow_param_fallback": allow_fallback,
    }

    generated: dict[str, dict] = {}
    attempts_est = 0
    rounds = 0
    worker_errors = 0
    runtime_rejects = 0

    console.print(
        f"Generating [cyan]{count:,}[/cyan] rules | workers={workers} batch={worker_batch}"
    )

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        gen_task = prog.add_task(f"Generated 0/{count:,}", total=count)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            while len(generated) < count and rounds < max_rounds:
                rounds += 1
                futures = []
                for i in range(workers):
                    pl = dict(payload_template)
                    pl["seed"] = _seed + rounds * 100_000 + i * 9973
                    futures.append(ex.submit(worker_generate, pl))

                for fut in as_completed(futures):
                    res = fut.result()
                    if not res.get("ok"):
                        worker_errors += 1
                        _log_error(
                            f"[worker-error] {res.get('error')}\n{res.get('trace', '')}"
                        )
                        continue

                    items = res.get("items", [])
                    attempts_est += int(res.get("tries", 0))

                    for rule, off_score, origin in items:
                        if rule in analyzer.unique_rules or rule in generated:
                            continue

                        if runtime_mode and runtime_eval is not None:
                            ok, outs, err = runtime_eval.outputs_for_rule(rule, eval_words)
                            if not ok or not outs:
                                runtime_rejects += 1
                                if err:
                                    _log_error(f"[runtime-reject] {rule!r} → {err}")
                                continue
                            novset = outs - baseline
                            nov = len(novset)
                            if nov < 5:
                                runtime_rejects += 1
                                continue
                            baseline |= novset
                            score = float(nov)
                        else:
                            score = float(off_score)

                        generated[rule] = {"score": score, "origin": origin, "round": rounds}
                        prog.advance(gen_task)
                        prog.update(gen_task, description=f"Generated {len(generated):,}/{count:,}")

                        if len(generated) >= count:
                            break
                    if len(generated) >= count:
                        break
                if len(generated) >= count:
                    break

    # Sort and write
    ranked = sorted(generated.items(), key=lambda kv: kv[1]["score"], reverse=True)[:count]
    out_rules = [r for r, _ in ranked]

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out_rules:
            fh.write(r + "\n")

    with score_tsv.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("rule\tscore\torigin\tround\n")
        for r, meta in ranked:
            fh.write(f"{r}\t{meta['score']:.6f}\t{meta['origin']}\t{meta['round']}\n")

    analysis_result = analyzer.result()
    report = Report(
        title="RuleForge Generation Report",
        config={
            "generate": count,
            "max_ops": max_ops,
            "mutate_ratio": mutate_ratio,
            "workers": workers,
            "seed": _seed,
        },
        analysis=analysis_result.to_dict(),
        generation={
            "kept_rules": len(out_rules),
            "attempts_estimate": attempts_est,
            "rounds": rounds,
            "worker_errors": worker_errors,
            "runtime_rejects": runtime_rejects,
        },
        top_rules=[
            {"rule": r, "score": meta["score"], "origin": meta["origin"], "round": meta["round"]}
            for r, meta in ranked[:1000]
        ],
        word_sampling=word_stats if runtime_mode else None,
    )
    Reporter(report).write_json(report_file)

    if db:
        database = Database(db)
        database.bulk_insert_rules(
            [(r, meta["score"], meta["origin"]) for r, meta in ranked]
        )
        database.close()

    console.print(f"\n[green]Done.[/green]")
    console.print(f"Rules: [cyan]{output}[/cyan] ({len(out_rules):,})")
    console.print(f"Scores: [cyan]{score_tsv}[/cyan]")
    console.print(f"Report: [cyan]{report_file}[/cyan]")
    console.print(f"Errors: [cyan]{errors_log}[/cyan]")


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------


@app.command()
def learn(
    input_files: Annotated[list[Path], typer.Argument(help="Rule or password file(s) to learn from")],
    templates_out: Annotated[Path, typer.Option("--templates-out")] = Path("templates.json"),
    markov_out: Annotated[Path, typer.Option("--markov-out")] = Path("markov.json"),
    markov_order: Annotated[int, typer.Option("--markov-order")] = 3,
    top_templates: Annotated[int, typer.Option("--top-templates")] = 100,
    ngram: Annotated[int, typer.Option("--ngram", help="N-gram order (0 to disable)")] = 0,
    ngram_out: Annotated[Path, typer.Option("--ngram-out")] = Path("ngram.json"),
    ngram_smoothing: Annotated[str, typer.Option("--ngram-smoothing", help="laplace|backoff|kneser_ney")] = "backoff",
    grammar: Annotated[bool, typer.Option("--grammar/--no-grammar", help="Learn PCFG grammar from input as password corpus")] = False,
    grammar_out: Annotated[Path, typer.Option("--grammar-out")] = Path("grammar.json"),
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Learn templates, Markov, N-gram, and PCFG grammar models from rule or password files."""
    _setup_logging(verbose)

    parser = Parser()
    rules: list[str] = []
    raw_lines: list[str] = []
    for f in input_files:
        if not f.is_file():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        rules.extend(parser.parse_file(f))
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped:
                    raw_lines.append(stripped)

    console.print(f"Loaded [cyan]{len(rules):,}[/cyan] rules, [cyan]{len(raw_lines):,}[/cyan] raw lines.")

    # Templates
    learner = TemplateLearner(parser)
    learner.ingest_rules(rules)
    learner.export_json(templates_out, top_n=top_templates)
    console.print(f"Templates: [cyan]{templates_out}[/cyan]")

    # Markov
    vom = VariableOrderMarkov(max_order=markov_order)
    vom.train(rules, parser)
    vom.save(markov_out)
    console.print(f"Markov model: [cyan]{markov_out}[/cyan]")

    # N-gram (optional)
    if ngram > 0:
        engine = NGramEngine(n=ngram, smoothing=ngram_smoothing)
        engine.train(rules, parser)
        engine.save(ngram_out)
        console.print(f"N-gram model ({ngram}-gram, {ngram_smoothing}): [cyan]{ngram_out}[/cyan]")

    # PCFG grammar (optional) — learns from raw lines treated as password corpus
    if grammar:
        from .grammar import PCFGLearner
        pcfg_learner = PCFGLearner()
        pcfg_learner.learn(raw_lines)
        pcfg = pcfg_learner.build()
        pcfg.save(grammar_out)
        console.print(f"PCFG grammar ({pcfg.total_passwords:,} passwords): [cyan]{grammar_out}[/cyan]")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    input_files: Annotated[list[Path], typer.Argument(help="Rule file(s) to train on")],
    model_type: Annotated[str, typer.Option("--model", help="markov|ngram|evolution")] = "markov",
    model_out: Annotated[Path, typer.Option("--out")] = Path("model.json"),
    markov_order: Annotated[int, typer.Option("--markov-order")] = 2,
    ngram_n: Annotated[int, typer.Option("--ngram-n")] = 3,
    ngram_smoothing: Annotated[str, typer.Option("--ngram-smoothing")] = "backoff",
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Train Markov, N-gram, or evolutionary models."""
    _setup_logging(verbose)

    parser = Parser()
    rules: list[str] = []
    for f in input_files:
        if not f.is_file():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        rules.extend(parser.parse_file(f))

    console.print(f"Training {model_type} on [cyan]{len(rules):,}[/cyan] rules…")

    if model_type == "markov":
        vom = VariableOrderMarkov(max_order=markov_order)
        vom.train(rules, parser)
        vom.save(model_out)
        console.print(f"[green]Markov model saved to {model_out}[/green]")
    elif model_type == "ngram":
        engine = NGramEngine(n=ngram_n, smoothing=ngram_smoothing)
        engine.train(rules, parser)
        engine.save(model_out)
        console.print(f"[green]N-gram model saved to {model_out}[/green]")
    else:
        console.print(f"[red]Unknown model type: {model_type!r}. Use markov|ngram[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    rules_file: Annotated[Path, typer.Argument(help="Rules file to evaluate")],
    words_file: Annotated[Optional[Path], typer.Option("--wordlist", "--words", help="Word list for --stdout evaluation")] = None,
    hash_file: Annotated[Optional[Path], typer.Option("--hash-file", help="Hash file for cracking-mode evaluation")] = None,
    hash_type: Annotated[int, typer.Option("--hash-type", "-m", help="Hashcat hash type (e.g. 0 for MD5)")] = 0,
    hashcat_bin: Annotated[str, typer.Option("--hashcat-bin")] = "hashcat",
    timeout: Annotated[int, typer.Option("--timeout")] = 30,
    top_n: Annotated[int, typer.Option("--top-n")] = 100,
    output: Annotated[Path, typer.Option("--output")] = Path("eval_results.json"),
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Evaluate rules against a word list using hashcat --stdout, or against a hash file."""
    _setup_logging(verbose)

    if words_file is None and hash_file is None:
        console.print("[red]Provide --wordlist for stdout evaluation or --hash-file for cracking evaluation.[/red]")
        raise typer.Exit(1)

    parser = Parser()
    rules = parser.parse_file(rules_file)
    if not rules:
        console.print("[red]No valid rules found.[/red]")
        raise typer.Exit(1)

    results: list[dict] = []

    if hash_file is not None:
        # Cracking-mode evaluation: run hashcat against real hashes
        if not hash_file.is_file():
            console.print(f"[red]Hash file not found: {hash_file}[/red]")
            raise typer.Exit(1)
        if words_file is None or not words_file.is_file():
            console.print("[red]--wordlist is required when using --hash-file.[/red]")
            raise typer.Exit(1)
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".rule", delete=False) as rf:
            for rule in rules[:top_n]:
                rf.write(rule + "\n")
            rule_tmp = Path(rf.name)
        try:
            cmd = [
                hashcat_bin, "-a", "0", "-m", str(hash_type),
                "--potfile-disable", "--quiet",
                "-r", str(rule_tmp),
                str(hash_file), str(words_file),
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * len(rules[:top_n]))
            cracked_lines = [l for l in cp.stdout.splitlines() if ":" in l]
            results.append({
                "mode": "cracking",
                "hash_file": str(hash_file),
                "wordlist": str(words_file),
                "rules_file": str(rules_file),
                "cracked": len(cracked_lines),
                "returncode": cp.returncode,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({"mode": "cracking", "error": str(exc)})
        finally:
            rule_tmp.unlink(missing_ok=True)
    else:
        # stdout evaluation mode
        assert words_file is not None
        if not words_file.is_file():
            console.print(f"[red]Word list not found: {words_file}[/red]")
            raise typer.Exit(1)
        rng = random.Random()
        words, _ = WordSampler.load_words(words_file, sample_size=5000, rng=rng)
        if not words:
            console.print("[red]No words loaded.[/red]")
            raise typer.Exit(1)

        evaluator = RuntimeEvaluator(hashcat_bin=hashcat_bin, timeout_sec=timeout)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), console=console) as prog:
            task = prog.add_task("Evaluating…", total=min(top_n, len(rules)))
            for rule in rules[:top_n]:
                ok, outs, err = evaluator.outputs_for_rule(rule, words)
                results.append({
                    "rule": rule,
                    "outputs": len(outs),
                    "success": ok,
                    "error": err,
                })
                prog.advance(task)

    results.sort(key=lambda r: r["outputs"], reverse=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"[green]Evaluation written to {output}[/green]")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    input_files: Annotated[list[Path], typer.Argument(help="Source rule files")],
    count: Annotated[int, typer.Option("--count")] = 10_000,
    workers: Annotated[int, typer.Option("--workers")] = max(1, (os.cpu_count() or 4) - 1),
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Benchmark rule generation speed."""
    _setup_logging(verbose)

    parser = Parser()
    analyzer = Analyzer(parser)
    for f in input_files:
        analyzer.ingest_file(f)

    if not analyzer.unique_rules:
        console.print("[red]No valid source rules found.[/red]")
        raise typer.Exit(1)

    seed = int(time.time())
    rng = random.Random(seed)
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

    console.print(f"Benchmarking single-threaded generation of {count:,} rules…")
    t0 = time.monotonic()
    rules: list[str] = []
    for _ in range(count):
        r = gen.generate_one()
        if r:
            rules.append(r)
    elapsed = time.monotonic() - t0

    console.print(f"Generated: [cyan]{len(rules):,}[/cyan] valid rules")
    console.print(f"Time: [cyan]{elapsed:.2f}s[/cyan]")
    console.print(f"Rate: [cyan]{len(rules)/elapsed:,.0f}[/cyan] rules/sec")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@app.command()
def optimize(
    input_files: Annotated[list[Path], typer.Argument(help="Source rule files")],
    method: Annotated[str, typer.Option("--method", help="evolution|bayesian|mcts")] = "evolution",
    output: Annotated[Path, typer.Option("--output", "--out")] = Path("optimized.rule"),
    generations: Annotated[int, typer.Option("--generations")] = 50,
    pop_size: Annotated[int, typer.Option("--pop-size", "--population")] = 200,
    target: Annotated[int, typer.Option("--target")] = 1000,
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Optimize rules using evolutionary, Bayesian, or MCTS methods."""
    _setup_logging(verbose)

    parser = Parser()
    analyzer = Analyzer(parser)
    for f in input_files:
        analyzer.ingest_file(f)

    if not analyzer.unique_rules:
        console.print("[red]No valid source rules found.[/red]")
        raise typer.Exit(1)

    rng = random.Random()
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

    fitness_fn = gen.offline_score

    if method == "evolution":
        from .evolution import GeneticOptimizer, EvolutionConfig
        cfg = EvolutionConfig(
            population_size=pop_size,
            max_generations=generations,
            target_rules=target,
        )
        optimizer = GeneticOptimizer(parser, gen, fitness_fn, cfg, rng)
        optimizer.initialize(list(analyzer.unique_rules))
        console.print(f"Running evolutionary optimization ({generations} generations)…")
        optimizer.run()
        rules = optimizer.top_rules(target)

    elif method == "bayesian":
        from .bayesian import BayesianOptimizer, BayesianConfig
        cfg_b = BayesianConfig(n_iterations=generations, batch_size_per_iter=pop_size)
        optimizer_b = BayesianOptimizer(parser, gen, fitness_fn, cfg_b, rng)
        console.print("Running Bayesian optimization…")
        optimizer_b.run()
        rules = optimizer_b.best_rules(target)

    elif method == "mcts":
        from .mcts import MCTSOptimizer, MCTSConfig
        cfg_m = MCTSConfig(n_simulations=generations * pop_size)
        optimizer_m = MCTSOptimizer(parser, gen, fitness_fn, cfg_m, rng)
        console.print("Running MCTS optimization…")
        rules = optimizer_m.search()[:target]

    else:
        console.print(f"[red]Unknown method {method!r}[/red]")
        raise typer.Exit(1)

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rules:
            fh.write(r + "\n")

    console.print(f"[green]Wrote {len(rules):,} rules to {output}[/green]")


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@app.command()
def resume(
    job_id: Annotated[str, typer.Argument(help="Job ID to resume")],
    checkpoint_dir: Annotated[Path, typer.Option("--checkpoint-dir")] = Path("checkpoints"),
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Resume an interrupted generation job from its latest checkpoint."""
    _setup_logging(verbose)
    manager = CheckpointManager(checkpoint_dir)
    cp = manager.load_latest(job_id)
    if cp is None:
        console.print(f"[red]No checkpoint found for job {job_id!r}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]Found checkpoint {cp.checkpoint_id} for job {job_id!r}.[/green]"
    )
    console.print(
        "[yellow]Resume logic: re-run 'generate' with the same config and "
        "--seed from the checkpoint.[/yellow]"
    )
    console.print(json.dumps(cp.config, indent=2))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    report_json: Annotated[Optional[Path], typer.Argument(help="Path to report JSON file (optional if --db is given)")] = None,
    db: Annotated[Optional[Path], typer.Option("--db", help="SQLite database to generate report from")] = None,
    formats: Annotated[
        list[str],
        typer.Option("--format", "-f", help="Output format(s): json csv tsv md html pdf all"),
    ] = ["all"],  # noqa: B006
    output_dir: Annotated[Path, typer.Option("--out-dir")] = Path("reports"),
    out: Annotated[Optional[Path], typer.Option("--out", help="Write a single output file (format inferred from extension)")] = None,
    top_n: Annotated[int, typer.Option("--top-n", help="Top rules to include from database")] = 1000,
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Generate reports from a stored JSON report file or a RuleForge database."""
    _setup_logging(verbose)

    if report_json is None and db is None:
        console.print("[red]Provide a report JSON file or --db to source data from a database.[/red]")
        raise typer.Exit(1)

    if db is not None:
        # Build report from database
        if not db.is_file():
            console.print(f"[red]Database not found: {db}[/red]")
            raise typer.Exit(1)
        database = Database(db)
        top_rules = database.top_rules_by_fitness(top_n)
        database.close()
        r = Report(
            title=f"RuleForge Report — {db.name}",
            config={"source": "database", "db": str(db), "top_n": top_n},
            top_rules=[
                {"rule": row["rule"], "score": row.get("fitness") or 0.0,
                 "origin": row.get("origin") or "db"}
                for row in top_rules
            ],
        )
        stem = db.stem
    else:
        assert report_json is not None
        if not report_json.is_file():
            console.print(f"[red]Report file not found: {report_json}[/red]")
            raise typer.Exit(1)
        data = json.loads(report_json.read_text(encoding="utf-8"))
        r = Report(
            title=data.get("title", "RuleForge Report"),
            timestamp=float(data.get("timestamp", time.time())),
            config=data.get("config", {}),
            analysis=data.get("analysis", {}),
            generation=data.get("generation", {}),
            top_rules=data.get("top_rules", []),
        )
        stem = report_json.stem

    reporter = Reporter(r)

    # Single-file output via --out
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lstrip(".").lower()
        if ext == "json":
            reporter.write_json(out)
        elif ext == "csv":
            reporter.write_csv(out)
        elif ext == "tsv":
            reporter.write_tsv(out)
        elif ext in ("md", "markdown"):
            reporter.write_markdown(out)
        elif ext in ("html", "htm"):
            reporter.write_html(out)
        elif ext == "pdf":
            try:
                reporter.write_pdf(out)
            except ImportError as exc:
                console.print(f"[yellow]PDF skipped: {exc}[/yellow]")
        else:
            # Fall back to the first requested format, or html
            fmt = formats[0] if formats and formats[0] != "all" else "html"
            _write_report_format(reporter, fmt, out)
        console.print(f"[green]Report written to {out}[/green]")
        return

    # Multi-format output to directory
    output_dir.mkdir(parents=True, exist_ok=True)
    do_all = "all" in formats
    if do_all or "json" in formats:
        reporter.write_json(output_dir / f"{stem}.json")
    if do_all or "csv" in formats:
        reporter.write_csv(output_dir / f"{stem}.csv")
    if do_all or "tsv" in formats:
        reporter.write_tsv(output_dir / f"{stem}.tsv")
    if do_all or "md" in formats:
        reporter.write_markdown(output_dir / f"{stem}.md")
    if do_all or "html" in formats:
        reporter.write_html(output_dir / f"{stem}.html")
    if "pdf" in formats:
        try:
            reporter.write_pdf(output_dir / f"{stem}.pdf")
        except ImportError as exc:
            console.print(f"[yellow]PDF skipped: {exc}[/yellow]")

    console.print(f"[green]Reports written to {output_dir}[/green]")


def _write_report_format(reporter: "Reporter", fmt: str, path: Path) -> None:
    """Write a single format to *path*."""
    if fmt == "json":
        reporter.write_json(path)
    elif fmt == "csv":
        reporter.write_csv(path)
    elif fmt == "tsv":
        reporter.write_tsv(path)
    elif fmt in ("md", "markdown"):
        reporter.write_markdown(path)
    elif fmt == "pdf":
        reporter.write_pdf(path)
    else:
        reporter.write_html(path)


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------


@app.command()
def visualize(
    analysis_json: Annotated[Path, typer.Argument(help="Analysis JSON from 'analyze' command")],
    output_dir: Annotated[Path, typer.Option("--out-dir")] = Path("charts"),
    verbose: _OPT_VERBOSE = 0,
) -> None:
    """Generate charts and graphs from an analysis JSON file."""
    _setup_logging(verbose)

    if not analysis_json.is_file():
        console.print(f"[red]File not found: {analysis_json}[/red]")
        raise typer.Exit(1)

    data = json.loads(analysis_json.read_text(encoding="utf-8"))
    from .visualization import Visualizer
    viz = Visualizer(output_dir)
    paths = viz.generate_all(data)
    for p in paths:
        console.print(f"  [cyan]{p}[/cyan]")
    console.print(f"[green]{len(paths)} chart(s) written to {output_dir}[/green]")


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

db_app = typer.Typer(name="database", help="Database management utilities.")
app.add_typer(db_app)


@db_app.command("stats")
def db_stats(
    db_path: Annotated[Path, typer.Argument()] = Path("ruleforge.db"),
) -> None:
    """Print database statistics."""
    database = Database(db_path)
    tbl = Table(title=f"Database: {db_path}")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Total rules", str(database.rule_count()))
    console.print(tbl)
    database.close()


@db_app.command("vacuum")
def db_vacuum(
    db_path: Annotated[Path, typer.Argument()] = Path("ruleforge.db"),
) -> None:
    """VACUUM the database to reclaim space."""
    database = Database(db_path)
    database.vacuum()
    database.close()
    console.print("[green]VACUUM complete.[/green]")


@db_app.command("export")
def db_export(
    output: Annotated[Path, typer.Option("--output")] = Path("exported_rules.rule"),
    top_n: Annotated[int, typer.Option("--top-n")] = 10_000,
    db_path: Annotated[Path, typer.Argument()] = Path("ruleforge.db"),
) -> None:
    """Export top rules from the database."""
    database = Database(db_path)
    rules = database.top_rules_by_fitness(top_n)
    database.close()
    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rules:
            fh.write(row["rule"] + "\n")
    console.print(f"[green]Exported {len(rules):,} rules to {output}[/green]")


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------

plugin_app = typer.Typer(name="plugins", help="Plugin management utilities.")
app.add_typer(plugin_app)


@plugin_app.command("list")
def plugins_list() -> None:
    """List all registered plugins."""
    registry = get_registry()
    tbl = Table(title="Registered Plugins")
    tbl.add_column("Type", style="cyan")
    tbl.add_column("Name")
    for entry in registry.list_plugins():
        tbl.add_row(entry.plugin_type, entry.name)
    console.print(tbl)


@plugin_app.command("load")
def plugins_load(
    path: Annotated[Path, typer.Argument(help="Plugin file to load")],
) -> None:
    """Load plugins from a Python file."""
    registry = get_registry()
    entries = registry.load_from_file(path)
    console.print(f"[green]Loaded {len(entries)} plugin(s).[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
