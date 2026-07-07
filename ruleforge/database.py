"""
ruleforge/database.py
---------------------
SQLite persistence layer.

Tables:
  rules       — every generated/evaluated rule with metadata
  templates   — learned rule templates
  statistics  — analyzer snapshot statistics
  runtime     — hashcat runtime evaluation results
  coverage    — coverage tracking per rule set
  fitness     — fitness score history
  models      — serialized model blobs
  jobs        — generation job records
  history     — execution history
  passwords   — analyzed password statistics
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS rules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rule            TEXT    NOT NULL UNIQUE,
    times_generated INTEGER NOT NULL DEFAULT 0,
    times_evaluated INTEGER NOT NULL DEFAULT 0,
    runtime_ms      REAL,
    coverage        REAL,
    duplicates      INTEGER NOT NULL DEFAULT 0,
    fitness         REAL,
    last_used       REAL,
    best_score      REAL,
    origin          TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS templates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signature   TEXT    NOT NULL UNIQUE,
    steps_json  TEXT    NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    score       REAL,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS statistics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          INTEGER,
    total_lines     INTEGER,
    valid_lines     INTEGER,
    invalid_lines   INTEGER,
    unique_count    INTEGER,
    entropy         REAL,
    mean_complexity REAL,
    cmd_freq_json   TEXT,
    len_dist_json   TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS runtime (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     INTEGER NOT NULL REFERENCES rules(id),
    outputs     INTEGER NOT NULL DEFAULT 0,
    novel       INTEGER NOT NULL DEFAULT 0,
    runtime_ms  REAL,
    evaluated_at REAL   NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS coverage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER,
    rule_set    TEXT,
    coverage    REAL,
    total_words INTEGER,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS fitness (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     INTEGER NOT NULL REFERENCES rules(id),
    score       REAL    NOT NULL,
    method      TEXT,
    generation  INTEGER,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    model_type  TEXT    NOT NULL,
    data_json   TEXT    NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    updated_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    status      TEXT    NOT NULL DEFAULT 'pending',
    config_json TEXT,
    started_at  REAL,
    finished_at REAL,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER REFERENCES jobs(id),
    event       TEXT    NOT NULL,
    detail_json TEXT,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS passwords (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          INTEGER,
    total           INTEGER,
    stats_json      TEXT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE INDEX IF NOT EXISTS idx_rules_rule ON rules(rule);
CREATE INDEX IF NOT EXISTS idx_rules_fitness ON rules(fitness);
CREATE INDEX IF NOT EXISTS idx_fitness_rule_id ON fitness(rule_id);
"""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """SQLite-backed persistence layer.

    Args:
        path: Path to the SQLite database file.
              Use ``":memory:"`` for an in-memory database.
    """

    def __init__(self, path: str | Path = "ruleforge.db") -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Database opened: %s", self._path)

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def upsert_rule(
        self,
        rule: str,
        *,
        fitness: float | None = None,
        origin: str | None = None,
        coverage: float | None = None,
        runtime_ms: float | None = None,
    ) -> int:
        """Insert or update a rule record. Returns the rule id."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO rules (rule, fitness, origin, coverage, runtime_ms,
                                   times_generated, last_used)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(rule) DO UPDATE SET
                    times_generated = times_generated + 1,
                    fitness = COALESCE(?, fitness),
                    origin  = COALESCE(?, origin),
                    coverage = COALESCE(?, coverage),
                    runtime_ms = COALESCE(?, runtime_ms),
                    best_score = MAX(COALESCE(best_score, -1e9), COALESCE(?, -1e9)),
                    last_used = ?
                """,
                (
                    rule, fitness, origin, coverage, runtime_ms, time.time(),
                    fitness, origin, coverage, runtime_ms, fitness, time.time(),
                ),
            )
            cur.execute("SELECT id FROM rules WHERE rule = ?", (rule,))
            row = cur.fetchone()
            return int(row["id"]) if row else -1

    def get_rule(self, rule: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM rules WHERE rule = ?", (rule,))
            row = cur.fetchone()
            return dict(row) if row else None

    def top_rules_by_fitness(self, n: int = 100) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM rules WHERE fitness IS NOT NULL "
                "ORDER BY fitness DESC LIMIT ?",
                (n,),
            )
            return [dict(row) for row in cur.fetchall()]

    def bulk_insert_rules(self, rules: list[tuple[str, float, str]]) -> None:
        """Bulk insert ``(rule, fitness, origin)`` tuples."""
        now = time.time()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rules (rule, fitness, origin, times_generated, last_used)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(rule) DO UPDATE SET
                    times_generated = times_generated + 1,
                    fitness = COALESCE(?, fitness),
                    best_score = MAX(COALESCE(best_score, -1e9), COALESCE(?, -1e9)),
                    last_used = ?
                """,
                [
                    (rule, fit, orig, now, fit, fit, now)
                    for rule, fit, orig in rules
                ],
            )

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def upsert_template(self, signature: str, steps: list[Any], score: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO templates (signature, steps_json, count, score)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    count = count + 1,
                    score = ?
                """,
                (signature, json.dumps(steps), score, score),
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def save_statistics(self, stats: dict[str, Any], job_id: int | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO statistics
                    (job_id, total_lines, valid_lines, invalid_lines,
                     unique_count, entropy, mean_complexity,
                     cmd_freq_json, len_dist_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    stats.get("total_lines"),
                    stats.get("valid_lines"),
                    stats.get("invalid_lines"),
                    stats.get("unique_count"),
                    stats.get("entropy"),
                    stats.get("mean_complexity"),
                    json.dumps(stats.get("cmd_freq", {})),
                    json.dumps(stats.get("len_dist", {})),
                ),
            )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(self, name: str, config: dict[str, Any]) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (name, status, config_json, started_at) VALUES (?, 'running', ?, ?)",
                (name, json.dumps(config), time.time()),
            )
            return cur.lastrowid or -1

    def finish_job(self, job_id: int, status: str = "done") -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ?",
                (status, time.time(), job_id),
            )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def save_model(self, name: str, model_type: str, data: dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO models (name, model_type, data_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    data_json = ?,
                    version = version + 1,
                    updated_at = ?
                """,
                (
                    name, model_type, json.dumps(data), time.time(),
                    json.dumps(data), time.time(),
                ),
            )

    def load_model(self, name: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM models WHERE name = ?", (name,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["data"] = json.loads(d["data_json"])
            return d

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def log_event(self, event: str, detail: dict[str, Any] | None = None, job_id: int | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO history (job_id, event, detail_json) VALUES (?, ?, ?)",
                (job_id, event, json.dumps(detail) if detail else None),
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def rule_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rules")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def vacuum(self) -> None:
        assert self._conn is not None
        self._conn.execute("VACUUM")
        self._conn.commit()
        logger.info("Database VACUUM complete")
