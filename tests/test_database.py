"""Tests for ruleforge/database.py"""

import pytest
from ruleforge.database import Database


@pytest.fixture
def db():
    """In-memory database for tests."""
    database = Database(":memory:")
    yield database
    database.close()


class TestDatabase:
    def test_upsert_rule(self, db):
        rule_id = db.upsert_rule("l$1", fitness=0.5, origin="markov")
        assert rule_id > 0

    def test_upsert_rule_twice(self, db):
        db.upsert_rule("l$1", fitness=0.5)
        db.upsert_rule("l$1", fitness=0.8)  # should not raise
        row = db.get_rule("l$1")
        assert row is not None

    def test_get_rule(self, db):
        db.upsert_rule("l", fitness=1.0)
        row = db.get_rule("l")
        assert row is not None
        assert row["rule"] == "l"

    def test_get_rule_missing(self, db):
        assert db.get_rule("nonexistent_rule_xyz") is None

    def test_top_rules_by_fitness(self, db):
        db.upsert_rule("l", fitness=1.0)
        db.upsert_rule("u", fitness=2.0)
        db.upsert_rule("c", fitness=0.5)
        top = db.top_rules_by_fitness(2)
        assert len(top) == 2
        assert top[0]["fitness"] >= top[1]["fitness"]

    def test_bulk_insert_rules(self, db):
        rules = [("l$1", 0.5, "markov"), ("u", 0.3, "mutate"), ("c$!", 0.7, "template")]
        db.bulk_insert_rules(rules)
        assert db.rule_count() == 3

    def test_rule_count(self, db):
        assert db.rule_count() == 0
        db.upsert_rule("l")
        assert db.rule_count() == 1

    def test_save_statistics(self, db):
        stats = {
            "total_lines": 100,
            "valid_lines": 90,
            "invalid_lines": 10,
            "unique_count": 80,
            "entropy": 3.5,
            "mean_complexity": 2.1,
        }
        db.save_statistics(stats)  # should not raise

    def test_create_finish_job(self, db):
        job_id = db.create_job("test_job", {"workers": 4})
        assert job_id > 0
        db.finish_job(job_id, status="done")

    def test_save_load_model(self, db):
        model_data = {"order": 2, "vocab": ["l", "u"]}
        db.save_model("markov_model", "markov", model_data)
        loaded = db.load_model("markov_model")
        assert loaded is not None
        assert loaded["data"]["order"] == 2

    def test_load_model_missing(self, db):
        assert db.load_model("nonexistent") is None

    def test_log_event(self, db):
        db.log_event("test_event", {"detail": "value"})

    def test_upsert_template(self, db):
        db.upsert_template("lowercase → append", [["l", ""], ["$", "digit"]], 0.8)

    def test_vacuum(self, db):
        db.vacuum()  # should not raise
